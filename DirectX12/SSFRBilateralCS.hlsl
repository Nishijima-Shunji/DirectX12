#include "SharedStruct.hlsli"

Texture2D<float> g_RawDepthTexture        : register(t0); // 粒子スプラット直後の線形深度
RWTexture2D<float> g_SmoothedDepthTexture : register(u0); // 平滑化後の深度を出力

static const float CLEAR_DEPTH_VALUE = 1000000.0f; // クリア時に書き込む遠方値

// スクリーンサイズを整数で扱い、アクセス範囲を統一する
int2 GetScreenExtent()
{
    return int2(screenSize);
}

// 深度テクスチャから値を安全に読み取る（未描画は 0 とする）
float LoadDepth(int2 pixel)
{
    int2 extent = GetScreenExtent();
    if (any(pixel < int2(0, 0)) || any(pixel >= extent))
    {
        return 0.0f;
    }

    float depth = g_RawDepthTexture.Load(int3(pixel, 0));
    return depth >= CLEAR_DEPTH_VALUE ? 0.0f : depth;
}

// バイラテラルフィルタの重みを計算する（GDC10 の手法に倣う）
float ComputeWeight(int2 offset, float depthDiff, float sigmaSpatial, float sigmaDepth, float depthCutoff)
{
    float dist2 = dot(offset, offset);
    float spatial = exp(-dist2 / (2.0f * sigmaSpatial * sigmaSpatial));
    float depthTerm = exp(-(depthDiff * depthDiff) / (2.0f * sigmaDepth * sigmaDepth));

    // 深度差が大き過ぎるサンプルは混入させない
    if (depthCutoff > 0.0f && abs(depthDiff) > depthCutoff)
    {
        return 0.0f;
    }

    return spatial * depthTerm;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    int2 extent = GetScreenExtent();
    if (pixel.x >= extent.x || pixel.y >= extent.y)
    {
        return; // 画面外は何もしない
    }

    float centerDepth = LoadDepth(int2(pixel));
    if (centerDepth <= 0.0f)
    {
        // 未描画領域はそのまま 0 を保持する
        g_SmoothedDepthTexture[pixel] = 0.0f;
        return;
    }

    float sigmaSpatial = max(bilateralSigma.x, 1e-4f);
    float sigmaDepth = max(bilateralSigma.y, 1e-4f);
    int radius = max(int(round(bilateralKernel.x)), 0);
    float depthCutoff = max(bilateralKernel.y, 0.0f);

    float weightSum = 0.0f;
    float depthSum = 0.0f;

    // 近傍を円形に走査し、深度差のあるサンプルを抑制する
    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            int2 offset = int2(x, y);
            float sampleDepth = LoadDepth(int2(pixel) + offset);
            if (sampleDepth <= 0.0f)
            {
                continue;
            }

            float depthDiff = sampleDepth - centerDepth;
            float weight = ComputeWeight(offset, depthDiff, sigmaSpatial, sigmaDepth, depthCutoff);
            if (weight <= 0.0f)
            {
                continue;
            }

            depthSum += sampleDepth * weight;
            weightSum += weight;
        }
    }

    float filteredDepth = centerDepth;
    if (weightSum > 0.0f)
    {
        filteredDepth = depthSum / weightSum;
    }

    g_SmoothedDepthTexture[pixel] = filteredDepth;
}
