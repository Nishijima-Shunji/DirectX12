#include "SharedStruct.hlsli"

Texture2D<float> g_RawDepthTexture        : register(t0); // 粒子スプラット直後の線形深度
RWTexture2D<float> g_SmoothedDepthTexture : register(u0); // 平滑化後の深度を出力

// スクリーンサイズを整数で扱う際の補助（画素外アクセス防止）
int2 GetScreenExtent()
{
    return int2(screenSize);
}

// 画素座標を NDC に変換する
float2 PixelToNDC(int2 pixel)
{
    float2 xy = (float2(pixel) + 0.5f) / screenSize;
    // DirectX はウィンドウ座標系で +Y が下向きなので NDC では符号を反転する
    return float2(xy.x * 2.0f - 1.0f, 1.0f - xy.y * 2.0f);
}

// 深度値（ビュー空間 z）と画素位置からビュー空間座標を逆算する
float3 ReconstructViewPosition(int2 pixel, float depth)
{
    float2 ndc = PixelToNDC(pixel);
    float vx = ndc.x * depth / proj._11;
    float vy = ndc.y * depth / proj._22;
    return float3(vx, vy, depth);
}

// 深度テクスチャから float を安全に読み取る（0 の場合は未描画と見なす）
float LoadDepth(int2 pixel)
{
    int2 extent = GetScreenExtent();
    pixel = clamp(pixel, int2(0, 0), extent - 1);
    float depth = g_RawDepthTexture.Load(int3(pixel, 0));
    const float emptyDepth = 999999.0f;
    // ※未描画領域はクリア値(1e6)で埋めているため 0.0f に置き換えて扱いやすくする
    return depth >= emptyDepth ? 0.0f : depth;
}

// 深度勾配から法線を推定する
float3 EstimateNormal(int2 pixel)
{
    float centerDepth = LoadDepth(pixel);
    if (centerDepth <= 0.0f)
    {
        return float3(0.0f, 0.0f, 0.0f);
    }

    float depthL = LoadDepth(pixel + int2(-1, 0));
    float depthR = LoadDepth(pixel + int2(1, 0));
    float depthU = LoadDepth(pixel + int2(0, -1));
    float depthD = LoadDepth(pixel + int2(0, 1));

    // 未描画画素は中心深度で補完して極端なノイズを防ぐ
    if (depthL <= 0.0f) depthL = centerDepth;
    if (depthR <= 0.0f) depthR = centerDepth;
    if (depthU <= 0.0f) depthU = centerDepth;
    if (depthD <= 0.0f) depthD = centerDepth;

    float3 pC = ReconstructViewPosition(pixel, centerDepth);
    float3 pL = ReconstructViewPosition(pixel + int2(-1, 0), depthL);
    float3 pR = ReconstructViewPosition(pixel + int2(1, 0), depthR);
    float3 pU = ReconstructViewPosition(pixel + int2(0, -1), depthU);
    float3 pD = ReconstructViewPosition(pixel + int2(0, 1), depthD);

    float3 dx = pR - pL;
    float3 dy = pD - pU;

    float3 n = normalize(cross(dy, dx));
    if (!all(isfinite(n)))
    {
        n = float3(0.0f, 0.0f, -1.0f);
    }
    return n;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        // 未描画領域はそのままコピー（0 を維持）
        g_SmoothedDepthTexture[pixel] = 0.0f;
        return;
    }

    float3 centerNormal = EstimateNormal(int2(pixel));

    float weightSum = 0.0f;
    float depthSum = 0.0f;

    // ※係数は b0 へ集約し、ルートシグネチャの CBV スロット競合を解消
    float sigmaS = max(bilateralSigma.x, 1e-4f);
    float sigmaD = max(bilateralSigma.y, 1e-4f);
    float sigmaN = max(bilateralNormalKernel.x, 1e-4f);
    int radius = max(int(bilateralNormalKernel.y + 0.5f), 0); // ※浮動小数で渡された半径を整数へ丸める

    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            int2 offset = int2(x, y);
            int2 samplePixel = clamp(int2(pixel) + offset, int2(0, 0), extent - 1);

            float sampleDepth = LoadDepth(samplePixel);
            if (sampleDepth <= 0.0f)
            {
                continue;
            }

            float3 sampleNormal = EstimateNormal(samplePixel);

            float spatialWeight = exp(-dot(offset, offset) / (2.0f * sigmaS * sigmaS));
            float depthDiff = sampleDepth - centerDepth;
            float depthWeight = exp(-(depthDiff * depthDiff) / (2.0f * sigmaD * sigmaD));
            float normalDot = saturate(dot(centerNormal, sampleNormal));
            float normalWeight = pow(normalDot, sigmaN);

            float weight = spatialWeight * depthWeight * normalWeight;
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
