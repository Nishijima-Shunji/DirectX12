#include "SharedStruct.hlsli"

Texture2D<float> g_SmoothedDepthTexture : register(t0);
RWTexture2D<float4> g_FluidNormalTexture : register(u0);

static const float CLEAR_DEPTH_VALUE = 1000000.0f; // 深度テクスチャ初期化値

// ヘルパー関数群はバイラテラルフィルタ版と揃えて扱う
int2 GetScreenExtent()
{
    return int2(screenSize);
}

float2 PixelToNDC(int2 pixel)
{
    float2 xy = (float2(pixel) + 0.5f) / screenSize;
    return float2(xy.x * 2.0f - 1.0f, 1.0f - xy.y * 2.0f);
}

float LoadDepth(int2 pixel)
{
    int2 extent = GetScreenExtent();
    if (any(pixel < int2(0, 0)) || any(pixel >= extent))
    {
        return 0.0f;
    }

    float depth = g_SmoothedDepthTexture.Load(int3(pixel, 0));
    return depth >= CLEAR_DEPTH_VALUE ? 0.0f : depth;
}

float3 ReconstructViewPosition(int2 pixel, float depth)
{
    float2 ndc = PixelToNDC(pixel);
    float vx = ndc.x * depth / proj._11;
    float vy = ndc.y * depth / proj._22;
    return float3(vx, vy, depth);
}

float3 ReconstructNormal(int2 pixel)
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

    if (depthL <= 0.0f) depthL = centerDepth;
    if (depthR <= 0.0f) depthR = centerDepth;
    if (depthU <= 0.0f) depthU = centerDepth;
    if (depthD <= 0.0f) depthD = centerDepth;

    float3 pC = ReconstructViewPosition(pixel, centerDepth);
    float3 pL = ReconstructViewPosition(pixel + int2(-1, 0), depthL);
    float3 pR = ReconstructViewPosition(pixel + int2(1, 0), depthR);
    float3 pU = ReconstructViewPosition(pixel + int2(0, -1), depthU);
    float3 pD = ReconstructViewPosition(pixel + int2(0, 1), depthD);

    // 2 辺差分を交差させて法線を復元する
    float3 dx = pR - pL;
    float3 dy = pD - pU;

    float3 normal = normalize(cross(dy, dx));
    if (!all(isfinite(normal)))
    {
        normal = float3(0.0f, 0.0f, -1.0f);
    }

    // ビュー空間では +Z が奥向きなので、常にカメラ側を向ける
    if (normal.z > 0.0f)
    {
        normal *= -1.0f;
    }

    return normal;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    int2 extent = GetScreenExtent();
    if (pixel.x >= extent.x || pixel.y >= extent.y)
    {
        return;
    }

    float depth = LoadDepth(int2(pixel));
    if (depth <= 0.0f)
    {
        g_FluidNormalTexture[pixel] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float3 normal = ReconstructNormal(int2(pixel));
    g_FluidNormalTexture[pixel] = float4(normal, 1.0f);
}
