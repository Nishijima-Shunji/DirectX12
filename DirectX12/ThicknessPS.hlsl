#include "Common_SSFR.hlsli"

// 厚みレンダリングおよびブラー。Beer-Lambert計算の信頼性向上が目的。
struct PSInput
{
    float4 position : SV_Position;
    float3 viewPos  : TEXCOORD0;
    float2 uv       : TEXCOORD1;
};

float PSMain(PSInput input) : SV_Target0
{
    float2 disc = input.uv * 2.0f - 1.0f;
    float r2 = dot(disc, disc);
    if (r2 > 1.0f)
    {
        discard;
    }

    float radius = gParticleRadius;
    float planar2 = r2 * radius * radius;
    float thickness = 2.0f * sqrt(max(radius * radius - planar2, 0.0f));
    return thickness;
}

// --- 以下は厚みブラー用コンピュート ---
Texture2D<float>      gThicknessIn  : register(t0);
RWTexture2D<float>    gThicknessOut : register(u0);

[numthreads(128, 1, 1)]
void ThicknessBlurCS_X(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 coord = dispatchThreadID.xy;
    float center = gThicknessIn.Load(int3(coord, 0));
    float radiusPx = (gWorldBlurRadius * gProj._22 / max(center + 1e-4f, 1e-4f)) * (gScreenSize.y * 0.5f);
    radiusPx = clamp(radiusPx, 1.0f, 20.0f);
    int radius = (int)ceil(radiusPx);
    float sigma = radiusPx * 0.6f;
    float invTwoSigma2 = 1.0f / max(2.0f * sigma * sigma, 1e-4f);

    float accum = 0.0f;
    float weight = 0.0f;
    for (int dx = -radius; dx <= radius; ++dx)
    {
        uint2 sampleCoord = uint2(clamp(int(coord.x) + dx, 0, int(gScreenSize.x) - 1), coord.y);
        float v = gThicknessIn.Load(int3(sampleCoord, 0));
        float w = exp(-(dx * dx) * invTwoSigma2);
        accum += v * w;
        weight += w;
    }
    gThicknessOut[coord] = accum / max(weight, 1e-4f);
}

[numthreads(128, 1, 1)]
void ThicknessBlurCS_Y(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 coord = dispatchThreadID.xy;
    float center = gThicknessIn.Load(int3(coord, 0));
    float radiusPx = (gWorldBlurRadius * gProj._22 / max(center + 1e-4f, 1e-4f)) * (gScreenSize.y * 0.5f);
    radiusPx = clamp(radiusPx, 1.0f, 20.0f);
    int radius = (int)ceil(radiusPx);
    float sigma = radiusPx * 0.6f;
    float invTwoSigma2 = 1.0f / max(2.0f * sigma * sigma, 1e-4f);

    float accum = 0.0f;
    float weight = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy)
    {
        uint2 sampleCoord = uint2(coord.x, clamp(int(coord.y) + dy, 0, int(gScreenSize.y) - 1));
        float v = gThicknessIn.Load(int3(sampleCoord, 0));
        float w = exp(-(dy * dy) * invTwoSigma2);
        accum += v * w;
        weight += w;
    }
    gThicknessOut[coord] = accum / max(weight, 1e-4f);
}
