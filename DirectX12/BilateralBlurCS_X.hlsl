#include "Common_SSFR.hlsli"

// X方向の両側ブラー。深度差に応じてフィルタ強度を抑え、粒子境界の破綻を防ぐ。
Texture2D<float>      gInput  : register(t0);
RWTexture2D<float>    gOutput : register(u0);

[numthreads(128, 1, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 coord = dispatchThreadID.xy;
    float depthCenter = gInput.Load(int3(coord, 0));
    if (depthCenter <= 0.0f)
    {
        gOutput[coord] = depthCenter;
        return;
    }

    float blurRadiusPx = (gWorldBlurRadius * gProj._22 / max(depthCenter, 1e-4f)) * (gScreenSize.y * 0.5f);
    blurRadiusPx = clamp(blurRadiusPx, 1.0f, 50.0f);
    int radius = (int)ceil(blurRadiusPx);
    float sigmaDist = blurRadiusPx * 0.5f;
    float twoSigma2 = 2.0f * sigmaDist * sigmaDist;

    float weightSum = 0.0f;
    float depthSum = 0.0f;

    for (int dx = -radius; dx <= radius; ++dx)
    {
        uint2 sampleCoord = uint2(clamp(int(coord.x) + dx, 0, int(gScreenSize.x) - 1), coord.y);
        float depthSample = gInput.Load(int3(sampleCoord, 0));
        float dist = abs((float)dx);
        float w = exp(-dist * dist / twoSigma2);
        if (gEnableBilateral != 0)
        {
            float diff = depthSample - depthCenter;
            float depthW = exp(-diff * diff / max(gDepthSigma * gDepthSigma * 2.0f, 1e-6f));
            w *= depthW;
        }
        weightSum += w;
        depthSum += depthSample * w;
    }

    gOutput[coord] = depthSum / max(weightSum, 1e-6f);
}
