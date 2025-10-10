#include "Common_SSFR.hlsli"

// 球前面の線形深度[m]のみを書き込む。MLS-MPM粒子の厚みは後段で加算合成。
struct PSInput
{
    float4 position : SV_Position;
    float3 viewPos  : TEXCOORD0;
    float2 uv       : TEXCOORD1;
};

float main(PSInput input) : SV_Target0
{
    float2 disc = input.uv * 2.0f - 1.0f;
    float r2 = dot(disc, disc);
    if (r2 > 1.0f)
    {
        discard;
    }

    float radius = gParticleRadius;
    float planar2 = r2 * radius * radius;
    float depthOffset = sqrt(max(radius * radius - planar2, 0.0f));
    float depthView = input.viewPos.z - depthOffset;
    return depthView;
}
