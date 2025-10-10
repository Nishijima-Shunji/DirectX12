#include "Common_SSFR.hlsli"

// 粒子位置SRVを直接読み取り、インスタンシングのみで深度用ビルボードを生成する。
StructuredBuffer<float3> gParticlePos : register(t8);

struct VSOutput
{
    float4 position : SV_Position;
    float3 viewPos  : TEXCOORD0;
    float2 uv       : TEXCOORD1;
};

static const float2 kCorners[4] = {
    float2(-1.0f, -1.0f),
    float2( 1.0f, -1.0f),
    float2( 1.0f,  1.0f),
    float2(-1.0f,  1.0f)
};

VSOutput VSMain(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    float3 worldPos = gParticlePos[instanceID];
    float4 viewPos4 = mul(float4(worldPos, 1.0f), gView);
    float3 viewPos = viewPos4.xyz;

    float2 corner = kCorners[vertexID & 3];
    float radiusPx = (gParticleRadius * gProj._22 / max(viewPos.z, 1e-4f)) * (gScreenSize.y * 0.5f);
    radiusPx = min(radiusPx, 50.0f);

    float4 clipCenter = mul(float4(viewPos, 1.0f), gProj);
    float2 clipOffset;
    clipOffset.x = corner.x * radiusPx * 2.0f * gInvScreenSize.x * clipCenter.w;
    clipOffset.y = -corner.y * radiusPx * 2.0f * gInvScreenSize.y * clipCenter.w;

    output.position = clipCenter;
    output.position.xy += clipOffset;
    output.viewPos = viewPos;
    output.uv = corner * 0.5f + 0.5f;
    return output;
}
