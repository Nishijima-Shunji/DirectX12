#define PASS_NORMAL_CS
#include "SharedStruct.hlsli"

Texture2D<float> g_DepthTexture : register(t0);
RWTexture2D<float4> g_NormalTexture : register(u0);

cbuffer SceneConstantBuffer : register(b0)
{
    matrix View;
    matrix Proj;
    matrix ViewProj;
    float3 CameraPos;
    uint FrameCount;
    float DeltaTime;
};


// 3x3のSobelフィルタで法線を計算
float3 reconstructNormal(uint2 id)
{
    float d0 = g_DepthTexture.Load(int3(id.x - 1, id.y - 1, 0)).r;
    float d1 = g_DepthTexture.Load(int3(id.x, id.y - 1, 0)).r;
    float d2 = g_DepthTexture.Load(int3(id.x + 1, id.y - 1, 0)).r;

    float d3 = g_DepthTexture.Load(int3(id.x - 1, id.y, 0)).r;
    float d4 = g_DepthTexture.Load(int3(id.x, id.y, 0)).r;
    float d5 = g_DepthTexture.Load(int3(id.x + 1, id.y, 0)).r;

    float d6 = g_DepthTexture.Load(int3(id.x - 1, id.y + 1, 0)).r;
    float d7 = g_DepthTexture.Load(int3(id.x, id.y + 1, 0)).r;
    float d8 = g_DepthTexture.Load(int3(id.x + 1, id.y + 1, 0)).r;
    
    // Sobelフィルタ
    float dz_dx = (d2 + 2.0f * d5 + d8) - (d0 + 2.0f * d3 + d6);
    float dz_dy = (d6 + 2.0f * d7 + d8) - (d0 + 2.0f * d1 + d2);

    float3 n = float3(-dz_dx, -dz_dy, 0.001f);
    return normalize(n);
}

[numthreads(32, 32, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 id = DTid.xy;
    float depth = g_DepthTexture.Load(int3(id, 0)).r;

    if (depth >= 1.0f)
    {
        g_NormalTexture[id] = float4(0, 0, 0, 0);
        return;
    }
    
    float3 normal = reconstructNormal(id);
    g_NormalTexture[id] = float4(normal, 1.0f);
}