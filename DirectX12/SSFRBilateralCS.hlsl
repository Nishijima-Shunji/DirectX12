#include "SharedStruct.h"

Texture2D<float> g_DepthTexture : register(t0);
Texture2D<float4> g_NormalTexture : register(t1);
RWTexture2D<float4> g_FilteredNormalTexture : register(u0);

cbuffer BlurConstantBuffer : register(b0)
{
    float sigma; // 空間の重み
    float k_d; // 深度の重み
    float k_n; // 法線の重み
};


[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    int2 id = DTid.xy;
    
    float center_depth = g_DepthTexture.Load(int3(id, 0));
    float3 center_normal = g_NormalTexture.Load(int3(id, 0)).xyz;
    
    // 何も描画されていないピクセルはスキップ
    if (center_depth >= 1.0f)
    {
        g_FilteredNormalTexture[id] = float4(0, 0, 0, 0);
        return;
    }
    
    float total_weight = 0.0f;
    float3 filtered_normal = float3(0, 0, 0);

    const int radius = 5;
    
    [unroll]
    for (int y = -radius; y <= radius; ++y)
    {
        [unroll]
        for (int x = -radius; x <= radius; ++x)
        {
            int2 offset = int2(x, y);
            int2 sample_pos = id + offset;

            float sample_depth = g_DepthTexture.Load(int3(sample_pos, 0));
            float3 sample_normal = g_NormalTexture.Load(int3(sample_pos, 0)).xyz;
            
            // 空間的な重み (ガウシアン)
            float weight_s = exp(-(dot(offset, offset)) / (2.0f * sigma * sigma));
            
            // 深度の差による重み
            float depth_diff = abs(center_depth - sample_depth);
            float weight_d = exp(-(depth_diff * depth_diff) / (2.0f * k_d * k_d));
            
            // 法線の差による重み
            float normal_diff = dot(center_normal, sample_normal);
            float weight_n = pow(max(0.0, normal_diff), k_n);

            float weight = weight_s * weight_d * weight_n;

            filtered_normal += sample_normal * weight;
            total_weight += weight;
        }
    }
    
    if (total_weight > 0.0f)
    {
        g_FilteredNormalTexture[id] = float4(normalize(filtered_normal), 1.0f);
    }
    else
    {
        g_FilteredNormalTexture[id] = float4(center_normal, 1.0f);
    }
}