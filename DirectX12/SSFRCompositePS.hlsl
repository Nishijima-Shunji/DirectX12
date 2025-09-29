#define PASS_COMPOSITE_PS
#include "SharedStruct.hlsli"

// フルスクリーン描画時の頂点シェーダー出力と整合する構造体
struct FullscreenTriangleOutput
{
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD0;
};

Texture2D<float> g_DepthTexture : register(t0);
Texture2D<float4> g_NormalTexture : register(t1);
Texture2D<float4> g_ParticleColorTexture : register(t2);

cbuffer SceneConstantBuffer : register(b0)
{
    matrix View;
    matrix Proj;
    matrix ViewProj;
    float3 CameraPos;
    uint FrameCount;
    float DeltaTime;
};


float4 main(FullscreenTriangleOutput input) : SV_TARGET
{
    float2 uv = input.uv;
    // テクスチャサイズからスクリーン座標を計算（定数バッファ依存を避ける）
    uint screenWidth, screenHeight;
    g_DepthTexture.GetDimensions(screenWidth, screenHeight);
    uint2 screen_pos = uint2(uv.x * screenWidth, uv.y * screenHeight);

    float depth = g_DepthTexture.Load(int3(screen_pos, 0));

    // 背景ピクセルはそのまま出力
    if (depth >= 1.0f)
    {
        return float4(0.25f, 0.25f, 0.25f, 1.0f); // 背景色
    }

    float3 normal = normalize(g_NormalTexture.Load(int3(screen_pos, 0)).xyz);
    float4 particle_color = g_ParticleColorTexture.Load(int3(screen_pos, 0));

    // ライティング
    float3 lightDir = normalize(float3(1, 1, -1));
    float3 viewDir = normalize(CameraPos - input.pos.xyz); // 仮のワールド座標
    
    // アンビエント光
    float3 ambient = float3(0.1, 0.2, 0.3);

    // ディフューズ光
    float diff = max(dot(normal, lightDir), 0.0);
    float3 diffuse = diff * float3(0.8, 0.8, 1.0);

    // スペキュラ光 (Blinn-Phong)
    float3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    float3 specular = spec * float3(1.0, 1.0, 1.0);

    // フレネル反射
    float fresnel = 0.04 + (1.0 - 0.04) * pow(1.0 - saturate(dot(normal, viewDir)), 5.0);

    // 水の厚みに基づく色の吸収 (Beer's Law)
    float thickness = particle_color.a; // alphaに厚み情報を格納しておくと良い
    float3 absorption = exp(-float3(0.1, 0.3, 0.4) * thickness);
    
    float3 water_base_color = float3(0.3, 0.6, 0.8) * absorption;

    // 最終的な色
    float3 final_color = water_base_color * (ambient + diffuse) + specular * fresnel;
    
    return float4(final_color, 1.0);
}
