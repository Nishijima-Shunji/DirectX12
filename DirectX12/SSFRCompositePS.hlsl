#include "SharedStruct.hlsli"

SamplerState g_LinearClamp : register(s0);

Texture2D<float>   g_FluidDepthTexture     : register(t0); // 平滑化済み流体深度
Texture2D<float4>  g_FluidNormalTexture    : register(t1); // ビュー空間法線
Texture2D<float>   g_FluidThicknessTexture : register(t2); // 粒子厚み
Texture2D<float>   g_SceneDepthTexture     : register(t3); // シーン深度バッファ（0〜1）
Texture2D<float4>  g_SceneColorTexture     : register(t4); // シーンカラー

struct PSInput
{
    float4 position : SV_POSITION;
    float2 uv       : TEXCOORD0;
};

int2 GetScreenExtent()
{
    return int2(screenSize);
}

float2 PixelToNDC(int2 pixel)
{
    float2 xy = (float2(pixel) + 0.5f) / screenSize;
    return float2(xy.x * 2.0f - 1.0f, 1.0f - xy.y * 2.0f);
}

float3 ReconstructViewPosition(int2 pixel, float depth)
{
    float2 ndc = PixelToNDC(pixel);
    float vx = ndc.x * depth / proj._11;
    float vy = ndc.y * depth / proj._22;
    return float3(vx, vy, depth);
}

// ※シーン深度をビュー空間へ揃えて流体深度との比較誤差を抑える
float LinearizeDepth(float deviceDepth)
{
    float denominator = farZ - deviceDepth * (farZ - nearZ);
    return (nearZ * farZ) / max(denominator, 1e-4f);
}

float3 DecodeNormal(float4 encodedNormal)
{
    float3 n = encodedNormal.xyz;
    float len = length(n);
    if (len < 1e-4f)
    {
        return float3(0.0f, 0.0f, -1.0f);
    }
    return n / len;
}

float ReconstructIOR()
{
    // F0 から屈折率を概算（RGB の平均を利用）
    float f0 = dot(iorF0, float3(0.333333f, 0.333333f, 0.333333f));
    float s = sqrt(saturate(f0));
    return (1.0f + s) / max(1.0f - s, 1e-3f);
}

float3 ApplyBeerLambert(float3 color, float thicknessValue)
{
    float3 absorption = exp(-absorb.xxx * thicknessValue);
    return color * absorption;
}

float3 ComputeFoam(float3 baseColor, float thicknessValue, float viewDotN)
{
    // 厚みが薄く、視線と法線のなす角が浅い場所を泡として明るくする
    float edgeFactor = saturate(1.0f - viewDotN);
    float thinFactor = saturate(exp(-thicknessValue * 2.0f));
    float foamStrength = edgeFactor * thinFactor;
    float3 foamColor = float3(0.95f, 0.97f, 1.0f);
    return lerp(baseColor, foamColor, foamStrength);
}

float4 main(PSInput input) : SV_TARGET
{
    float2 sceneUV = input.position.xy / framebufferSize;
    sceneUV = saturate(sceneUV); // 画面外サンプリングを防ぐ

    float2 fluidCoord = sceneUV * screenSize;
    fluidCoord = clamp(fluidCoord, float2(0.0f, 0.0f), screenSize - 1.0f);
    uint2 pixel = uint2(fluidCoord);
    int2 extent = GetScreenExtent();
    if (pixel.x >= extent.x || pixel.y >= extent.y)
    {
        return g_SceneColorTexture.SampleLevel(g_LinearClamp, sceneUV, 0);
    }

    float fluidDepth = g_FluidDepthTexture.Load(int3(pixel, 0));
    if (fluidDepth <= 0.0f)
    {
        // 流体が無い画素は背景をそのまま返す
        return g_SceneColorTexture.SampleLevel(g_LinearClamp, sceneUV, 0);
    }

    float sceneDepth = g_SceneDepthTexture.SampleLevel(g_LinearClamp, sceneUV, 0);
    float sceneViewDepth = LinearizeDepth(sceneDepth);
    if (sceneViewDepth <= fluidDepth - 1e-3f)
    {
        // 既存ジオメトリが手前にある場合は SSFR を重ねない
        return g_SceneColorTexture.SampleLevel(g_LinearClamp, sceneUV, 0);
    }

    float3 normal = DecodeNormal(g_FluidNormalTexture.Load(int3(pixel, 0)));
    if (normal.z > 0.0f)
    {
        normal *= -1.0f; // 常にカメラへ向けておく
    }

    float thickness = g_FluidThicknessTexture.Load(int3(pixel, 0));
    float3 viewPos = ReconstructViewPosition(int2(pixel), fluidDepth);
    float3 viewDir = normalize(-viewPos);
    float cosTheta = saturate(dot(normal, viewDir));

    float ior = ReconstructIOR();
    float3 fresnel = iorF0 + (1.0f - iorF0) * pow(1.0f - cosTheta, 5.0f);

    float3 sceneColor = g_SceneColorTexture.SampleLevel(g_LinearClamp, sceneUV, 0).rgb;

    float3 refractDir = refract(-viewDir, normal, 1.0f / ior);
    bool totalInternalReflection = dot(refractDir, refractDir) < 1e-5f;

    // 屈折オフセットは PDF の近似式を踏襲し、法線傾きと厚みで強さを調整
    float baseRefraction = 0.03f;
    float thicknessScale = saturate(thickness * 0.25f);
    float2 refractOffset = refractDir.xy / max(-refractDir.z, 0.1f) * (baseRefraction + thicknessScale * 0.02f);
    float2 refractUV = clamp(sceneUV + refractOffset, float2(0.0f, 0.0f), float2(1.0f, 1.0f));

    float3 refractedColor = g_SceneColorTexture.SampleLevel(g_LinearClamp, refractUV, 0).rgb;
    float3 transmitted = ApplyBeerLambert(refractedColor, thickness);
    float3 foamAdjusted = ComputeFoam(transmitted, thickness, cosTheta);

    float3 refractionContribution = totalInternalReflection ? float3(0.0f, 0.0f, 0.0f) : foamAdjusted;
    float3 reflectionContribution = sceneColor;

    // 反射と屈折をフレネル係数で合成し、指定 PDF に沿った水らしい見た目へ
    float3 color = lerp(refractionContribution, reflectionContribution, fresnel);

    return float4(color, 1.0f);
}
