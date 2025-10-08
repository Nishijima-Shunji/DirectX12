#include "SharedStruct.hlsli"

SamplerState g_LinearClamp : register(s0);

Texture2D<float>   g_FluidDepthTexture     : register(t0); // 平滑化済み流体深度（float 格納）
Texture2D<float4>  g_FluidNormalTexture    : register(t1); // ビュー空間法線
Texture2D<float>   g_FluidThicknessTexture : register(t2); // 粒子厚み（float 格納）
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

float ViewDepthFromDeviceDepth(float deviceDepth)
{
    return (nearZ * farZ) / max(farZ - deviceDepth * (farZ - nearZ), 1e-4f);
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
    // 合成パスではビューポートがフル解像度へ戻るため、半解像度バッファ用の画素座標へ変換する
    float2 ssfrCoord = (input.position.xy + 0.5f) * (screenSize / framebufferSize);
    uint2 pixel = uint2(ssfrCoord);
    float2 sceneUV = saturate((input.position.xy + 0.5f) / framebufferSize);
    int2 extent = GetScreenExtent();
    if (pixel.x >= extent.x || pixel.y >= extent.y)
    {
        return g_SceneColorTexture.SampleLevel(g_LinearClamp, sceneUV, 0);
    }

    float fluidDepth = g_FluidDepthTexture.Load(int3(pixel, 0));
    if (fluidDepth <= 0.0f || fluidDepth >= farZ - 1e-3f)
    {
        // 流体が存在しない画素は背景をそのまま返す
        return g_SceneColorTexture.SampleLevel(g_LinearClamp, sceneUV, 0);
    }

    float sceneDepth = g_SceneDepthTexture.Load(int3(pixel, 0));
    float sceneViewDepth = ViewDepthFromDeviceDepth(sceneDepth);

    if (sceneViewDepth <= fluidDepth - 1e-3f)
    {
        // シーンジオメトリが手前にある場合は上書きしない
        return g_SceneColorTexture.SampleLevel(g_LinearClamp, sceneUV, 0);
    }

    float3 normal = DecodeNormal(g_FluidNormalTexture.Load(int3(pixel, 0)));
    if (normal.z > 0.0f)
    {
        normal *= -1.0f; // カメラ側に向ける
    }

    float thickness = g_FluidThicknessTexture.Load(int3(pixel, 0));

    float3 viewPos = ReconstructViewPosition(int2(pixel), fluidDepth);
    float3 viewDir = normalize(-viewPos);

    float cosTheta = saturate(dot(normal, viewDir));

    float ior = ReconstructIOR();

    // Schlick 近似によるフレネル反射率
    float3 fresnel = iorF0 + (1.0f - iorF0) * pow(1.0f - cosTheta, 5.0f);

    // 屈折ベクトル（ビュー空間）を求め、スクリーンオフセットに変換
    float3 refractDir = refract(-viewDir, normal, 1.0f / ior);
    bool totalInternalReflection = all(abs(refractDir) < 1e-5f);
    float refractionScale = 0.05f;
    float2 refractOffset = refractDir.xy / max(refractDir.z, 0.1f) * refractionScale;
    float2 refractUV = clamp(sceneUV + refractOffset, float2(0.0f, 0.0f), float2(1.0f, 1.0f));

    float3 sceneColor = g_SceneColorTexture.SampleLevel(g_LinearClamp, sceneUV, 0).rgb;
    float3 refractedColor = g_SceneColorTexture.SampleLevel(g_LinearClamp, refractUV, 0).rgb;

    // 吸収による減衰を厚みに応じて適用
    float3 transmitted = ApplyBeerLambert(refractedColor, thickness);

    // 泡のハイライト
    float3 foamAdjusted = ComputeFoam(transmitted, thickness, cosTheta);

    // 全反射時は屈折成分を使わず背景を反射として利用
    float3 reflectionColor = sceneColor;
    float3 refractionContribution = totalInternalReflection ? float3(0.0f, 0.0f, 0.0f) : foamAdjusted;

    float3 color = lerp(refractionContribution, reflectionColor, fresnel);

    return float4(color, 1.0f);
}
