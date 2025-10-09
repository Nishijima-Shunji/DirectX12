#include "SharedStruct.hlsli"

SamplerState g_LinearClamp : register(s0);

Texture2D<float>   g_SmoothedDepthTexture  : register(t0); // バイラテラルフィルタ後の線形深度
Texture2D<float>   g_FluidThicknessTexture : register(t1); // 粒子厚み積算
Texture2D<float4>  g_FluidNormalTexture    : register(t2); // スクリーンスペース法線
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

float LinearizeDepth(float deviceDepth)
{
    float denominator = farZ - deviceDepth * (farZ - nearZ);
    return (nearZ * farZ) / max(denominator, 1e-4f);
}

float SampleFluidDepth(int2 pixel)
{
    int2 extent = GetScreenExtent();
    pixel = clamp(pixel, int2(0, 0), extent - 1);
    float depth = g_SmoothedDepthTexture.Load(int3(pixel, 0));
    const float emptyDepth = 999999.0f;
    return (depth >= emptyDepth) ? 0.0f : depth;
}

float3 ComputeViewNormal(int2 pixel, float centerDepth)
{
    if (centerDepth <= 0.0f)
    {
        return float3(0.0f, 0.0f, -1.0f);
    }

    float depthL = SampleFluidDepth(pixel + int2(-1, 0));
    float depthR = SampleFluidDepth(pixel + int2(1, 0));
    float depthU = SampleFluidDepth(pixel + int2(0, -1));
    float depthD = SampleFluidDepth(pixel + int2(0, 1));

    if (depthL <= 0.0f) depthL = centerDepth;
    if (depthR <= 0.0f) depthR = centerDepth;
    if (depthU <= 0.0f) depthU = centerDepth;
    if (depthD <= 0.0f) depthD = centerDepth;

    float3 pC = ReconstructViewPosition(pixel, centerDepth);
    float3 pL = ReconstructViewPosition(pixel + int2(-1, 0), depthL);
    float3 pR = ReconstructViewPosition(pixel + int2(1, 0), depthR);
    float3 pU = ReconstructViewPosition(pixel + int2(0, -1), depthU);
    float3 pD = ReconstructViewPosition(pixel + int2(0, 1), depthD);

    float3 dx = pR - pL;
    float3 dy = pD - pU;

    float3 normal = normalize(cross(dy, dx));
    if (!all(isfinite(normal)))
    {
        normal = float3(0.0f, 0.0f, -1.0f);
    }
    return normal;
}

float ReconstructIOR()
{
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
    float edgeFactor = saturate(1.0f - viewDotN);
    float thinFactor = saturate(exp(-thicknessValue * 2.0f));
    float foamStrength = edgeFactor * thinFactor;
    float3 foamColor = float3(0.95f, 0.97f, 1.0f);
    return lerp(baseColor, foamColor, foamStrength);
}

float4 main(PSInput input) : SV_TARGET
{
    float2 sceneUV = input.position.xy / framebufferSize;
    sceneUV = saturate(sceneUV);

    float2 ratio = screenSize / framebufferSize;
    float2 halfResCoord = input.position.xy * ratio;
    halfResCoord = clamp(halfResCoord, float2(0.0f, 0.0f), screenSize - 1.0f);
    int2 pixel = int2(halfResCoord);
    int2 extent = GetScreenExtent();
    if (pixel.x >= extent.x || pixel.y >= extent.y)
    {
        return g_SceneColorTexture.SampleLevel(g_LinearClamp, sceneUV, 0);
    }

    float fluidDepth = SampleFluidDepth(pixel);
    if (fluidDepth <= 0.0f)
    {
        return g_SceneColorTexture.SampleLevel(g_LinearClamp, sceneUV, 0);
    }

    float sceneDepth = g_SceneDepthTexture.SampleLevel(g_LinearClamp, sceneUV, 0);
    float sceneViewDepth = LinearizeDepth(sceneDepth);
    if (sceneViewDepth <= fluidDepth - 1e-3f)
    {
        return g_SceneColorTexture.SampleLevel(g_LinearClamp, sceneUV, 0);
    }

    float3 viewPos = ReconstructViewPosition(pixel, fluidDepth);
    float3 viewDir = normalize(-viewPos);
    float4 normalSample = g_FluidNormalTexture.Load(int3(pixel, 0)); // ※CSで生成した法線を優先的に利用
    float3 normal = normalSample.xyz;
    if (normalSample.w < 0.5f)
    {
        normal = ComputeViewNormal(pixel, fluidDepth); // ※計算結果が無効な場合は深度勾配から再構築
    }
    normal = normalize(normal);
    if (!all(isfinite(normal)))
    {
        normal = float3(0.0f, 0.0f, -1.0f);
    }
    if (dot(normal, viewDir) < 0.0f)
    {
        normal *= -1.0f; // ※視線ベクトルと法線が逆向きだと屈折率計算で常に反射100%となり流体が見えないため矯正する
    }

    float thickness = g_FluidThicknessTexture.Load(int3(pixel, 0));
    float cosTheta = saturate(dot(normal, viewDir));

    float ior = ReconstructIOR();
    float3 fresnel = iorF0 + (1.0f - iorF0) * pow(1.0f - cosTheta, 5.0f);

    float3 refractDir = refract(-viewDir, normal, 1.0f / ior);
    bool totalInternalReflection = all(abs(refractDir) < 1e-4f);
    float2 refractOffset = refractDir.xy / max(refractDir.z, 0.1f) * refractionScale;
    float2 refractUV = clamp(sceneUV + refractOffset, float2(0.0f, 0.0f), float2(1.0f, 1.0f));

    float3 sceneColor = g_SceneColorTexture.SampleLevel(g_LinearClamp, sceneUV, 0).rgb;
    float3 refractedColor = g_SceneColorTexture.SampleLevel(g_LinearClamp, refractUV, 0).rgb;

    float3 transmitted = ApplyBeerLambert(refractedColor, thickness * thicknessScale);
    float3 foamAdjusted = ComputeFoam(transmitted, thickness * thicknessScale, cosTheta);

    float3 reflectionColor = sceneColor;
    float3 refractionContribution = totalInternalReflection ? float3(0.0f, 0.0f, 0.0f) : foamAdjusted;

    float3 color = lerp(refractionContribution, reflectionColor, fresnel);

    return float4(color, 1.0f);
}
