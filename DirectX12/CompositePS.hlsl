#include "Common_SSFR.hlsli"

// フルスクリーン合成。流体深度と厚みから屈折・減衰・反射をまとめて評価する。
// gFlags: bit0=屈折, bit1=減衰, bit2=フレネル, bit3=シーンカラー有効, bit4=シーン深度有効, bit5=環境マップ有効。
struct VSOutput
{
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
};

// FXCがVSMainを参照するためエントリーポイント名を一致させる
VSOutput VSMain(uint vertexID : SV_VertexID)
{
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    VSOutput o;
    o.position = float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    o.uv = uv;
    return o;
}

Texture2D<float>      gFluidDepth    : register(t0);
Texture2D<float>      gThicknessTex  : register(t1);
Texture2D<float4>     gSceneColorTex : register(t2);
Texture2D<float>      gSceneDepthTex : register(t3);
TextureCube<float4>   gEnvCube       : register(t4);

static const float3 kF0 = float3(0.02f, 0.02f, 0.02f);

float3 ReconstructViewPosition(float depth, float2 uv)
{
    float2 ndc;
    ndc.x = uv.x * 2.0f - 1.0f;
    ndc.y = 1.0f - uv.y * 2.0f;
    float x = ndc.x * depth / gProj._11;
    float y = ndc.y * depth / gProj._22;
    return float3(x, y, depth);
}

float4 PSMain(VSOutput input) : SV_Target
{
    float2 uv = input.uv;
    float fluidDepth = gFluidDepth.SampleLevel(gSampLinearClamp, uv, 0.0f);
    if (fluidDepth <= 0.0f)
    {
        return gSceneColorTex.SampleLevel(gSampLinearClamp, uv, 0.0f);
    }

    float sceneDepth = gFarZ;
    if ((gFlags & 0x10u) != 0)
    {
        sceneDepth = gSceneDepthTex.SampleLevel(gSampLinearClamp, uv, 0.0f);
    }

    if (sceneDepth > 0.0f && sceneDepth < fluidDepth)
    {
        return gSceneColorTex.SampleLevel(gSampLinearClamp, uv, 0.0f);
    }

    float thickness = gThicknessTex.SampleLevel(gSampLinearClamp, uv, 0.0f);
    if (thickness <= 0.0f)
    {
        return gSceneColorTex.SampleLevel(gSampLinearClamp, uv, 0.0f);
    }

    float2 offset = gInvScreenSize * float2(1.0f, 0.0f);
    float depthR = gFluidDepth.SampleLevel(gSampLinearClamp, uv + offset, 0.0f);
    float depthL = gFluidDepth.SampleLevel(gSampLinearClamp, uv - offset, 0.0f);
    float depthU = gFluidDepth.SampleLevel(gSampLinearClamp, uv + float2(0.0f, -gInvScreenSize.y), 0.0f);
    float depthD = gFluidDepth.SampleLevel(gSampLinearClamp, uv + float2(0.0f, gInvScreenSize.y), 0.0f);
    depthR = depthR > 0.0f ? depthR : fluidDepth;
    depthL = depthL > 0.0f ? depthL : fluidDepth;
    depthU = depthU > 0.0f ? depthU : fluidDepth;
    depthD = depthD > 0.0f ? depthD : fluidDepth;

    float3 posC = ReconstructViewPosition(fluidDepth, uv);
    float3 posR = ReconstructViewPosition(depthR, uv + offset);
    float3 posL = ReconstructViewPosition(depthL, uv - offset);
    float3 posU = ReconstructViewPosition(depthU, uv + float2(0.0f, -gInvScreenSize.y));
    float3 posD = ReconstructViewPosition(depthD, uv + float2(0.0f, gInvScreenSize.y));

    float3 dx = posR - posL;
    float3 dy = posD - posU;
    float3 normal = normalize(cross(dx, dy));
    if (dot(normal, normal) < 1e-6f)
    {
        normal = float3(0.0f, 0.0f, 1.0f);
    }

    float3 viewDir = normalize(-posC);
    float cosTheta = saturate(dot(normal, viewDir));

    float3 sceneColor = ((gFlags & 0x8u) != 0) ? gSceneColorTex.SampleLevel(gSampLinearClamp, uv, 0.0f).rgb : gFluidColor;
    float3 refractColor = sceneColor;
    if ((gFlags & 0x1u) != 0)
    {
        float2 refractUV = uv + normal.xy * (gRefractScale * thickness);
        refractColor = gSceneColorTex.SampleLevel(gSampLinearClamp, refractUV, 0.0f).rgb;
    }

    if ((gFlags & 0x2u) != 0)
    {
        float3 absorb = exp(-gAbsorbK * thickness);
        refractColor *= absorb;
        refractColor += gFluidColor * (1.0f - absorb);
    }
    else
    {
        refractColor = lerp(refractColor, gFluidColor, 0.3f);
    }

    float3 reflection = gFluidColor;
    if ((gFlags & 0x4u) != 0 && (gFlags & 0x20u) != 0)
    {
        float3 reflectDir = reflect(-viewDir, normal);
        reflection = gEnvCube.SampleLevel(gSampLinearClamp, reflectDir, 0.0f).rgb;
    }

    float3 color = refractColor;
    if ((gFlags & 0x4u) != 0)
    {
        float3 fresnel = kF0 + (1.0f - kF0) * pow(1.0f - cosTheta, 5.0f);
        color = lerp(refractColor, reflection, fresnel);
    }

    return float4(color, 1.0f);
}
