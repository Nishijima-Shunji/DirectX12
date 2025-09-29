cbuffer CameraCB : register(b0)
{
    float4x4 g_View;
    float4x4 g_Proj;
    float2   g_ScreenSize;
    float2   g_InvScreenSize;
    float    g_NearZ;
    float    g_FarZ;
    float3   g_IorF0;
    float    g_Absorb;
    uint4    g_Options;
};

Texture2D<float4> g_SceneColor : register(t0);
Texture2D<float>  g_SceneDepth : register(t1);
Texture2D<float>  g_FluidDepth : register(t2);
Texture2D<float>  g_FluidThickness : register(t3);
Texture2D<float4> g_FluidNormal : register(t4);
Texture2D<float4> g_SSRColor : register(t5);
SamplerState g_LinearClamp : register(s0);

struct PSIn
{
    float4 position : SV_POSITION;
    float2 uv       : TEXCOORD0;
};

float3 ApplyBeerLambert(float3 color, float thickness)
{
    float3 absorb = float3(g_Absorb, g_Absorb * 0.7f, g_Absorb * 0.5f);
    return color * exp(-absorb * thickness);
}

float3 GetViewDir(float2 uv)
{
    float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    float4 clip = float4(ndc, 1.0f, 1.0f);
    float4 view = mul(inverse(g_Proj), clip);
    view.xyz /= max(view.w, 1e-5f);
    return normalize(view.xyz);
}

float4 PSMain(PSIn input) : SV_TARGET
{
    float2 uv = input.uv;
    float sceneDepth = g_SceneDepth.SampleLevel(g_LinearClamp, uv, 0.0f);
    float fluidDepth = g_FluidDepth.SampleLevel(g_LinearClamp, uv, 0.0f);
    float thickness = g_FluidThickness.SampleLevel(g_LinearClamp, uv, 0.0f);
    float4 normalSample = g_FluidNormal.SampleLevel(g_LinearClamp, uv, 0.0f);
    float3 normal = normalize(normalSample.xyz * 2.0f - 1.0f);

    float4 sceneColor = g_SceneColor.SampleLevel(g_LinearClamp, uv, 0.0f);
    float4 ssrColor = g_SSRColor.SampleLevel(g_LinearClamp, uv, 0.0f);

    if (thickness <= 1e-3f)
    {
        return sceneColor;
    }

    float3 viewDir = normalize(-GetViewDir(uv));
    float cosTheta = saturate(dot(normal, viewDir));
    float3 fresnel = g_IorF0 + (1.0f - g_IorF0) * pow(1.0f - cosTheta, 5.0f);

    float2 refrUV = uv + normal.xy * 0.03f;
    float3 refrColor = g_SceneColor.SampleLevel(g_LinearClamp, refrUV, 0.0f).rgb;
    refrColor = ApplyBeerLambert(refrColor, thickness);

    float3 reflectColor = ssrColor.rgb;
    if (g_Options.z > 0.5f)
    {
        reflectColor = lerp(reflectColor, sceneColor.rgb, 0.3f);
    }

    float3 waterColor = lerp(refrColor, reflectColor, fresnel);

    if (g_Options.x == 1)
    {
        waterColor = saturate(fluidDepth / g_FarZ).xxx;
    }
    else if (g_Options.x == 2)
    {
        waterColor = saturate(thickness * 0.1f).xxx;
    }
    else if (g_Options.x == 3)
    {
        waterColor = normalize(normal) * 0.5f + 0.5f;
    }

    return float4(waterColor, 1.0f);
}
