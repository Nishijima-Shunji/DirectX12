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
Texture2D<float4> g_Normal     : register(t2);
SamplerState g_LinearClamp : register(s0);

struct PSIn
{
    float4 position : SV_POSITION;
    float2 uv       : TEXCOORD0;
};

float3 ReconstructPosition(float2 uv, float depth)
{
    float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    float4 clip = float4(ndc, 1.0f, 1.0f);
    float4 view = mul(inverse(g_Proj), clip);
    view.xyz /= max(view.w, 1e-5f);
    return normalize(view.xyz) * depth;
}

float4 PSMain(PSIn input) : SV_TARGET
{
    float2 uv = input.uv;
    float depth = g_SceneDepth.SampleLevel(g_LinearClamp, uv, 0.0f);
    if (depth >= g_FarZ - 1e-3f)
    {
        return float4(0, 0, 0, 1);
    }

    float3 normal = normalize(g_Normal.SampleLevel(g_LinearClamp, uv, 0.0f).xyz * 2.0f - 1.0f);
    float3 viewPos = ReconstructPosition(uv, depth);
    float3 viewDir = normalize(-viewPos);
    float3 reflectDir = normalize(reflect(viewDir, normal));

    float3 origin = viewPos;
    float stepLength = 1.0f;
    uint maxSteps = max(8u, g_Options.y * 8u);
    float3 hitColor = float3(0.0f, 0.0f, 0.0f);
    bool hit = false;

    [loop]
    for (uint step = 0; step < maxSteps; ++step)
    {
        float travel = (step + 1) * stepLength;
        float3 samplePos = origin + reflectDir * travel;
        if (samplePos.z < g_NearZ || samplePos.z > g_FarZ)
        {
            break;
        }

        float4 proj = mul(g_Proj, float4(samplePos, 1.0f));
        float2 sampleUV = proj.xy / proj.w * float2(0.5f, -0.5f) + float2(0.5f, 0.5f);
        if (sampleUV.x < 0.0f || sampleUV.x > 1.0f || sampleUV.y < 0.0f || sampleUV.y > 1.0f)
        {
            continue;
        }

        float sceneDepth = g_SceneDepth.SampleLevel(g_LinearClamp, sampleUV, 0.0f);
        float depthDiff = sceneDepth - samplePos.z;
        if (abs(depthDiff) < 0.8f)
        {
            hitColor = g_SceneColor.SampleLevel(g_LinearClamp, sampleUV, 0.0f).rgb;
            hit = true;
            break;
        }
    }

    if (!hit)
    {
        float3 envColor = float3(0.2f, 0.35f, 0.45f);
        return float4(envColor, 1.0f);
    }

    return float4(hitColor, 1.0f);
}
