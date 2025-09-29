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

cbuffer FluidParams : register(b1)
{
    float g_Radius;
    float g_ThicknessScale;
    uint  g_ParticleCount;
    float g_Downsample;
};

struct ParticleInstance
{
    float3 position;
    float  radius;
};
StructuredBuffer<ParticleInstance> g_Particles : register(t0);
SamplerState g_LinearClamp : register(s0);

struct VSOut
{
    float4 position : SV_POSITION;
    float2 uv       : TEXCOORD0;
};

struct PSOut
{
    float thickness : SV_Target0;
    float depth     : SV_Target1;
};

float3 ReconstructRay(float2 uv)
{
    float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    float4 clip = float4(ndc, 1.0f, 1.0f);
    float4 view = mul(inverse(g_Proj), clip);
    view.xyz /= max(view.w, 1e-5f);
    return normalize(view.xyz);
}

PSOut PSMain(VSOut input)
{
    PSOut o;
    o.thickness = 0.0f;
    float minDepth = g_FarZ;

    float3 rayDir = ReconstructRay(input.uv);
    float3 rayOrigin = float3(0.0f, 0.0f, 0.0f);

    [loop]
    for (uint i = 0; i < g_ParticleCount; ++i)
    {
        ParticleInstance instance = g_Particles[i];
        float4 viewPos4 = mul(g_View, float4(instance.position, 1.0f));
        float3 viewPos = viewPos4.xyz;

        float radius = max(instance.radius, g_Radius);
        float3 toCenter = rayOrigin - viewPos;
        float b = dot(toCenter, rayDir);
        float c = dot(toCenter, toCenter) - radius * radius;
        float discriminant = b * b - c;
        if (discriminant < 0.0f)
        {
            continue;
        }

        float sqrtDisc = sqrt(discriminant);
        float t0 = -b - sqrtDisc;
        float t1 = -b + sqrtDisc;
        if (t1 <= 0.0f)
        {
            continue;
        }

        float nearT = max(t0, 0.0f);
        float farT = max(t1, 0.0f);
        float segment = max(farT - nearT, 0.0f);
        o.thickness += segment;
        minDepth = min(minDepth, nearT);
    }

    o.thickness *= g_ThicknessScale;
    o.thickness = min(o.thickness, 50.0f);
    o.depth = (minDepth >= g_FarZ) ? g_FarZ : minDepth;
    return o;
}
