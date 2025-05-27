cbuffer ScreenCB : register(b0)
{
    float2 screenSize;
    float threshold;
    uint   particleCount;
    float pad0;
};

struct VSOutput
{
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD;
};

StructuredBuffer<float4> Particles : register(t0);

float4 PS_Main(VSOutput IN) : SV_TARGET
{
    float2 uv = IN.uv;
    float sum = 0;

    // [unroll] を外して動的ループに
    for (uint i = 0; i < particleCount; ++i)
    {
        float2 p = Particles[i].xy;
        float  r = Particles[i].z;
        float  d = distance(uv, p);
        sum += saturate(1 - d/r);
    }

    float alpha = smoothstep(threshold, 1.0, sum);
    clip(alpha - 0.01);
    return float4(0.2, 0.4, 1.0, alpha);
}
