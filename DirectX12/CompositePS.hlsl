Texture2D<float> Density : register(t0);
SamplerState samp : register(s0);

cbuffer CompositeCB : register(b0)
{
    float Threshold;    // 0.25
    float Softness;     // 0.1 （smoothstep用）
    float3 BaseColor;   // 水色
    float Opacity;      // 1.0
};

struct PSIn
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

float4 main(PSIn i) : SV_Target
{
    float d = Density.Sample(samp, i.uv);
    // smooth threshold
    float a = smoothstep(Threshold - Softness, Threshold + Softness, d);
    return float4(BaseColor, a * Opacity);
}
