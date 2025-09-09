Texture2D<float> Src : register(t0);
SamplerState samp : register(s0);

cbuffer BlurCB : register(b0)
{
    float2 TexelSize;   // (1/W, 1/H)
    float2 Dir;         // (1,0)=â° or (0,1)=èc
};

struct PSIn
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

float main(PSIn i) : SV_Target
{
    // 5tapÉKÉEÉX
    const float w[5] = { 0.204164f, 0.304005f, 0.193783f, 0.072184f, 0.025864f };
    float2 o = Dir * TexelSize;

    float s = w[0] * Src.Sample(samp, i.uv);
    s += w[1] * (Src.Sample(samp, i.uv + 1 * o) + Src.Sample(samp, i.uv - 1 * o));
    s += w[2] * (Src.Sample(samp, i.uv + 2 * o) + Src.Sample(samp, i.uv - 2 * o));
    s += w[3] * (Src.Sample(samp, i.uv + 3 * o) + Src.Sample(samp, i.uv - 3 * o));
    s += w[4] * (Src.Sample(samp, i.uv + 4 * o) + Src.Sample(samp, i.uv - 4 * o));
    return s;
}
