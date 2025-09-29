Texture2D<float> SrcTex : register(t0);
SamplerState samLinear : register(s0);

cbuffer BlurCB : register(b0)
{
    float2 texelSize; // 1/width, 1/height
    float2 dir; // (1,0)=, (0,1)=c
};

struct VSOut
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

float4 main(VSOut i) : SV_Target
{
    // 5tap KEVAiyʁj
    static const float w[5] = { 0.204164f, 0.304005f, 0.193783f, 0.072184f, 0.025864f };
    float2 o = dir * texelSize;
    // SampleLevel を使うと一部のターゲットで "texlod" 命令が生成されるためエラーになる。
    // このぼかし処理では通常の Sample で問題ないので、LOD 指定を外す。
    float v = SrcTex.Sample(samLinear, i.uv);
    v += SrcTex.Sample(samLinear, i.uv + o * 1) * w[1];
    v += SrcTex.Sample(samLinear, i.uv - o * 1) * w[1];
    v += SrcTex.Sample(samLinear, i.uv + o * 2) * w[2];
    v += SrcTex.Sample(samLinear, i.uv - o * 2) * w[2];
    v += SrcTex.Sample(samLinear, i.uv + o * 3) * w[3];
    v += SrcTex.Sample(samLinear, i.uv - o * 3) * w[3];
    v += SrcTex.Sample(samLinear, i.uv + o * 4) * w[4];
    v += SrcTex.Sample(samLinear, i.uv - o * 4) * w[4];
    return float4(v, 0, 0, 0);
}

//cbuffer BlurCB : register(b0)
//{
//    float2 texelSize;
//    float2 dir;
//}
//Texture2D<float> Src : register(t0);
//SamplerState samLinear : register(s0);

//float main(float2 uv : TEXCOORD) : SV_Target
//{
//    // 5-tap Gaussianij
//    const float w[5] = { 0.227027, 0.194595, 0.121622, 0.054054, 0.016216 };
//    float a = Src.SampleLevel(samLinear, uv, 0).r * w[0];
//    [unroll]
//    for (int k = 1; k < 5; ++k)
//    {
//        float2 off = dir * texelSize * k;
//        a += Src.SampleLevel(samLinear, uv + off, 0).r * w[k];
//        a += Src.SampleLevel(samLinear, uv - off, 0).r * w[k];
//    }
//    return a;
//}
