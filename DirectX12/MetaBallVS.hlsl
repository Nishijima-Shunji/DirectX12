// register(b0) は後で ConstantBuffer に紐づけ
cbuffer ScreenCB : register(b0)
{
    float2 screenSize;   // (幅, 高さ)
    float threshold;     // メタボール閾値
    float pad0, pad1;
};

// t0 に StructuredBuffer<float4> パーティクル配列をバインド
StructuredBuffer<float4> Particles : register(t0);
// float4(x_ndc, y_ndc, radius_ndc, unused)

struct VSInput
{
    float2 pos : POSITION;  // {-1,1},{3,1},{-1,-3} の三角
};

struct VSOutput
{
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD;
};

VSOutput VS_Main(VSInput IN)
{
    VSOutput OUT;
    OUT.pos = float4(IN.pos, 0, 1);
    // UV は 0→1 スペースにマップ
    OUT.uv = IN.pos * 0.5 + float2(0.5,0.5);
    return OUT;
}
