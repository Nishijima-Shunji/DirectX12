struct VSOutput
{
    float4 svpos : SV_POSITION;
    float4 color : COLOR;
    float2 uv    : TEXCOORD;
};

SamplerState smp    : register(s0);
Texture2D    _MainTex : register(t0);

float4 pixel(VSOutput input) : SV_TARGET
{
    // テクスチャがバインドされていない場合は0が返るため、
    // アルファ値を用いて頂点カラーへフォールバックする
    float4 tex = _MainTex.Sample(smp, input.uv);
    return lerp(input.color, tex, tex.a);
}

