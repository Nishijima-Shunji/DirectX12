struct VSOutput
{
    float4 svpos : SV_POSITION;
    float4 color : COLOR;
    float2 uv : TEXCOORD;
};

// Gg[|Cgmainɍ킹PSOƓ
float4 main(VSOutput input) : SV_TARGET
Texture2D _MainTex : register(t0); // ƒeƒNƒXƒ`ƒƒ

float4 pixel(VSOutput input) : SV_TARGET
{
    return _MainTex.Sample(smp, input.uv);
}
