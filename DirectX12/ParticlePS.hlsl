struct VSOutput
{
    float4 svpos : SV_POSITION;
    float2 uv    : TEXCOORD0;
};

float4 PSMain(VSOutput input) : SV_TARGET
{
    // ※UV を円形フェードに変換してアルファを作成
    float2 centered = input.uv * 2.0f - 1.0f;
    float dist = dot(centered, centered);
    float alpha = saturate(1.0f - dist);

    float3 baseColor = float3(0.2f, 0.6f, 1.0f);
    return float4(baseColor, alpha);
}
