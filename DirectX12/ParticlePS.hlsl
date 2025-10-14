struct VSOutput
{
    float4 svpos   : SV_POSITION;
    float3 normal  : NORMAL;
    float3 worldPos : TEXCOORD0;
};

// エントリーポイントをmainに統一して描画パスを共通化
float4 main(VSOutput input) : SV_TARGET
{
    const float3 lightDir = normalize(float3(0.3f, 0.8f, 0.4f)); // 粒子を立体的に見せる簡易平行光源
    const float3 baseColor = float3(0.2f, 0.6f, 1.0f);

    float diffuse = saturate(dot(normalize(input.normal), lightDir));
    float ambient = 0.25f;
    float lighting = ambient + diffuse * 0.75f;

    return float4(baseColor * lighting, 1.0f);
}
