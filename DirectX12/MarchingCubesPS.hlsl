// マーチングキューブのサーフェスをシンプルなライティングで描画するピクセルシェーダー
struct PSInput
{
    float4 svpos : SV_POSITION;
    float3 normal : NORMAL;
    float3 worldPos : TEXCOORD0;
};

float4 main(PSInput input) : SV_TARGET
{
    float3 lightDir = normalize(float3(0.3f, 0.8f, 0.4f));
    float3 baseColor = float3(0.2f, 0.6f, 1.0f);
    float diffuse = saturate(dot(normalize(input.normal), lightDir));
    float ambient = 0.25f;
    float lighting = ambient + diffuse * 0.75f;
    return float4(baseColor * lighting, 1.0f);
}
