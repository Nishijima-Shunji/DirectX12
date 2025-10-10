struct VSOutput
{
    float4 svpos    : SV_POSITION; // クリップ座標
    float3 worldPos : TEXCOORD0;   // 世界座標
    float3 normal   : TEXCOORD1;   // 世界法線
    float4 color    : COLOR0;      // 色
    float2 uv       : TEXCOORD2;   // UV
};

float4 main(VSOutput input) : SV_TARGET
{
    // シンプルな環境光とディフューズで水面を表現
    float3 n = normalize(input.normal);
    float3 lightDir = normalize(float3(-0.35f, -1.0f, -0.3f));
    float diff = saturate(dot(n, -lightDir));

    float3 viewDir = normalize(float3(0.0f, 1.0f, 0.0f));
    float3 halfDir = normalize(-lightDir + viewDir);
    float spec = pow(saturate(dot(n, halfDir)), 64.0f);

    float3 baseColor = input.color.rgb;
    float3 lighting = baseColor * (0.35f + 0.65f * diff) + 0.25f * spec;
    return float4(lighting, input.color.a);
}
