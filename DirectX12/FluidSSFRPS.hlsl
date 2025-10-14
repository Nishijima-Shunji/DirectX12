// SSFR風の半透明表現で液体らしさを出す簡易ピクセルシェーダー。
cbuffer Transform : register(b0)
{
    matrix World;
    matrix View;
    matrix Proj;
};

struct VSOutput
{
    float4 svpos   : SV_POSITION;
    float3 normal  : NORMAL;
    float3 worldPos : TEXCOORD0;
};

float SchlickFresnel(float cosTheta, float bias, float scale, float power)
{
    return bias + scale * pow(1.0 - cosTheta, power);
}

// エントリーポイントをmainに統一してレンダリングパイプラインと整合
float4 main(VSOutput input) : SV_TARGET
{
    float3 N = normalize(input.normal);
    float3 V = normalize(-input.worldPos);
    float3 L = normalize(float3(0.2f, 0.8f, 0.3f));
    float3 H = normalize(L + V);

    float diffuse = saturate(dot(N, L));
    float spec = pow(saturate(dot(N, H)), 32.0f);

    float fres = SchlickFresnel(saturate(dot(N, V)), 0.02f, 0.98f, 5.0f);
    float thickness = saturate(1.0f - (input.worldPos.y * 0.15f));

    float3 transmission = float3(0.1f, 0.45f, 0.7f) * thickness;
    float3 reflection = float3(0.6f, 0.85f, 1.0f) * fres;
    float3 lighting = transmission * (0.2f + diffuse * 0.7f) + reflection * spec;

    float alpha = saturate(0.35f + thickness * 0.4f + fres * 0.25f);
    return float4(lighting, alpha);
}
