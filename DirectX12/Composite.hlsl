
cbuffer CompositeCB : register(b0)
{
    float iso; // 例: 0.35f  (厚みのしきい値)
    float gain; // 例: 6.0f   (立ち上がりの強さ)
    float2 invSize; // 1/width, 1/height (SSA解像度)
};

Texture2D<float> Accum : register(t0);
SamplerState samLinear : register(s0);

float3 normalFromThickness(float2 uv)
{
    float dx = (Accum.SampleLevel(samLinear, uv + float2(invSize.x, 0), 0)
              - Accum.SampleLevel(samLinear, uv - float2(invSize.x, 0), 0)) * 0.5;
    float dy = (Accum.SampleLevel(samLinear, uv + float2(0, invSize.y), 0)
              - Accum.SampleLevel(samLinear, uv - float2(0, invSize.y), 0)) * 0.5;

    // 画面空間勾配→擬似法線（Zを1にして正規化）
    float3 n = normalize(float3(-dx, -dy, 1.0));
    return n;
}

float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD) : SV_Target
{
    float t = Accum.SampleLevel(samLinear, uv, 0); // 厚み

    // なめらかなしきい値（エッジが出る）
    float a = saturate((t - iso) * gain);
    // もっと綺麗にするなら：a = smoothstep(iso - 0.03, iso + 0.03, t);

    if (a <= 0.001)
        discard;

    float3 n = normalFromThickness(uv);

    // 簡易ライティング
    float3 L = normalize(float3(0.4, 0.6, 0.7));
    float diff = saturate(dot(n, L));
    float fres = pow(1.0 - saturate(n.z), 3.0); // 画面法線で簡易フレネル

    float3 base = float3(0.15, 0.35, 0.9); // 水色
    float3 spec = float3(0.9, 0.95, 1.0) * fres;

    float3 col = base * (0.25 + 0.75 * diff) + spec;

    return float4(col, a); // αはaか、常に1ならブレンドをSrcAlphaに
}
