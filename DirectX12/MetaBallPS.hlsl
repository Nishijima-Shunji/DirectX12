// GPU に渡す定数バッファスロット b0
cbuffer MetaballCB : register(b0)
{
    float threshold;      // メタボール領域のしきい値 (例: 0.2)
    float eps;            // エッジぼかし幅 (例: 0.05)
    float maxSum;         // 正規化用の最大合計値 (例: 5.0)
    float4 color;         // 出力色 RGB (例: float4(0.2,0.4,1.0,1.0))
    uint  particleCount;  // パーティクル数
    float pad0, pad1, pad2; // 16 バイト境界合わせ用
};

// パーティクル位置／半径を格納する StructuredBuffer スロット t0
StructuredBuffer<float4> Particles : register(t0);

static const float PI = 3.14159265359;

// Poly6 カーネル関数
float Poly6(float d, float h)
{
    if (d > h) return 0.0;
    float x = (h*h - d*d);
    // 315/(64 π h^9) * (h^2 - d^2)^3
    return (315.0 / (64.0 * PI * pow(h, 9.0))) * x * x * x;
}

struct PSIn
{
    float4 pos : SV_POSITION; // フルスクリーントライアングル頂点から補間
    float2 uv  : TEXCOORD0;   // 0–1 にマップ済み UV
};

float4 PSMain(PSIn IN) : SV_TARGET
{
    float2 uv = IN.uv;
    float  sum = 0.0;

    // 各パーティクルの寄与を累積
    for (uint i = 0; i < particleCount; ++i)
    {
        float2 p = Particles[i].xy;  // 粒子スクリーン位置
        float  r = Particles[i].z;   // 粒子影響半径
        float  d = distance(uv, p);
        sum += Poly6(d, r);
    }

    // 0–1 に正規化
    float sumNorm = saturate(sum / maxSum);

    // しきい値前後 eps 幅で α をぼかす
    float alpha = smoothstep(threshold - eps, threshold + eps, sumNorm);

    return float4(color.rgb, alpha);
}