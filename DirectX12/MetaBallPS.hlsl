cbuffer MetaCB : register(b0)
{
    float4x4 invViewProj;   // ビュー射影逆行列
    float4   camRadius;     // xyz: カメラ位置, w: 粒子半径（描画用）
    float4   isoCount;      // x: しきい値, y: 粒子数, z: レイステップ係数, w: 未使用
};

struct ParticleMeta
{
    float3 pos;  // 粒子のワールド座標
    float  r;    // 個別半径（今回は共通値）
};

struct VSOutput
{
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD;
};

StructuredBuffer<ParticleMeta> Particles : register(t0);

// 粒子群からスカラー場を評価し、同時に勾配（法線）を計算する
float Field(float3 p, out float3 grad)
{
    grad = float3(0, 0, 0);
    float sum = 0;
    const uint count = (uint)isoCount.y;
    if (count == 0)
    {
        return -1.0; // 粒子が無ければヒットしない
    }

    const float iso = isoCount.x;
    // 粒子毎に滑らかなガウス核で寄与を合成する（SSFRの厚み感を少し意識）
    [loop]
    for (uint i = 0; i < count; ++i)
    {
        float3 d = p - Particles[i].pos;
        float  radius = max(Particles[i].r, 1e-4);
        float  r2 = radius * radius;
        float  dist2 = dot(d, d);
        float  influence = exp(-dist2 / (r2));
        sum += influence;
        grad += (-2.0 * influence / r2) * d;

        // 充分密度が溜まったら早期終了してレイマーチ回数を節約
        if (sum > iso + 0.75)
        {
            break;
        }
    }

    return sum - iso;
}

// main関数はVS側からの構造体をそのまま使用
float4 main(VSOutput IN) : SV_TARGET
{
    float4 clip = float4(IN.uv * 2.0 - 1.0, 0.0, 1.0);
    float4 wp = mul(invViewProj, clip);
    wp /= wp.w;

    float3 ro = camRadius.xyz;
    float3 rd = normalize(wp.xyz - ro);

    float3 p = ro;
    float  d = 0.0;
    float3 grad = float3(0, 0, 0);
    const int MAX_STEP = 96;
    const float stepScale = max(isoCount.z, 0.05);

    [loop]
    for (int i = 0; i < MAX_STEP; ++i)
    {
        d = Field(p, grad);
        if (abs(d) < 0.01)
        {
            break; // 等値面に十分近づいたら終了
        }
        p += rd * (-d) * stepScale; // SSFR風に負方向へ押し戻す
    }

    if (abs(d) >= 0.02)
    {
        discard; // 交差しなかったので何も描かない
    }

    float3 n = (dot(grad, grad) > 1e-6) ? normalize(grad) : float3(0, 1, 0);
    float3 viewDir = normalize(ro - p);
    float3 lightDir = normalize(float3(-0.3, 0.9, -0.2));

    float diff = saturate(dot(n, lightDir));
    float3 envColor = float3(0.05, 0.2, 0.35);
    float3 baseWater = float3(0.15, 0.4, 0.85);
    float3 shallowTint = float3(0.45, 0.75, 1.0);

    // フレネルによる境界のハイライト（簡易版）
    float fresnel = pow(1.0 - saturate(dot(n, viewDir)), 3.0);
    float spec = pow(saturate(dot(reflect(-lightDir, n), viewDir)), 32.0);

    // 勾配の長さで泡っぽいハイライトを加算（密度が高い境界ほど白く）
    float foam = saturate(1.0 - length(grad) * 0.25);

    float3 color = envColor * (1.0 - fresnel);
    color += lerp(baseWater, shallowTint, diff) * (0.35 + diff * 0.65);
    color += fresnel * float3(0.6, 0.85, 1.0);
    color += spec * 0.25;
    color += foam * 0.1;

    float alpha = saturate(0.55 + fresnel * 0.35);
    color = lerp(envColor, color, alpha); // 透過時は背景色と自然に混ざるように調整

    return float4(saturate(color), alpha);
}
