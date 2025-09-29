cbuffer MetaCB : register(b0)
{
    float4x4 invViewProj;   // ビュー射影逆行列
    float4   camRadius;     // xyz: カメラ位置, w: 粒子半径（描画用）
    float4   isoCount;      // x: しきい値, y: 粒子数, z: レイステップ係数, w: 未使用
    float4   waterDeep;     // xyz: 深い水の色, w: 吸収係数
    float4   waterShallow;  // xyz: 浅い水の色, w: 泡検出のしきい値
    float4   shadingParams; // x: 泡強度, y: 反射割合, z: スペキュラパワー, w: 経過時間
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
    const float kernelMul = 2.5f; // 遠すぎる粒子は無視して負荷を下げる

    [loop]
    for (uint i = 0; i < count; ++i)
    {
        float3 d = p - Particles[i].pos;
        float  radius = max(Particles[i].r, 1e-4);
        float  r2 = radius * radius;
        float  dist2 = dot(d, d);
        float  cutoff2 = r2 * kernelMul * kernelMul;
        if (dist2 > cutoff2)
        {
            continue;
        }

        float  influence = exp(-dist2 / r2);
        if (influence < 1e-4)
        {
            continue;
        }
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
    const int MAX_STEP = 80;
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
    float3 lightDir = normalize(float3(-0.35, 0.9, -0.25));

    float diff = saturate(dot(n, lightDir));

    // 水の厚さを簡易的に推定して色の吸収を行う
    float viewDist = length(ro - p);
    float absorption = exp(-viewDist * waterDeep.w);
    float3 baseWater = lerp(waterDeep.xyz, waterShallow.xyz, absorption);

    // 周囲光の色を少し青寄りに設定して水らしい落ち着いたトーンに
    float3 envColor = float3(0.05, 0.16, 0.28);

    float3 diffuse = baseWater * (0.25 + diff * 0.75);

    // フレネルで反射寄りにする領域を抽出し、反射強度はパラメータで制御
    float fresnel = pow(1.0 - saturate(dot(n, viewDir)), 5.0);
    float3 reflection = lerp(baseWater, float3(0.75, 0.9, 1.0), shadingParams.y);

    // 泡は勾配の強さと簡易波動（正弦波）で変動させる
    float curvature = saturate(1.0 - length(grad) * waterShallow.w);
    float foamWave = 0.5 + 0.5 * sin(dot(p.xz, float2(0.8, 0.45)) * 2.2 + shadingParams.w * 1.4);
    float foamMask = saturate((curvature - 0.25) * 2.0) * foamWave;

    float specPower = max(shadingParams.z, 4.0);
    float spec = pow(saturate(dot(reflect(-lightDir, n), viewDir)), specPower);

    float3 color = diffuse;
    color += reflection * fresnel;
    color += spec * 0.35;
    color = lerp(envColor, color, 0.65);
    color += shadingParams.x * foamMask * float3(1.0, 1.0, 1.0);

    // 厚さによる透過感を持たせるためアルファも吸収とフレネルで調整
    float alpha = saturate(0.35 + absorption * 0.45 + fresnel * 0.35);
    color = lerp(envColor, color, alpha);

    return float4(saturate(color), alpha);

}
