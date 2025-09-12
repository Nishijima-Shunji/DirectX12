cbuffer MetaCB : register(b0)
{
    float4x4 invViewProj;
    float3 camPos;
    float isoLevel;
    uint particleCount;
    float pad[3];
};

struct ParticleMeta
{
    float3 pos; // ワールド空間位置
    float r; // 半径
};

struct VSOutput
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD;
};

static const int MAX_STEP = 16;
StructuredBuffer<ParticleMeta> Particles : register(t0);

// MetaBallのフィールド値と勾配を同時に計算する
float Field(float3 p, out float3 grad)
{
    float sum = 0;
    grad = float3(0, 0, 0); // 勾配の初期化
    [loop]
    for (uint i = 0; i < particleCount; ++i)
    {
        float3 d = p - Particles[i].pos;
        float r2 = Particles[i].r * Particles[i].r;
        float denom = dot(d, d) + 1e-6;
        sum += r2 / denom; // フィールド値を累積
        grad += (-2 * r2 * d) / pow(denom, 2); // 勾配を累積

        // 規定値を超えたら早期終了して無駄な計算を抑える
        if (sum > isoLevel)
            break;
    }
    return sum - isoLevel;
}


float4 main(VSOutput IN) : SV_TARGET
{
    float4 clip = float4(IN.uv * 2 - 1, 0, 1);
    float4 wp = mul(invViewProj, clip);
    wp /= wp.w;
    float3 ro = camPos, rd = normalize(wp.xyz - camPos);
    float3 p = ro;
    float d;
    float3 grad; // フィールドの勾配
    [loop]
    for (int i = 0; i < MAX_STEP; ++i)
    {
        d = Field(p, grad);
        // フィールド値と勾配を取得
        if (abs(d) < 0.001)
            break;
        p += rd * d * 0.2; // ステップ幅を調整
    }

    if (abs(d) >= 0.001)
        discard;

    float3 n = normalize(grad); // 累積した勾配から法線を計算
    float diff = saturate(dot(n, normalize(float3(1, 1, 1))));
    return float4(diff * 0.2, diff * 0.4, diff, 1);
}