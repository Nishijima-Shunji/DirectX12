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
    float r;    // 半径
};

struct VSOutput
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD;
};

static const int MAX_STEP = 32; // 描画用の制限
StructuredBuffer<ParticleMeta> Particles : register(t0);

// MetaBallのフィールド関数
float Field(float3 p)
{
    float sum = 0;
  [loop]
    for (uint i = 0; i < particleCount; ++i)
    {
        float3 d = p - Particles[i].pos;
        sum += (Particles[i].r * Particles[i].r) / (dot(d, d) + 1e-6);
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
  [loop]
    for (int i = 0; i < MAX_STEP; ++i)
    {
        d = Field(p);
        if (abs(d) < 0.001)
            break;
        p += rd * d * 0.5;
    }
    if (abs(d) >= 0.001)
        return float4(0, 0, 0, 0);
    float3 n = normalize(float3(
    Field(p + float3(0.001, 0, 0)) - Field(p - float3(0.001, 0, 0)),
    Field(p + float3(0, 0.001, 0)) - Field(p - float3(0, 0.001, 0)),
    Field(p + float3(0, 0, 0.001)) - Field(p - float3(0, 0, 0.001))
  ));
    float diff = saturate(dot(n, normalize(float3(1, 1, 1))));
            // Water-like bluish tint
    return float4(diff * 0.2, diff * 0.4, diff, 1);
}
