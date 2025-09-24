cbuffer AccumCB : register(b0)
{
    row_major float4x4 View;
    row_major float4x4 Proj;
    float3 CameraRight;
    float _pad0;
    float3 CameraUp;
    float _pad1;
};

struct ParticleMeta
{
    float3 pos;
    float r;
};
StructuredBuffer<ParticleMeta> Particles : register(t0);

struct VSOut
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

static const float2 kQuad[4] =
{
    float2(-1, -1), float2(1, -1),
    float2(-1, 1), float2(1, 1)
};

VSOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    VSOut o;
    float2 l = kQuad[vid];

    float3 wp = Particles[iid].pos;
    float ra = Particles[iid].r;

    // ワールド空間のビルボード
    float3 right = normalize(CameraRight);
    float3 up = normalize(CameraUp);

    float3 cornerWS = wp + right * (ra * l.x) + up * (ra * l.y);

    float4 v = mul(float4(cornerWS, 1), View);
    o.pos = mul(v, Proj);

    // 0..1 に
    o.uv = l * 0.5f + 0.5f; // もし合成時に上下が反転するなら o.uv.y = 1 - o.uv.y;
    return o;
}
