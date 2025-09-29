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

    // ビルボードの向きを決めるためにカメラの右方向ベクトルと上方向ベクトルを正規化する
    float3 right = normalize(CameraRight);
    float3 up = normalize(CameraUp);

    float3 cornerWS = wp + right * (ra * l.x) + up * (ra * l.y);

    float4 v = mul(float4(cornerWS, 1), View);
    o.pos = mul(v, Proj);

    // UV 座標を 0〜1 の範囲に正規化し、後段で扱いやすい形にする
    o.uv = l * 0.5f + 0.5f; // 必要であれば o.uv.y = 1 - o.uv.y; で上下を反転できる
    return o;
}
