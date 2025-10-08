#include "SharedStruct.hlsli"

// 粒子の中心座標と半径をまとめた構造体（StructuredBuffer 経由で受け取る想定）
struct ParticleData
{
    float3 position;
    float  radius;
};

StructuredBuffer<ParticleData> g_Particles : register(t0);

struct VSOut
{
    float4 position    : SV_POSITION;     // クリップ空間座標
    float3 viewCenter  : TEXCOORD0;       // ビュー空間での粒子中心
    float  radius      : TEXCOORD1;       // 粒子半径（ビュー空間スケール）
    float2 localOffset : TEXCOORD2;       // ビルボード内でのローカル座標（-1〜1）
    float  clipNear    : SV_ClipDistance0;// ニア面より手前の粒子は全部捨てるためのクリップ距離
};

static const float2 kBillboardCorners[4] =
{
    float2(-1.0f, -1.0f),
    float2( 1.0f, -1.0f),
    float2(-1.0f,  1.0f),
    float2( 1.0f,  1.0f)
};

VSOut main(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOut output;

    ParticleData particle = g_Particles[instanceID];

    // ビュー空間へ変換してカメラ基準で位置と半径を扱う
    float4 viewPos = mul(view, float4(particle.position, 1.0f));

    float2 local = kBillboardCorners[vertexID];

    // ビュー空間 XY 平面で粒子半径分だけ四隅を押し広げる
    float3 cornerView = viewPos.xyz + float3(local * particle.radius, 0.0f);

    // 透視投影してクリップ空間へ
    float4 clipPos = mul(proj, float4(cornerView, 1.0f));

    output.position    = clipPos;
    output.viewCenter  = viewPos.xyz;
    output.radius      = particle.radius;
    output.localOffset = local;
    // ここで視点手前の粒子を描画しないようにする（ニア面との距離が負なら GPU が自動的に破棄）
    output.clipNear    = viewPos.z - (nearZ + particle.radius);

    return output;
}
