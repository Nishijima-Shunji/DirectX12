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
    float4 viewPos4 = mul(view, float4(particle.position, 1.0f));
    float3 viewPos = viewPos4.xyz;

    float2 local = kBillboardCorners[vertexID];

    // W 補正無しだとスクリーン奥行きでつぶれるため、投影行列からスケールを抽出して視差補正する
    float4 clipCenter = mul(proj, float4(viewPos, 1.0f));
    float2 projScale = float2(proj._11, proj._22); // ※FOV 依存の拡縮係数
    float2 pixelSize = (particle.radius / max(viewPos.z, 1e-4f)) * projScale; // 奥行きに応じて縮小
    float2 ndcOffset = local * pixelSize;

    float4 clipPos;
    clipPos.xy = clipCenter.xy + ndcOffset * clipCenter.w; // W を掛けてスクリーン空間の歪みを補正
    clipPos.zw = clipCenter.zw;

    output.position    = clipPos;
    output.viewCenter  = viewPos;
    output.radius      = particle.radius;
    output.localOffset = local;
    // ここで視点手前の粒子を描画しないようにする（ニア面との距離が負なら GPU が自動的に破棄）
    output.clipNear    = viewPos.z - (nearZ + particle.radius);

    return output;
}
