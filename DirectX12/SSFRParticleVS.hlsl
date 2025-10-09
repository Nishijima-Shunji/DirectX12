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

VSOut main(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOut output;

    ParticleData particle = g_Particles[instanceID];

    float radius = particle.radius;

    float4 worldPos = mul(world, float4(particle.position, 1.0f));
    float4 viewPos = mul(view, worldPos);

    // ※行列内の行ベクトルからカメラの右・上方向（ビュー空間基底）を取得し、矩形を正しく構築する
    float3 viewRight = normalize(float3(view._11, view._12, view._13));
    float3 viewUp = normalize(float3(view._21, view._22, view._23));

    // ※ローカル空間の球半径へワールド行列のスケールを反映させ、ドット描画と同一スケールにそろえる
    float3 axisX = float3(world._11, world._12, world._13);
    float3 axisY = float3(world._21, world._22, world._23);
    float3 axisZ = float3(world._31, world._32, world._33);
    float radiusScale = (length(axisX) + length(axisY) + length(axisZ)) / 3.0f;
    float scaledRadius = max(radius * radiusScale, 1e-6f);

    // ※頂点IDからコーナー（-1〜+1）を求め、ビュー空間上で正しくオフセットしてから射影する
    float2 local = float2((vertexID & 1) ? +1.0f : -1.0f, (vertexID & 2) ? +1.0f : -1.0f);
    float3 cornerView = viewPos.xyz + (viewRight * local.x + viewUp * local.y) * scaledRadius;
    float4 clipPos = mul(proj, float4(cornerView, 1.0f));

    output.position    = clipPos;
    output.viewCenter  = viewPos.xyz;
    output.radius      = scaledRadius;
    output.localOffset = local;
    // ※ワールド半径を反映した距離でニア面判定を行い、点群と同じ位置関係を保つ
    output.clipNear    = viewPos.z - (nearZ + scaledRadius);

    return output;
}
