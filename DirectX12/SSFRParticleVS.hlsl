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

    float4 worldPos = float4(particle.position, 1.0f);
    float4 viewPos = mul(view, worldPos);
    float4 clipPos = mul(proj, viewPos);

    float3 viewCenter = viewPos.xyz;

    // ビュー行列の列ベクトルからカメラの右・上方向を取り出し、粒子クアッドを常に正対させる
    float3 camRight = normalize(float3(view._11, view._21, view._31));
    float3 camUp = normalize(float3(view._12, view._22, view._32));

    // 頂点IDからスクリーン内のコーナー（-1〜+1）を作り、ローカル座標にも流用する
    float2 local = float2((vertexID & 1) ? +1.0f : -1.0f, (vertexID & 2) ? +1.0f : -1.0f);
    float3 billboardDir = camRight * local.x + camUp * local.y;

    // NDC 半径へ変換する際に w で正しくスケールし、距離に応じた見かけの大きさのずれを抑える
    float2 ndcRadius = (radius * float2(proj._11, proj._22)) / max(clipPos.w, 1e-6f);
    ndcRadius = max(ndcRadius, 1.0f / screenSize); // 画素より小さい場合の潰れ防止

    clipPos.xy += local * ndcRadius * clipPos.w;

    output.position    = clipPos;
    output.viewCenter  = viewCenter;
    output.radius      = radius;
    // ビルボードの向きをカメラ基準で保持し、PS での円判定が常に正しく働くようにする
    output.localOffset = float2(dot(billboardDir, camRight), dot(billboardDir, camUp));
    // ここで視点手前の粒子を描画しないようにする（ニア面との距離が負なら GPU が自動的に破棄）
    output.clipNear    = viewCenter.z - (nearZ + radius);

    return output;
}
