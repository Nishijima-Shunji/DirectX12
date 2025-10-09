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
    float3 viewCenter = mul(view, float4(particle.position, 1.0f)).xyz;
    float4 clipCenter = mul(proj, float4(viewCenter, 1.0f));

    float2 local = kBillboardCorners[vertexID];

    // 投影行列のスケールから画面上の半径（NDC）を算出し、1ピクセル未満はクランプする
    float2 projScale = float2(proj._11, proj._22);
    float radius = particle.radius;
    float depth = max(viewCenter.z, 1e-4f);
    float2 ndcHalf = projScale * (radius / depth);
    ndcHalf = max(ndcHalf, 1.0f / screenSize);

    float4 clipPos = clipCenter;
    clipPos.xy += local * ndcHalf * clipCenter.w; // ※w補正付きでビルボードを展開

    output.position    = clipPos;
    output.viewCenter  = viewCenter;
    output.radius      = radius;
    output.localOffset = local;
    // 近傍クリップの境界計算を球前面の射影に差し替え（ニア面と衝突したときの欠けを防止するため）
    float2 sphereXY = local * radius;
    float  radiusSq = radius * radius;
    float  insideSq = max(radiusSq - dot(sphereXY, sphereXY), 0.0f);
    float  zFront   = viewCenter.z - sqrt(insideSq);
    float4 clipFront = mul(proj, float4(viewCenter.xy, zFront, 1.0f));
    output.clipNear = clipFront.z;

    return output;
}
