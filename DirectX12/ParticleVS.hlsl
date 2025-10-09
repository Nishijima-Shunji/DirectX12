cbuffer Transform : register(b0)
{
    matrix World;
    matrix View;
    matrix Proj;
};

struct VSInput
{
    float3 center : POSITION0;
};

struct VSOutput
{
    float4 svpos : SV_POSITION;
    float2 uv    : TEXCOORD0;
};

static const float2 kBillboardCorners[4] =
{
    float2(-1.0f, -1.0f),
    float2( 1.0f, -1.0f),
    float2(-1.0f,  1.0f),
    float2( 1.0f,  1.0f)
};

VSOutput VSMain(VSInput input, uint vertexID : SV_VertexID)
{
    VSOutput output;

    // ※粒子中心をワールド→ビュー→クリップへ順に変換
    float4 worldCenter = mul(World, float4(input.center, 1.0f));
    float4 viewCenter  = mul(View, worldCenter);
    float4 clipCenter  = mul(Proj, viewCenter);

    // ※SV_VertexID からコーナーを決定し、w補正付きでスクリーンサイズへ拡張
    const float radius = 0.1f; // ※半径は暫定値（ライン化防止のため十分な大きさを確保）
    float2 projScale = float2(Proj._11, Proj._22);
    float depth = max(viewCenter.z, 1.0e-4f);
    float2 ndcHalf = projScale * (radius / depth);
    float2 corner = kBillboardCorners[vertexID];

    float4 clipPos = clipCenter;
    clipPos.xy += corner * ndcHalf * clipCenter.w;

    output.svpos = clipPos;
    output.uv = corner * 0.5f + 0.5f;
    return output;
}
