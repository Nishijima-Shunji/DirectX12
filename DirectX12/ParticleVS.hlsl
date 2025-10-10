cbuffer Transform : register(b0)
{
    matrix World;
    matrix View;
    matrix Proj;
};

struct VSInput
{
    float3 localPos : POSITION0;
    float3 localNormal : NORMAL0;
    float3 instancePos : POSITION1;
    float  instanceRadius : TEXCOORD0;
};

struct VSOutput
{
    float4 svpos   : SV_POSITION;
    float3 normal  : NORMAL;
    float3 worldPos : TEXCOORD0;
};


// FXCがVSMainを参照するためエントリーポイント名を一致させる
VSOutput VSMain(VSInput input)
{
    VSOutput output;

    // メッシュ頂点を半径で拡縮し、インスタンスのワールド座標へ移動させる
    float3 worldPosition = (input.localPos * input.instanceRadius) + input.instancePos;

    // ワールド座標をビュー、射影行列で変換し、最終的なクリップ座標を計算する
    float4 viewPos = mul(View, float4(worldPosition, 1.0f));
    output.svpos = mul(Proj, viewPos);

    // 法線も正しくワールド空間へ変換する (Worldは単位行列なので変換不要)
    output.normal = normalize(input.localNormal);
    output.worldPos = worldPosition;

    return output;
}
