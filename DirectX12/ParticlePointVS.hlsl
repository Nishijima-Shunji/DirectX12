cbuffer Transform : register(b0)
{
    float4x4 World;
    float4x4 View;
    float4x4 Proj;
};

struct VSInput
{
    float3 position : POSITION0; // 粒子中心座標
    float  radius   : TEXCOORD0; // SSFR 用半径（点スプライトでは直接使わないがレイアウト互換のため保持）
};

struct VSOutput
{
    float4 svpos      : SV_POSITION;
    float4 color      : COLOR0;
    float pointSize : PSIZE;
};

VSOutput VSMain(VSInput input)
{
    VSOutput output;
    float4 worldPos = mul(World, float4(input.position, 1.0f)); // 粒子の中心をワールドへ変換
    float4 viewPos = mul(View, worldPos);
    output.svpos = mul(Proj, viewPos); // 通常の射影でスクリーンへ

    // SSFR では半径を拡張しているため、その値を使わず一定ピクセルで描画して中心位置のみを強調する
    output.pointSize = 4.0f; // 目視で確認しやすいサイズ
    output.color = float4(1.0f, 0.2f, 0.2f, 1.0f); // 目視確認しやすい赤系カラー
    return output;
}
