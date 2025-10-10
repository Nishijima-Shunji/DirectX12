cbuffer OceanConstant : register(b0)
{
    float4x4 World; // 世界行列
    float4x4 View;  // ビュー行列
    float4x4 Proj;  // プロジェクション行列
    float4 SurfaceColor; // 水面色
}

struct VSInput
{
    float3 position : POSITION; // 頂点座標
    float3 normal   : NORMAL;   // 法線ベクトル
    float2 uv       : TEXCOORD; // UV座標
    float4 color    : COLOR;    // 頂点カラー
};

struct VSOutput
{
    float4 svpos    : SV_POSITION; // クリップ座標
    float3 worldPos : TEXCOORD0;   // 世界座標
    float3 normal   : TEXCOORD1;   // 世界法線
    float4 color    : COLOR0;      // 色
    float2 uv       : TEXCOORD2;   // UV
};

VSOutput main(VSInput input)
{
    VSOutput output;
    float4 localPos = float4(input.position, 1.0f);
    float4 worldPos = mul(World, localPos);
    float4 viewPos = mul(View, worldPos);
    float4 projPos = mul(Proj, viewPos);

    float3 worldNormal = normalize(mul((float3x3)World, input.normal));

    output.svpos = projPos;
    output.worldPos = worldPos.xyz;
    output.normal = worldNormal;
    output.color = input.color * SurfaceColor;
    output.uv = input.uv;
    return output;
}
