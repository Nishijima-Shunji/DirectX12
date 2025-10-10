// ※SSFR初段: 粒子をスクリーン空間の四角形へ展開し、球とみなすための頂点シェーダー。
cbuffer DepthConstants : register(b0)
{
    float4x4 gView;    // ビュー行列（転置済み）
    float4x4 gProj;    // 射影行列（転置済み）
    float2   gClipZ;   // ニア/ファー平面
    float2   _pad0;    // 16byte境界合わせ
};

struct VSInput
{
    float2 corner     : POSITION0; // 粒子ビルボードの隅[-1,+1]
    float3 instancePos: POSITION1; // 粒子中心のワールド座標
    float  radius     : TEXCOORD0; // 粒子半径[m]
};

struct VSOutput
{
    float4 position : SV_Position;
    float3 viewPos  : TEXCOORD0;
    float2 uv       : TEXCOORD1;
    float  radius   : TEXCOORD2;
};

VSOutput main(VSInput input)
{
    VSOutput output;

    float4 worldCenter = float4(input.instancePos, 1.0f);
    float4 viewCenter = mul(gView, worldCenter);

    float2 offsetXY = input.corner * input.radius;
    float3 viewPos = viewCenter.xyz + float3(offsetXY, 0.0f);

    output.position = mul(gProj, float4(viewPos, 1.0f));
    output.viewPos = viewCenter.xyz;
    output.uv = input.corner * 0.5f + 0.5f;
    output.radius = input.radius;

    return output;
}
