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


VSOutput VSMain(VSInput input)
{
    VSOutput output;
    float3 scaledPos = input.localPos * input.instanceRadius; // 影響半径に合わせて球を拡大
    float4 worldPos = mul(World, float4(input.instancePos + scaledPos, 1.0f)); // 粒子の中心へ平行移動
    float4 viewPos = mul(View, worldPos);
    output.svpos = mul(Proj, viewPos);

    float3x3 world3x3 = (float3x3)World;
    output.normal = normalize(mul(world3x3, input.localNormal)); // 法線もワールドへ変換
    output.worldPos = worldPos.xyz;
    return output;
}
