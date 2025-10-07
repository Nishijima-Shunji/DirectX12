cbuffer Transform : register(b0)
{
    float4x4 World;
    float4x4 View;
    float4x4 Proj;
}

struct VSInput
{
    float3 position : POSITION;
};

struct VSOutput
{
    float4 svpos : SV_POSITION;
};

VSOutput main(VSInput input)
{
    VSOutput output;
    float4 worldPos = mul(World, float4(input.position, 1.0f)); // ワールド行列で格子線の位置をそろえる
    float4 viewPos = mul(View, worldPos);
    output.svpos = mul(Proj, viewPos);
    return output;
}
