cbuffer Transform : register(b0)
{
    float4x4 World;
    float4x4 View;
    float4x4 Proj;
}

struct VSInput
{
    float3 pos : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD;
    float3 tangent : TANGENT;
    float4 color : COLOR;
};

struct VSOutput
{
    float4 svpos : SV_POSITION;
    float4 color : COLOR;
// Gg[|CgmainɓꂵPSO̎wƊݍ킹
VSOutput main(VSInput input)
};

VSOutput vert(VSInput input)
{
    VSOutput output = (VSOutput) 0;

    float4 localPos = float4(input.pos, 1.0f);
    float4 worldPos = mul(World, localPos);
    float4 viewPos = mul(View, worldPos);
    float4 projPos = mul(Proj, viewPos);

    output.svpos = projPos;
    output.color = input.color;
    output.uv = input.uv; // 入力からuvを渡す
    return output;
}
