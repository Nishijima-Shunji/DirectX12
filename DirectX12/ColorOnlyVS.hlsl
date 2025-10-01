cbuffer ColorPass : register(b0)
{
    matrix View;
    matrix Proj;
};

struct VSInput
{
    float3 position : POSITION;
    float4 color    : COLOR;
};

struct VSOutput
{
    float4 position : SV_POSITION;
    float4 color    : COLOR;
};

VSOutput main(VSInput input)
{
    VSOutput output;
    float4 viewPos = mul(View, float4(input.position, 1.0f));
    output.position = mul(Proj, viewPos);
    output.color = input.color;
    return output;
}
