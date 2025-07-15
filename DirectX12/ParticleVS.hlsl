cbuffer Transform : register(b0)
{
    matrix World;
    matrix View;
    matrix Proj;
};

struct VSInput
{
    float3 pos : POSITION;
};

struct VSOutput
{
    float4 pos : SV_POSITION;
    float4 color : COLOR;
};


VSOutput VSMain(VSInput input)
{
    VSOutput output;
    float4 worldPos = mul(World, float4(input.pos, 1.0f));
    float4 viewPos = mul(View, worldPos);
    output.pos = mul(Proj, viewPos);
    output.color = float4(0.2, 0.6, 1.0, 1.0);
    return output;
}
