cbuffer SceneCB : register(b0)
{
    matrix viewProj;
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
    output.pos = mul(viewProj, float4(input.pos, 1.0f));
    output.color = float4(0.2, 0.6, 1.0, 1.0); // ê¬Ç¡Ç€Ç¢
    return output;
}
