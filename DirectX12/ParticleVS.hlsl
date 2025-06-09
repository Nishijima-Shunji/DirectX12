cbuffer Transform : register(b0)
{
    matrix World;
    matrix View;
    matrix Proj;
};

struct VSInput
{
    float3 position : POSITION;

    float4 world0 : INSTANCE_WORLD0;
    float4 world1 : INSTANCE_WORLD1;
    float4 world2 : INSTANCE_WORLD2;
    float4 world3 : INSTANCE_WORLD3;
};

struct VSOutput
{
    float4 position : SV_POSITION;
};

VSOutput VSMain(VSInput input)
{
    float4x4 instanceWorld = float4x4(
        input.world0,
        input.world1,
        input.world2,
        input.world3
    );

    float4 localPos = float4(input.position, 1.0f);

    // çsóÒÇÃä|ÇØèáÇç∂ä|ÇØÇ…ïœçX
    float4 worldPos = mul(instanceWorld, localPos);
    float4 viewPos = mul(View, worldPos);
    float4 projPos = mul(Proj, viewPos);

    VSOutput output;
    output.position = projPos;
    return output;
}
