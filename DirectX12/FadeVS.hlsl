struct VSInput
{
    float3 pos : POSITION;
};

struct VSOutput
{
    float4 svpos : SV_POSITION;
};

VSOutput main(VSInput input)
{
    VSOutput o;
    o.svpos = float4(input.pos, 1.0f);
    return o;
}
