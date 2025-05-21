struct VSOutput
{
    float4 Position : SV_POSITION;
    //float2 TexCoord : COLOR;
};

float4 PSMain(VSOutput input) : SV_TARGET
{
    return float4(0.1f, 0.1f, 1.0f, 1.0f);
}
