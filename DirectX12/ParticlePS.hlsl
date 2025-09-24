struct VSOutput
{
    float4 Position : SV_POSITION;
    float4 Color    : COLOR;
};

float4 PSMain(VSOutput input) : SV_TARGET
{
    // VSから受け取った色をそのまま出力するだけのシンプルなPS
    return input.Color;
}
