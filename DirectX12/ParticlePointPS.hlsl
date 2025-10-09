struct PSInput
{
    float4 svpos : SV_POSITION;
    float4 color : COLOR0;
};

float4 PSMain(PSInput input) : SV_TARGET
{
    return input.color; // 単色で粒子位置だけを強調表示
}
