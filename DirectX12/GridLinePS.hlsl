struct VSOutput
{
    float4 svpos : SV_POSITION;
};

float4 main(VSOutput input) : SV_TARGET
{
    return float4(1.0f, 0.85f, 0.2f, 1.0f); // グリッド境界は視認しやすい暖色で描画
}
