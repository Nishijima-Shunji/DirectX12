// マーチングキューブで生成したサーフェスをワールドへ配置する頂点シェーダー
cbuffer Transform : register(b0)
{
    matrix World;
    matrix View;
    matrix Proj;
    float3 CameraPos; // PS 側と定数バッファを揃えるためにカメラ位置も保持
    float _padding;   // 16バイト境界を守るためのダミー
}

struct VSInput
{
    float3 position : POSITION0;
    float3 normal   : NORMAL0;
};

struct VSOutput
{
    float4 svpos : SV_POSITION;
    float3 normal : NORMAL;
    float3 worldPos : TEXCOORD0;
};

VSOutput main(VSInput input)
{
    VSOutput output;
    float4 worldPos = mul(World, float4(input.position, 1.0f));
    float4 viewPos = mul(View, worldPos);
    output.svpos = mul(Proj, viewPos);

    float3x3 world3x3 = (float3x3)World;
    output.normal = normalize(mul(world3x3, input.normal));
    output.worldPos = worldPos.xyz;
    return output;
}
