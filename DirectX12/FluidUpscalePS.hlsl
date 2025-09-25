// FluidUpscalePS.hlsl
// 低解像度でレイマーチした結果を線形補間で拡大するだけのシンプルなピクセルシェーダー
Texture2D<float4> gFluidColor : register(t0);
SamplerState gLinearClamp : register(s0);

float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET
{
    // 低解像度テクスチャをそのまま線形サンプリングして返す
    return gFluidColor.Sample(gLinearClamp, uv);
}
