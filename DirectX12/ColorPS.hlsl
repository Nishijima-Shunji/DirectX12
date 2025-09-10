// 頂点カラーのみを出力するシンプルなピクセルシェーダー
struct VSOutput {
    float4 svpos : SV_POSITION;
    float4 color : COLOR;
    float2 uv : TEXCOORD;
};

float4 main(VSOutput input) : SV_TARGET {
    return input.color;
}
