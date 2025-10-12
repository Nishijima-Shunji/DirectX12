// 画面全体に三角形を張ってポストエフェクト処理を行う頂点シェーダー。
struct VSOutput
{
    float4 position : SV_POSITION;
    float2 uv       : TEXCOORD0;
};

VSOutput main(uint vertexID : SV_VertexID)
{
    // フルスクリーントライアングルの頂点座標（UVは0-1で正規化）
    float2 pos[3] = {
        float2(-1.0f, -1.0f),
        float2(-1.0f,  3.0f),
        float2( 3.0f, -1.0f)
    };

    VSOutput output;
    output.position = float4(pos[vertexID], 0.0f, 1.0f);
    output.uv = float2(output.position.x * 0.5f + 0.5f,
                       -output.position.y * 0.5f + 0.5f);
    return output;
}
