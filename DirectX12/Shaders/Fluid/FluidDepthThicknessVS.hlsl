// 画面全面を覆う三角形を生成する頂点シェーダ
struct VSOut
{
    float4 position : SV_POSITION;
    float2 uv       : TEXCOORD0;
};

VSOut VSMain(uint vertexId : SV_VertexID)
{
    VSOut o;
    float2 pos = float2((vertexId << 1) & 2, vertexId & 2);
    o.position = float4(pos * float2(2, -2) + float2(-1, 1), 0, 1);
    o.uv = float2(pos.x * 0.5f, pos.y * 0.5f);
    return o;
}
