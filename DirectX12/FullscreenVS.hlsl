struct VSOut
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

VSOut main(uint vid : SV_VertexID)
{
    // 0:(0,0) 1:(2,0) 2:(0,2)
    float2 p = float2((vid << 1) & 2, vid & 2);

    VSOut o;
    // DirectX は上が +Y なので位置は Y を反転して出す
    o.pos = float4(p * float2(2, -2) + float2(-1, 1), 0, 1);

    // UV は 0..1 に正規化し、テクスチャの原点(左上)に合わせて Y を反転
    o.uv = float2(p.x * 0.5, 1.0 - p.y * 0.5);

    return o;
}
