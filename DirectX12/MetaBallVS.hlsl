struct VSOutput
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD;
};


VSOutput main(uint id : SV_VertexID)
{
    VSOutput o;
    float2 v[3] = { float2(-1, -1), float2(3, -1), float2(-1, 3) };
    o.pos = float4(v[id], 0, 1);
    o.uv = v[id] * 0.5 + 0.5;
    return o;
}
