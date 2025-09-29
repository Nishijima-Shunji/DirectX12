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
    // DirectX �͏オ +Y �Ȃ̂ňʒu�� Y �𔽓]���ďo��
    o.pos = float4(p * float2(2, -2) + float2(-1, 1), 0, 1);

    // UV �� 0..1 �ɐ��K�����A�e�N�X�`���̌��_(����)�ɍ��킹�� Y �𔽓]
    o.uv = float2(p.x * 0.5, 1.0 - p.y * 0.5);

    return o;
}
