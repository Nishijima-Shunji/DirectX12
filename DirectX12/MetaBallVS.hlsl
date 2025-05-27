// register(b0) �͌�� ConstantBuffer �ɕR�Â�
cbuffer ScreenCB : register(b0)
{
    float2 screenSize;   // (��, ����)
    float threshold;     // ���^�{�[��臒l
    float pad0, pad1;
};

// t0 �� StructuredBuffer<float4> �p�[�e�B�N���z����o�C���h
StructuredBuffer<float4> Particles : register(t0);
// float4(x_ndc, y_ndc, radius_ndc, unused)

struct VSInput
{
    float2 pos : POSITION;  // {-1,1},{3,1},{-1,-3} �̎O�p
};

struct VSOutput
{
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD;
};

VSOutput VS_Main(VSInput IN)
{
    VSOutput OUT;
    OUT.pos = float4(IN.pos, 0, 1);
    // UV �� 0��1 �X�y�[�X�Ƀ}�b�v
    OUT.uv = IN.pos * 0.5 + float2(0.5,0.5);
    return OUT;
}
