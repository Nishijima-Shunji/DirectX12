
// 1 ���q�����C���X�^���V���O�ŕ`���AFluidDepth/Thickness�����
// 2 �[�x���o�C���e�������� -> �@���č\��
// 3 ���� + Fresnel���� + Beer-Lambert �ō���

cbuffer CameraCB : register(b0)
{
    float4x4 proj; // �������e
    float4x4 view;
    float2 screenSize;
    float2 invScreenSize;
    float nearZ;
    float farZ;
    float3 iorF0; // IOR����ϊ�����F0�i��: �� 0.02�j
    float absorb; // �z���W���iBeer-Lambert�j
}

SamplerState samplerLinearClamp : register(s0);
Texture2D SceneColor : register(t0);
Texture2D SceneDepth : register(t1);
RWTexture2D<uint> FluidDepth : register(u0); // R32_FLOAT
RWTexture2D<uint> Thickness : register(u1); // R16_FLOAT or R32_FLOAT
RWTexture2D<float4> FluidNormal : register(u2); // 8:8:8:8_UNORM �ł���

// 1 ���q�X�v���b�g�iVS/PS �ŏ��j
struct VSIn
{
    float3 pos : POSITION;
    float3 center : TEXCOORD0;
    float radius : TEXCOORD1;
};
struct VSOut
{
    float4 pos : SV_POSITION;
    float3 viewPos : TEXCOORD0;
    float radius : TEXCOORD1;
};
VSOut VS_Particle(VSIn v)
{
    VSOut o;
    float4 wpos = float4(v.center, 1);
    float4 vpos = mul(view, wpos);
    // �X�N���[����Ԃ̉~�ցi�ȈՁF�|�C���g�X�v���C�g��ցj
    o.pos = mul(proj, vpos);
    o.viewPos = vpos.xyz;
    o.radius = v.radius;
    return o;
}

float sphereDepth(float2 uv, float3 viewCenter, float radius)
{
    // ��ʏ�ŋ��̐[�x���ߎ��i�X�v���C�g����Z�����߂�j
    // �����ł͊ȗ����F���S�[�x���̗p�i���^�p��UV���狅�ʕ�������Z�␳�j
    return -viewCenter.z;
}

float4 PS_DepthThickness(VSOut i) : SV_TARGET
{
    float2 uv = i.pos.xy * 0.5 / float2(screenSize.x * 0.5, screenSize.y * 0.5); // ��
    float d = sphereDepth(uv, i.viewPos, i.radius);
    InterlockedMin(FluidDepth[uint2(i.pos.xy)], d);             // �߂����̐[�x
    InterlockedAdd(Thickness[uint2(i.pos.xy)], i.radius * 0.5); // �ȈՌ���
    return 0;
}

// 2 �o�C���e�������� & �@��
[numthreads(8, 8, 1)]
void CS_Bilateral(uint3 id : SV_DispatchThreadID)
{
    // �ߖT�̐[�x��[�x���d�݂ŕ������i�����ȗ��F�K�E�X�~�[�x���j
    // FluidDepth[...] = blurredDepth;
}

float3 reconstructViewPos(uint2 px)
{
    float z = FluidDepth[px];
    // �t�ˉe��view���W��߂��i�ȗ��F�����͊����̋t�ˉe�֐����j
    return float3(0, 0, z);
}

[numthreads(8, 8, 1)]
void CS_Normal(uint3 id : SV_DispatchThreadID)
{
    uint2 p = id.xy;
    float3 C = reconstructViewPos(p);
    float3 Rx = reconstructViewPos(p + uint2(1, 0)) - C;
    float3 Ry = reconstructViewPos(p + uint2(0, 1)) - C;
    float3 N = normalize(cross(Rx, Ry));
    FluidNormal[p] = float4(N * 0.5 + 0.5, 1);
}

// 3 �����iPS�j
float4 PS_Composite(float4 svpos : SV_POSITION, float2 uv : TEXCOORD) : SV_TARGET
{
    float d = FluidDepth[uint2(svpos.xy)];
    if (d == 0)
        discard;

    float t = Thickness[uint2(svpos.xy)];
    float3 N = normalize(FluidNormal[uint2(svpos.xy)].xyz * 2 - 1);

    // Fresnel�iSchlick�j
    float3 V = float3(0, 0, 1);
    float cosT = saturate(dot(N, V));
    float3 F = iorF0 + (1 - iorF0) * pow(1 - cosT, 5);

    // ���܁i�ȈՁF�w�i���I�t�Z�b�g�T���v���j
    float2 refrUV = uv + N.xy * 0.02; // �W���͒���
    float3 refr = SceneColor.SampleLevel(samplerLinearClamp, refrUV, 0).rgb;

    // Beer-Lambert
    float3 trans = exp(-absorb.xxx * t);
    float3 col = lerp(refr * trans, 1.0.xxx, F); // ���˂͏ȗ�or�L���[�u�}�b�v�ŉ��Z

    return float4(col, 1);
}
