
cbuffer CompositeCB : register(b0)
{
    float iso; // ��: 0.35f  (���݂̂������l)
    float gain; // ��: 6.0f   (�����オ��̋���)
    float2 invSize; // 1/width, 1/height (SSA�𑜓x)
};

Texture2D<float> Accum : register(t0);
SamplerState samLinear : register(s0);

float3 normalFromThickness(float2 uv)
{
    float dx = (Accum.SampleLevel(samLinear, uv + float2(invSize.x, 0), 0)
              - Accum.SampleLevel(samLinear, uv - float2(invSize.x, 0), 0)) * 0.5;
    float dy = (Accum.SampleLevel(samLinear, uv + float2(0, invSize.y), 0)
              - Accum.SampleLevel(samLinear, uv - float2(0, invSize.y), 0)) * 0.5;

    // ��ʋ�Ԍ��z���[���@���iZ��1�ɂ��Đ��K���j
    float3 n = normalize(float3(-dx, -dy, 1.0));
    return n;
}

float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD) : SV_Target
{
    float t = Accum.SampleLevel(samLinear, uv, 0); // ����

    // �Ȃ߂炩�Ȃ������l�i�G�b�W���o��j
    float a = saturate((t - iso) * gain);
    // �������Y��ɂ���Ȃ�Fa = smoothstep(iso - 0.03, iso + 0.03, t);

    if (a <= 0.001)
        discard;

    float3 n = normalFromThickness(uv);

    // �ȈՃ��C�e�B���O
    float3 L = normalize(float3(0.4, 0.6, 0.7));
    float diff = saturate(dot(n, L));
    float fres = pow(1.0 - saturate(n.z), 3.0); // ��ʖ@���ŊȈՃt���l��

    float3 base = float3(0.15, 0.35, 0.9); // ���F
    float3 spec = float3(0.9, 0.95, 1.0) * fres;

    float3 col = base * (0.25 + 0.75 * diff) + spec;

    return float4(col, a); // ����a���A���1�Ȃ�u�����h��SrcAlpha��
}
