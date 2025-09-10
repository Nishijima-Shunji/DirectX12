struct PSInput
{
    float4 posH : SV_POSITION;
    float2 uv : TEXCOORD0;
    float rPix : TEXCOORD1;
};

float main(PSInput i) : SV_Target
{
    // 円形ガウス：中心1.0→縁0.0へ
    float2 d = i.uv * 2.0 - 1.0; // [-1,1]^2
    float r2 = dot(d, d);
    if (r2 > 1.0)
        discard;

    float sigma = 0.5;
    float w = exp(-r2 / (2.0 * sigma * sigma));

    return w;
}
