struct VSOut
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

float4 main(VSOut i) : SV_Target
{
    float2 xy = i.uv * 2.0 - 1.0;
    float r2 = dot(xy, xy);
    if (r2 > 1.0)
        discard;

    // ‹…‚ÌüÏ•ª‚ÌŒú‚İ
    float thickness = 2.0 * sqrt(1.0 - r2);

    thickness = pow(thickness, 1.0);

    // R16F ‚É‰ÁZ‚·‚é
    return float4(thickness, 0, 0, 0);
}
