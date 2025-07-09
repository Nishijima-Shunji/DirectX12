cbuffer ScreenCB : register(b0)
{
    float2 screenSize;
    float  threshold;
    uint   particleCount;
    float  pad0;
};

struct VSOutput
{
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD;
};

//StructuredBuffer<float4> Particles : register(t0);
struct ParticleMeta
{
    float x;   // UV or NDC.x
    float y;   // UV or NDC.y
    float r;   // 半径
    float pad; // ダミー
};
StructuredBuffer<ParticleMeta> Particles : register(t0);


float4 PS_Main(VSOutput IN) : SV_TARGET
{
    float2 uv = IN.uv;

    // スクリーン（ピクセル）座標へ
    float2 pix = uv * screenSize;   // screenSize は (幅, 高さ)

    float sum = 0;
    for (uint i = 0; i < particleCount; ++i)
    {
        float2 p_ndc;
        p_ndc.x = Particles[i].x * 2.0 - 1.0;
        p_ndc.y = Particles[i].y * 2.0 - 1.0;
        float r_w = Particles[i].r;

        // NDC 位置をピクセルに
        float2  p_pix = (p_ndc * 0.5 + 0.5) * screenSize;

        // 投影で縮む分を考慮（遠いほど小さく）
        float   r_pix = r_w * (screenSize.x * 0.5); // ざっくり: 視野90°前提

        float   d = distance(pix, p_pix);
        sum += saturate(1 - d / r_pix);
    }

    float alpha = smoothstep(threshold, threshold + 1.0, sum);
    clip(alpha - 0.01);                 // 透明ピクセルは捨てる
    return float4(0.2, 0.4, 1.0, alpha);
}
