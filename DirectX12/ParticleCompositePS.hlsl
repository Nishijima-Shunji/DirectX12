// スクリーンスペース流体レンダリング用の合成ピクセルシェーダー。
Texture2D<float> gDepthTex : register(t0);
SamplerState gLinearClamp  : register(s0);

cbuffer CompositeCB : register(b0)
{
    float4 misc0; // x=1/width, y=1/height, z=半径, w=法線強調
    float4 misc1; // x,y,z=流体色, w=不透明度基準
    float4 misc2; // x,y,z=ライト方向, w=ハイライト硬さ
};

struct PSInput
{
    float4 position : SV_POSITION;
    float2 uv       : TEXCOORD0;
};

float4 main(PSInput input) : SV_TARGET
{
    float depthCenter = gDepthTex.SampleLevel(gLinearClamp, input.uv, 0);
    if (depthCenter <= 0.0f)
    {
        // 深度が無い画素はそのまま透過
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float2 invScreen = misc0.xy;
    float radius = misc0.z;
    float normalScale = misc0.w;
    float3 fluidColor = misc1.xyz;
    float baseOpacity = misc1.w;
    float3 lightDir = normalize(misc2.xyz);
    float specPower = misc2.w;

    // 近傍ピクセルの深度差分からスクリーンスペース法線を推定
    float depthRight = gDepthTex.SampleLevel(gLinearClamp, input.uv + float2(invScreen.x, 0.0f), 0);
    float depthLeft  = gDepthTex.SampleLevel(gLinearClamp, input.uv - float2(invScreen.x, 0.0f), 0);
    float depthUp    = gDepthTex.SampleLevel(gLinearClamp, input.uv - float2(0.0f, invScreen.y), 0);
    float depthDown  = gDepthTex.SampleLevel(gLinearClamp, input.uv + float2(0.0f, invScreen.y), 0);

    depthRight = (depthRight <= 0.0f) ? depthCenter : depthRight;
    depthLeft  = (depthLeft  <= 0.0f) ? depthCenter : depthLeft;
    depthUp    = (depthUp    <= 0.0f) ? depthCenter : depthUp;
    depthDown  = (depthDown  <= 0.0f) ? depthCenter : depthDown;

    float dx = (depthRight - depthLeft) * 0.5f;
    float dy = (depthDown - depthUp) * 0.5f;
    float3 normal = normalize(float3(-dx * normalScale, -dy * normalScale, 1.0f));

    float3 viewDir = float3(0.0f, 0.0f, -1.0f);
    float diffuse = saturate(dot(normal, lightDir));
    float3 halfDir = normalize(lightDir + viewDir);
    float specular = pow(saturate(dot(normal, halfDir)), specPower);

    // 深度の周辺変化量を基に液体の厚みを推定してアルファに反映
    float thickness = saturate(radius / max(depthCenter, 1e-3f));
    float alpha = saturate(baseOpacity * (0.5f + thickness * 0.5f));

    float3 color = fluidColor * (0.35f + diffuse * 0.55f) + specular * 0.2f;
    return float4(color, alpha);
}
