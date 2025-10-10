// マーチングキューブのサーフェスをシンプルなライティングで描画するピクセルシェーダー
cbuffer Transform : register(b0)
{
    matrix World;
    matrix View;
    matrix Proj;
    float3 CameraPos; // 視線方向を使って水面を軽やかに見せるためのカメラ位置
    float _padding;   // 16バイト境界を保つための調整
}
struct PSInput
{
    float4 svpos : SV_POSITION;
    float3 normal : NORMAL;
    float3 worldPos : TEXCOORD0;
};

float4 main(PSInput input) : SV_TARGET
{
    float3 normal = normalize(input.normal);
    float3 lightDir = normalize(float3(0.25f, 0.9f, 0.45f)); // 少し上から柔らかく当たる光で重さを減らす
    float3 viewDir = normalize(CameraPos - input.worldPos); // ギャップを目立たなくする反射表現に視線を利用
    float3 halfVec = normalize(lightDir + viewDir);

    float diffuse = saturate(dot(normal, lightDir));
    float specular = pow(saturate(dot(normal, halfVec)), 48.0f); // 滑らかなハイライトで水らしい輝きを加える
    float fresnel = pow(1.0f - saturate(dot(normal, viewDir)), 3.0f); // エッジを明るくして隙間感を軽減

    float ambient = 0.35f;
    float3 baseColor = float3(0.35f, 0.78f, 1.05f); // 彩度を落として軽い水色に調整
    float3 color = baseColor * (ambient + diffuse * 0.65f);
    color += specular * float3(0.9f, 0.95f, 1.0f) * 0.4f;
    color += fresnel * float3(0.3f, 0.5f, 0.8f) * 0.5f;

    return float4(saturate(color), 1.0f);
}
