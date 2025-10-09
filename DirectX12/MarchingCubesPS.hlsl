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
    float specular = pow(saturate(dot(normal, halfVec)), 64.0f); // 鋭さを上げて水面のハイライトを強調
    float fresnel = pow(1.0f - saturate(dot(normal, viewDir)), 3.0f); // エッジを明るくして隙間感を軽減

    float ambient = 0.32f;
    float3 baseColor = float3(0.32f, 0.74f, 0.98f); // 少し透け感のある水色
    float depthFade = saturate(length(CameraPos - input.worldPos) * 0.08f); // 視線距離による減衰で水の厚みを演出

    // 水らしく見せるためのカラー調整（距離で深み、Fresnel でエッジを強調）
    float3 color = baseColor * (ambient + diffuse * 0.7f);
    color = lerp(color, float3(0.18f, 0.38f, 0.6f), depthFade * 0.45f);
    color += specular * float3(0.95f, 0.98f, 1.0f) * 0.55f;
    color += fresnel * float3(0.45f, 0.65f, 0.9f) * 0.6f;

    return float4(saturate(color), 0.92f);
}
