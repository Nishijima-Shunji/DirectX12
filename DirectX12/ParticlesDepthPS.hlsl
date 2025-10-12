// ※SSFR初段: 粒子を球とみなして線形深度のみを描画するピクセルシェーダー。
cbuffer DepthConstants : register(b0)
{
    float4x4 gView;    // ビュー行列（転置済み）
    float4x4 gProj;    // 射影行列（転置済み）
    float2   gClipZ;   // x=ニア平面、y=ファー平面
    float2   _pad0;    // 16byte境界を守るダミー
};

struct PSInput
{
    float4 position : SV_Position;
    float3 viewPos  : TEXCOORD0; // 粒子中心のビュー空間座標
    float2 uv       : TEXCOORD1; // スクリーン空間の正規化オフセット
    float  radius   : TEXCOORD2; // 粒子半径[m]
};

float main(PSInput input) : SV_Target0
{
    float2 disc = input.uv * 2.0f - 1.0f; // [-1,1]へ戻して円板上の座標を得る
    float r2 = dot(disc, disc);
    if (r2 > 1.0f)
    {
        discard; // 円外は球の外周なので破棄する
    }

    float radius = input.radius;
    float planar2 = r2 * radius * radius; // 平面上の距離^2
    float depthOffset = sqrt(max(radius * radius - planar2, 0.0f));
    float depthView = input.viewPos.z - depthOffset; // 球の前面深度
    return -depthView; // ビュー空間前方が負方向なので符号反転して線形深度として保存
}
