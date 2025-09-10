cbuffer AccumCB : register(b0)
{
    float4x4 View;
    float4x4 Proj;
    float3 CameraRight;
    float pad0;
    float3 CameraUp;
    float pad1;
    float2 ViewportSize;    // (W, H)
    float2 _pad2;
    float PixelScale;       // 画面上での粒子半径→ピクセル換算係数
    float3 _pad3;
};

struct ParticleMeta
{
    float3 pos;
    float radius;
};

StructuredBuffer<ParticleMeta> Particles : register(t0);

struct VSOut
{
    float4 posH : SV_POSITION;
    float2 uv : TEXCOORD0;
    float rPix : TEXCOORD1; // ピクセル半径
};

VSOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    // インスタンス=粒子
    ParticleMeta P = Particles[iid];

    // 画面上の表示半径
    float rPix = max(1.0, P.radius * PixelScale);

    // ビルボードのローカルコーナー
    float2 corner = float2((vid == 1 || vid == 2) ? 1.0 : -1.0,
                           (vid == 2 || vid == 3) ? 1.0 : -1.0);

    // ワールド→ビュー→プロジェクション
    float4 posW = float4(P.pos, 1.0);
    float4 posV = mul(View, posW);
    float4 posH = mul(Proj, posV);

    // 画面空間オフセット（CameraRight/Upからスクリーンスペースへ）
    // posH は既に同次座標、スクリーンオフセットはNDCへ正規化後に加える
    float2 ndc = posH.xy / max(1e-6, posH.w);
    // ピクセル→NDCへの換算： 2/ViewportSize
    float2 px2ndc = 2.0 / ViewportSize;
    float2 ndcOffset = corner * rPix * px2ndc;
    ndc += ndcOffset;

    VSOut o;
    o.posH = float4(ndc * max(1e-6, posH.w), posH.z, posH.w);
    o.uv = (corner * 0.5f + 0.5f); // (0,0)-(1,1)
    o.rPix = rPix;
    return o;
}
