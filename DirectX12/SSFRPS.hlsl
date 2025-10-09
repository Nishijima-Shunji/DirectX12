
// 1 粒子を球インスタンシングで描き、FluidDepth/Thicknessを作る
// 2 深度をバイラテラル平滑 -> 法線再構成
// 3 屈折 + Fresnel反射 + Beer-Lambert で合成

cbuffer CameraCB : register(b0)
{
    float4x4 proj; // 透視投影
    float4x4 view;
    float2 screenSize;
    float2 invScreenSize;
    float nearZ;
    float farZ;
    float3 iorF0; // IORから変換したF0（例: 水 0.02）
    float absorb; // 吸収係数（Beer-Lambert）
}

SamplerState samplerLinearClamp : register(s0);
Texture2D SceneColor : register(t0);
Texture2D SceneDepth : register(t1);
Texture2D<float> ThicknessTex : register(t3);
RWTexture2D<uint> FluidDepth : register(u1); // R32_FLOAT
RWTexture2D<uint> Thickness : register(u2); // R16_FLOAT or R32_FLOAT
RWTexture2D<float4> FluidNormal : register(u3); // 8:8:8:8_UNORM でも可

// 1 粒子スプラット（VS/PS 最小）
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
    // スクリーン空間の円板へ（簡易：ポイントスプライト代替）
    o.pos = mul(proj, vpos);
    o.viewPos = vpos.xyz;
    o.radius = v.radius;
    return o;
}

float sphereDepth(float2 uv, float3 viewCenter, float radius)
{
    // 画面上で球の深度を近似（スプライト内のZを求める）
    // ここでは簡略化：中心深度を採用（実運用はUVから球面方程式でZ補正）
    return -viewCenter.z;
}

float4 PS_DepthThickness(VSOut i) : SV_TARGET
{
    float2 uv = i.pos.xy * 0.5 / float2(screenSize.x * 0.5, screenSize.y * 0.5); // 略
    float d = sphereDepth(uv, i.viewPos, i.radius);
    InterlockedMin(FluidDepth[uint2(i.pos.xy)], d);             // 近い方の深度
    InterlockedAdd(Thickness[uint2(i.pos.xy)], i.radius * 0.5); // 簡易厚み
    return 0;
}

// 2 バイラテラル平滑 & 法線
[numthreads(8, 8, 1)]
void CS_Bilateral(uint3 id : SV_DispatchThreadID)
{
    // 近傍の深度を深度差重みで平滑化（実装省略：ガウス×深度差）
    // FluidDepth[...] = blurredDepth;
}

// FluidDepth に蓄積したビュー空間深度からビュー座標を復元する補助関数
// 引数 px : 画素インデックス（整数ピクセル）。FluidDepth の読み出しと NDC 変換に使用。
// 戻り値   : ビュー空間位置（z 成分は正の深度値）を返却し、長さ 0 の場合は未初期化画素を示す。
float3 reconstructViewPos(uint2 px)
{
    // FluidDepth は R32_FLOAT を asuint で格納しているため、Load → asfloat で復元する。
    uint rawDepth = FluidDepth.Load(int3(px, 0));
    float viewDepth = asfloat(rawDepth);
    if (viewDepth <= 0.0f)
    {
        // 深度 0 は初期値（流体未ヒット）を表すため、ゼロベクトルを返す。
        return float3(0.0f, 0.0f, 0.0f);
    }

    // ピクセル中心座標を 0〜1 の UV に正規化し、DirectX 規約に合わせて NDC（-1〜1）へ写像する。
    float2 pixelCenter = (float2(px) + 0.5f) * invScreenSize;
    float2 ndc;
    ndc.x = pixelCenter.x * 2.0f - 1.0f;
    ndc.y = 1.0f - pixelCenter.y * 2.0f;

    // 射影行列の逆行列を使ってクリップ空間からビュー空間へ戻す。
    // クリップ座標の Z=1 平面上の位置を逆射影し、得られた方向ベクトルを深度でスケールする。
    float4x4 invProj = inverse(proj);
    float4 clipPos = float4(ndc, 1.0f, 1.0f);
    float4 viewRay = mul(invProj, clipPos);
    viewRay.xyz /= viewRay.w;

    // viewRay.z は Z=1 面での距離なので、実際のビュー深度に合わせてスケールする。
    float scale = viewDepth / max(viewRay.z, 1e-6f);
    float3 viewPos = viewRay.xyz * scale;
    viewPos.z = viewDepth; // Z は視線方向の正の深度として保持する。

    return viewPos;
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

// 3 合成（PS）
float4 PS_Composite(float4 svpos : SV_POSITION, float2 uv : TEXCOORD) : SV_TARGET
{
    uint2 p = uint2(svpos.xy);

    // 深度はuintとして格納しているのでasfloatで復元
    uint du = FluidDepth.Load(int3(p, 0));
    float d = asfloat(du);
    if (d == 0.0)
        discard; // 事前クリア次第で条件は調整

    // 厚みはRTVに描いたテクスチャをSRVとして読む（t3想定）
    float t = ThicknessTex.Load(int3(p, 0));

    float3 N = normalize(FluidNormal.Load(int3(p, 0)).xyz * 2 - 1);

    // ビュー座標を再構成して視線方向を得る。流体の各画素に固有の入射角を反映するための更新。
    float3 viewPos = reconstructViewPos(p);
    float3 V = float3(0.0f, 0.0f, 1.0f);
    if (any(viewPos))
    {
        // viewPos が有効な場合はカメラからの視線ベクトルを正規化して利用する。
        V = normalize(-viewPos);
    }

    // Fresnel（Schlick）
    float cosT = saturate(dot(N, V));
    float3 F = iorF0 + (1 - iorF0) * pow(1 - cosT, 5);

    // 屈折（簡易オフセット）
    float2 refrUV = uv + N.xy * 0.02;
    float3 refr = SceneColor.SampleLevel(samplerLinearClamp, refrUV, 0).rgb;

    // Beer-Lambert
    float3 trans = exp(-absorb.xxx * t);
    float3 col = lerp(refr * trans, 1.0.xxx, F);
    return float4(col, 1);
}


//============================================================
// main
#if defined(PASS_PARTICLE_VS)
VSOut main(VSIn v)
{
    return VS_Particle(v);
}
#elif defined(PASS_PARTICLE_PS)
float4 main(VSOut i) : SV_TARGET
{
    return PS_DepthThickness(i);
}
#elif defined(PASS_BILATERAL_CS)
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    CS_Bilateral(id);
}
#elif defined(PASS_NORMAL_CS)
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    CS_Normal(id);
}
#elif defined(PASS_COMPOSITE_PS)
float4 main(float4 svpos : SV_POSITION, float2 uv : TEXCOORD) : SV_TARGET
{
    return PS_Composite(svpos, uv);
}
#else
#error "Define one of: PASS_PARTICLE_VS / PASS_PARTICLE_PS / PASS_BILATERAL_CS / PASS_NORMAL_CS / PASS_COMPOSITE_PS"
#endif


