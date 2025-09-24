
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
RWTexture2D<uint> FluidDepth : register(u0); // R32_FLOAT
RWTexture2D<uint> Thickness : register(u1); // R16_FLOAT or R32_FLOAT
RWTexture2D<float4> FluidNormal : register(u2); // 8:8:8:8_UNORM でも可

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

float3 reconstructViewPos(uint2 px)
{
    float z = FluidDepth[px];
    // 逆射影でview座標を戻す（省略：実装は既存の逆射影関数を）
    return float3(0, 0, z);
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
    float d = FluidDepth[uint2(svpos.xy)];
    if (d == 0)
        discard;

    float t = Thickness[uint2(svpos.xy)];
    float3 N = normalize(FluidNormal[uint2(svpos.xy)].xyz * 2 - 1);

    // Fresnel（Schlick）
    float3 V = float3(0, 0, 1);
    float cosT = saturate(dot(N, V));
    float3 F = iorF0 + (1 - iorF0) * pow(1 - cosT, 5);

    // 屈折（簡易：背景をオフセットサンプル）
    float2 refrUV = uv + N.xy * 0.02; // 係数は調整
    float3 refr = SceneColor.SampleLevel(samplerLinearClamp, refrUV, 0).rgb;

    // Beer-Lambert
    float3 trans = exp(-absorb.xxx * t);
    float3 col = lerp(refr * trans, 1.0.xxx, F); // 反射は省略orキューブマップで加算

    return float4(col, 1);
}
