
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
    // スクリーン座標(0-1)を射影空間(-1～1)へ戻し、視線方向の位置を推定
    // 1. viewCenter.zを用いて現在ピクセルの視空間位置(xy)を復元
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float viewZ = viewCenter.z;
    float2 viewPosXY;
    viewPosXY.x = (ndc.x * viewZ) / proj._11;
    viewPosXY.y = (ndc.y * viewZ) / proj._22;

    // 2. 球中心との差分から画面上の半径内か判定
    float2 offset = viewPosXY - viewCenter.xy;
    float rr = radius * radius;
    float len2 = dot(offset, offset);
    if (len2 > rr)
    {
        // 半径外は球に当たらないためピクセル破棄
        clip(-1);
    }

    // 3. 球面方程式 (x^2 + y^2 + z^2 = r^2) から z 成分を解き、手前側の解を採用
    float zOffset = sqrt(rr - len2);
    float zNear = viewCenter.z - zOffset;
    float zFar = viewCenter.z + zOffset;
    float bestZ = (abs(zNear) < abs(zFar)) ? zNear : zFar;

    // 4. InterlockedMin 用に距離は正数深度として返却
    return abs(bestZ);
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
    uint2 p = uint2(svpos.xy);

    // 深度はuintとして格納しているのでasfloatで復元
    uint du = FluidDepth.Load(int3(p, 0));
    float d = asfloat(du);
    if (d == 0.0)
        discard; // 事前クリア次第で条件は調整

    // 厚みはRTVに描いたテクスチャをSRVとして読む（t3想定）
    float t = ThicknessTex.Load(int3(p, 0));

    float3 N = normalize(FluidNormal.Load(int3(p, 0)).xyz * 2 - 1);

    // Fresnel（Schlick）
    float3 V = float3(0, 0, 1); // 簡易。実際は view 空間の -normalize(viewPos) 推奨
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


