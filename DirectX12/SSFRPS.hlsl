
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

static const float CLEAR_DEPTH_VALUE = 1000000.0f; // 深度バッファ初期化値

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
    float2 local : TEXCOORD2;
};
VSOut VS_Particle(VSIn v)
{
    VSOut o;
    float4 wpos = float4(v.center, 1);
    float4 vCenter = mul(view, wpos);

    // ※カメラ行列の列ベクトルから右・上方向を取り出し、必ずカメラ正対のビルボードにする
    float3 camRight = normalize(float3(view._11, view._21, view._31));
    float3 camUp = normalize(float3(view._12, view._22, view._32));

    // pos.xy は [-1,1] のローカル座標と仮定し、半径でスケールしてクアッドを構成
    float2 local = v.pos.xy;
    float3 billboardOffset = (camRight * local.x + camUp * local.y) * v.radius;

    float3 viewPos = vCenter.xyz + billboardOffset;
    float4 clipPos = mul(proj, float4(viewPos, 1));

    o.pos = clipPos;
    o.viewPos = vCenter.xyz;
    o.radius = v.radius;
    o.local = local;
    return o;
}

uint PackFloatForAtomic(float value)
{
    // ※深度・厚みはいずれも正値なので IEEE754 のまま asuint しても大小関係が保たれる
    return asuint(value);
}

// ※テクスチャ UAV には float 用の原子加算が無いため、CAS で擬似的に実装する
void AtomicAddFloat(RWTexture2D<uint> tex, uint2 pixel, float value)
{
    uint expected = tex[pixel];
    for (;;)
    {
        float current = asfloat(expected);
        float summed = current + value;
        uint desired = asuint(summed);
        uint original;
        InterlockedCompareExchange(tex[pixel], expected, desired, original);
        if (original == expected)
        {
            break; // ※期待値と一致したので書き込み成功
        }
        expected = original; // ※他スレッドが更新した場合は再試行
    }
}

float sphereDepth(float2 local, float3 viewCenter, float radius, out float thickness)
{
    // local は [-1,1] のローカル座標。球面方程式から表面位置と厚みを求める
    float r2 = dot(local, local);
    if (r2 > 1.0f)
    {
        thickness = 0.0f;
        return CLEAR_DEPTH_VALUE; // ※円外は描画しない
    }

    float radiusSq = radius * radius;
    float surfaceSq = max(radiusSq - radiusSq * r2, 0.0f);
    float viewOffset = sqrt(surfaceSq);

    thickness = 2.0f * viewOffset; // ※ビューレイの前面＋背面を通過する距離

    // XMMatrixLookAtRH の右手系では +Z が奥なので、負の view.z を反転して距離にする
    float surfaceDepth = -(viewCenter.z) - viewOffset;
    return max(surfaceDepth, nearZ);
}

float4 PS_DepthThickness(VSOut i) : SV_TARGET
{
    float thickness;
    float depth = sphereDepth(i.local, i.viewPos, i.radius, thickness);
    if (depth >= CLEAR_DEPTH_VALUE)
    {
        discard; // ※ビルボード円外は早期終了
    }

    uint2 pixel = uint2(i.pos.xy);

    // ※原子 Min へ渡す際に asuint へ変換し、正値の比較順序を維持
    InterlockedMin(FluidDepth[pixel], PackFloatForAtomic(depth));

    if (thickness > 0.0f)
    {
        AtomicAddFloat(Thickness, pixel, thickness);
    }

    return float4(depth, thickness, 0.0f, 0.0f);
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
    uint packed = FluidDepth[px];
    float depth = asfloat(packed);
    if (depth <= 0.0f || depth >= CLEAR_DEPTH_VALUE)
    {
        return float3(0.0f, 0.0f, 0.0f); // ※未描画領域は原点へ退避
    }

    float2 pixel = (float2(px) + 0.5f) / screenSize;
    float2 ndc = float2(pixel.x * 2.0f - 1.0f, 1.0f - pixel.y * 2.0f);

    float vx = ndc.x * depth / proj._11;
    float vy = ndc.y * depth / proj._22;
    return float3(vx, vy, depth);
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


