struct VSOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };

struct ParticleMeta { float3 pos; float r; };
StructuredBuffer<ParticleMeta> Particles : register(t0);

cbuffer Meta : register(b0)
{
    float4x4 invViewProj;
    float4x4 viewProj;
    float4 camRadius; // xyz: camera world, w: particle radius
    float4 isoCount;  // x: iso, y: count, z: step, w:unused
    float4 gridMinCell; // 未使用
    uint gridDimX, gridDimY, gridDimZ, totalCells; // 未使用
    float4 waterDeep;    // rgb, w=absorb係数
    float4 waterShallow; // rgb, w=foam閾値
    float4 shadingParams;// x:泡 y:反射係数 z:specPow w:time
};

#ifndef MB_MAX_STEPS
#define MB_MAX_STEPS 64
#endif

float3 skyColor(float3 dir)
{
    float t = saturate(dir.y * 0.5 + 0.5);
    float3 top = float3(0.45, 0.70, 1.00);
    float3 bot = float3(0.10, 0.18, 0.35);
    return lerp(bot, top, t);
}

// C2連続なスムースステップ核（滑らかなメタボール曲面用）
float kernelC2(float x)
{
    float x2 = x * x;
    float x3 = x2 * x;
    return x3 * (x * (x * 6.0 - 15.0) + 10.0);
}

// 粒子群からスカラー場を評価
float field(float3 p)
{
    float sum = 0.0;
    uint n = (uint)isoCount.y;
    [loop]
    for (uint i = 0; i < n; ++i)
    {
        float3 d = p - Particles[i].pos;
        float r = max(Particles[i].r, 1e-4);
        float dist = length(d);
        if (dist < r)
        {
            float t = saturate(1.0 - dist / r);
            sum += kernelC2(t);
        }
    }
    return sum;
}

// 有限差分で法線ベクトルを推定
float3 gradient(float3 p, float eps)
{
    float3 ex = float3(eps, 0, 0);
    float3 ey = float3(0, eps, 0);
    float3 ez = float3(0, 0, eps);
    float gx = field(p + ex) - field(p - ex);
    float gy = field(p + ey) - field(p - ey);
    float gz = field(p + ez) - field(p - ez);
    return float3(gx, gy, gz) / (2.0 * eps);
}

// 色収差風の揺らぎで水らしいきらめきを付加
float3 applyDispersion(float3 color, float thickness)
{
    float wave = sin(shadingParams.w * 2.0 + thickness * 6.0);
    float3 tint = float3(0.015, 0.01, 0.03) * wave;
    return saturate(color + tint);
}

float4 main(VSOut i) : SV_Target
{
    float2 ndc = i.uv * 2.0 - 1.0;
    float4 p0 = mul(invViewProj, float4(ndc, 0.0, 1.0)); p0.xyz /= p0.w;
    float4 p1 = mul(invViewProj, float4(ndc, 1.0, 1.0)); p1.xyz /= p1.w;
    float3 ro = p0.xyz;
    float3 rd = normalize(p1.xyz - p0.xyz);

    float stepLen = max(0.02 * isoCount.z, 0.005);
    float iso = max(isoCount.x, 1e-4);
    float t = 0.0;
    float prevT = 0.0;
    float prevField = 0.0;
    bool first = true;
    const float tMax = 4.0;

    // 早期終了付きレイマーチングループ
    [loop]
    for (int s = 0; s < MB_MAX_STEPS && t < tMax; ++s)
    {
        float3 pos = ro + rd * t;
        float value = field(pos);

        if (value >= iso)
        {
            float hitT = t;
            if (!first)
            {
                float denom = max(value - prevField, 1e-4);
                float ratio = saturate((iso - prevField) / denom);
                hitT = lerp(prevT, t, ratio);
            }

            float3 hitPos = ro + rd * hitT;

            float tExit = hitT;
            // 透過厚の近似：ヒット位置から数ステップ進んで抜ける地点を探す
            [loop]
            for (int k = 0; k < MB_MAX_STEPS; ++k)
            {
                tExit += stepLen;
                if (tExit >= tMax)
                {
                    break;
                }
                float sample = field(ro + rd * tExit);
                if (sample < iso)
                {
                    break;
                }
            }

            float thickness = max(tExit - hitT, stepLen);

            float eps = max(stepLen * 0.5, 0.002);
            // 法線は密度勾配から算出
            float3 N = gradient(hitPos, eps);
            float lenN = max(length(N), 1e-4);
            N /= lenN;

            float3 V = normalize(camRadius.xyz - hitPos);
            float3 L = normalize(float3(0.4, 0.8, 0.3));

            float F0 = 0.02;
            float VoN = saturate(dot(V, N));
            float Fschlick = F0 + (1.0 - F0) * pow(1.0 - VoN, 5.0);

            float3 R = reflect(-V, N);
            // 屈折が発生しない場合は反射色を流用して破綻を防ぐ
            float eta = 1.0 / 1.33;
            float3 T = refract(-V, N, eta);
            float lenT = dot(T, T);
            float3 colRefr = (lenT > 1e-4) ? skyColor(normalize(T)) : skyColor(R);
            float3 colRefl = skyColor(normalize(R));

            float att = exp(-waterDeep.w * thickness);
            colRefr *= att;

            float ndl = saturate(dot(N, L));
            float3 base = lerp(waterShallow.rgb, waterDeep.rgb, saturate(thickness * 0.7));
            float3 diffuse = base * (0.25 + 0.75 * ndl);

            float specPow = max(shadingParams.z, 8.0);
            float3 H = normalize(L + V);
            float spec = pow(saturate(dot(N, H)), specPow);

            float3 refrPart = lerp(diffuse, colRefr, 0.6);
            float reflectFactor = saturate(Fschlick * shadingParams.y);
            float3 color = lerp(refrPart, colRefl, reflectFactor);

            float foamMask = saturate(1.0 - smoothstep(0.0, waterShallow.w, thickness));
            // 薄い領域に泡色をブレンド
            color = lerp(color, 1.0.xxx, foamMask * shadingParams.x);

            color += spec * 0.12;
            color = applyDispersion(color, thickness);

            return float4(color, 1.0);
        }

        first = false;
        prevField = value;
        prevT = t;
        t += stepLen;
    }

    return float4(0.0, 0.0, 0.0, 1.0);
}

