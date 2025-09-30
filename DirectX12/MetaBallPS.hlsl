cbuffer MetaCB : register(b0)
{
    float4x4 gInvViewProj;   // ビュー射影逆行列
    float4x4 gViewProj;      // ビュー射影行列
    float4   gCameraIso;     // xyz: カメラ位置, w: アイソ値
    float4   gParams;        // x: ステップ係数, y: 最大距離, z: 粒子数, w: 時間
};

struct VSOutput
{
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD;
};

struct ParticleMeta
{
    float3 pos;  // 粒子位置
    float  radius; // 粒子半径
};

StructuredBuffer<ParticleMeta> Particles : register(t0);

static const int kMaxStep = 96;

float EvaluateField(float3 p, out float3 grad)
{
    grad = float3(0.0, 0.0, 0.0);
    float value = 0.0;
    uint count = (uint)gParams.z;

    [loop]
    for (uint i = 0; i < count; ++i)
    {
        ParticleMeta meta = Particles[i];
        float3 d = p - meta.pos;
        float  r = max(meta.radius, 1e-3);
        float  r2 = r * r;
        float  dist2 = dot(d, d);
        float  influence = exp(-dist2 / r2);
        value += influence;
        grad += (-2.0 * influence / r2) * d;
    }

    return value;
}

float3 ShadeSurface(float3 position, float3 normal, float3 cameraPos)
{
    float3 lightDir = normalize(float3(-0.4, 0.85, 0.3));
    float diff = saturate(dot(normal, lightDir));
    float3 baseColor = float3(0.1, 0.35, 0.6);
    float3 highlight = float3(0.7, 0.9, 1.0);
    float3 viewDir = normalize(cameraPos - position);
    float spec = pow(saturate(dot(reflect(-lightDir, normal), viewDir)), 32.0);
    float fresnel = pow(1.0 - saturate(dot(normal, viewDir)), 5.0);
    float3 color = baseColor * (0.35 + diff * 0.65);
    color = lerp(color, highlight, fresnel * 0.5);
    color += highlight * spec * 0.25;
    return saturate(color);
}

struct PSOutput
{
    float4 color : SV_TARGET;
    float  depth : SV_DEPTH;
};

PSOutput main(VSOutput IN)
{
    float4 clip = float4(IN.uv * 2.0 - 1.0, 0.0, 1.0);
    float4 world = mul(gInvViewProj, clip);
    world /= world.w;

    float3 ro = gCameraIso.xyz;
    float3 rd = normalize(world.xyz - ro);

    float3 pos = ro;
    float iso = gCameraIso.w;
    float maxDistance = gParams.y;
    float stepScale = gParams.x;

    float travelled = 0.0;
    bool hit = false;
    float3 grad = float3(0.0, 0.0, 0.0);

    [loop]
    for (int step = 0; step < kMaxStep && travelled < maxDistance; ++step)
    {
        float3 localGrad;
        float value = EvaluateField(pos, localGrad);
        float field = value - iso;

        if (field >= 0.0)
        {
            grad = localGrad;
            hit = true;
            break;
        }

        float stepLen = max(stepScale * (iso - value), 0.02);
        pos += rd * stepLen;
        travelled += stepLen;
    }

    PSOutput OUT;

    if (!hit)
    {
        OUT.color = float4(0.02, 0.05, 0.09, 1.0);
        OUT.depth = 1.0;
        return OUT;
    }

    float3 normal = normalize(grad);
    if (!any(isfinite(normal)))
    {
        normal = float3(0.0, 1.0, 0.0);
    }

    float3 color = ShadeSurface(pos, normal, ro);

    float4 clipPos = mul(gViewProj, float4(pos, 1.0));
    float depth = saturate(clipPos.z / clipPos.w);

    OUT.color = float4(color, 1.0);
    OUT.depth = depth;
    return OUT;
}
