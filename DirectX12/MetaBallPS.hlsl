#define MAX_PARTICLES_PER_CELL 64

cbuffer MetaCB : register(b0)
{
    float4x4 invViewProj;
    float3 cam;
    float iso;
    float3 gridMin;
    uint count; // particleCountのエイリアス
    uint3 gridDim;
    float radius;
};

struct ParticleMeta
{
    float3 pos;
    float r;
};
struct VSOutput
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD;
};

StructuredBuffer<ParticleMeta> Particles : register(t0);
StructuredBuffer<uint> GridCount : register(t1);
StructuredBuffer<uint> GridTable : register(t2);

// グリッドを利用して高速化したField関数
float Field(float3 p, out float3 grad)
{
    float sum = 0;
    grad = float3(0, 0, 0);
    int3 cellId = (int3) floor((p - gridMin) / radius);
    bool early_exit = false;
    [loop]
    for (int z = cellId.z - 1; z <= cellId.z + 1; ++z)
    {
        if (z < 0 || z >= (int) gridDim.z)
            continue;
        for (int y = cellId.y - 1; y <= cellId.y + 1; ++y)
        {
            if (y < 0 || y >= (int) gridDim.y)
                continue;
            for (int x = cellId.x - 1; x <= cellId.x + 1; ++x)
            {
                if (x < 0 || x >= (int) gridDim.x)
                    continue;
                uint gridIndex = x + gridDim.x * (y + gridDim.y * z);
                uint countInCell = GridCount[gridIndex];
                uint loopCount = min(countInCell, MAX_PARTICLES_PER_CELL);
                for (uint i = 0; i < loopCount; ++i)
                {
                    uint particle_idx = GridTable[gridIndex * MAX_PARTICLES_PER_CELL + i];
                    float3 d_vec = p - Particles[particle_idx].pos;
                    float r2_particle = Particles[particle_idx].r * Particles[particle_idx].r;
                    float denom = dot(d_vec, d_vec) + 1e-6f;
                    sum += r2_particle / denom;
                    grad += (-2.0 * r2_particle * d_vec) / (denom * denom);
                    if (sum > iso)
                    {
                        early_exit = true;
                        break;
                    }
                }
                if (early_exit)
                    break;
            }
            if (early_exit)
                break;
        }
    }
    return sum - iso;
}

// main関数はオリジナルのロジックを維持
float4 main(VSOutput IN) : SV_TARGET
{
    float4 clip = float4(IN.uv * 2 - 1, 0, 1);
    float4 wp = mul(invViewProj, clip);
    wp /= wp.w;
    float3 ro = cam;
    float3 rd = normalize(wp.xyz - ro);

    float3 p = ro;
    float d = 0;
    float3 grad = float3(0, 0, 0);
    const int MAX_STEP = 80;
    [loop]
    for (int i = 0; i < MAX_STEP; ++i)
    {
        d = Field(p, grad);
        if (abs(d) < 0.01)
            break;
        p += rd * d * 0.4;
    }

    if (abs(d) >= 0.01)
    {
        discard;
    }

    float3 n = normalize(grad);
    float diff = saturate(dot(n, normalize(float3(1, 1, 1))));
    return float4(diff * float3(0.2, 0.4, 1.0), 1.0);
}