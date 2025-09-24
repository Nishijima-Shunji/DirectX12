struct Particle
{
    float3 x;
    float pad0;
    float3 v;
    float pad1;
    float lambda;
    float density;
    float3 x_pred;
    float pad2;
};

StructuredBuffer<Particle> gParticlesIn : register(t0);
RWStructuredBuffer<Particle> gParticlesOut : register(u0);
StructuredBuffer<uint> gSortedIndices : register(t1);
StructuredBuffer<uint> gCellStart : register(t2);
StructuredBuffer<uint> gCellEnd : register(t3);

cbuffer PBFParams : register(b0)
{
    float dt, restDensity, mass, h;
    uint numParticles, solverIterations;
    float xsphC, sCorrK, sCorrN, deltaQ;
    float3 gravity;
    float epsilon;
    float cellSize;
    uint3 gridDim;
    float3 gridMin;
    float _pad_;
};

// 段階切替用：0=Predict, 1=Lambda, 2=DeltaP, 3=Velocity
cbuffer PassCB : register(b1){
    int passnum;
    }

#ifndef THREAD_GROUP_SIZE
#define THREAD_GROUP_SIZE 128
#endif


// ===== SPHカーネル =====
float W_poly6(float r, float h)
{
    if (r >= h)
        return 0.0;
    float x = (h * h - r * r);
    const float c = 315.0 / (64.0 * 3.14159265 * pow(h, 9));
    return c * x * x * x;
}

float3 gradW_spiky(float3 rij, float r, float h)
{
    if (r <= 1e-6 || r >= h)
        return float3(0, 0, 0);
    const float c = -45.0 / (3.14159265 * pow(h, 6));
    return c * pow(h - r, 2) * (rij / max(r, 1e-6));
}

// ===== 近傍ユーティリティ =====
uint3 cellCoord(float3 p)
{
    int3 c = int3(floor((p - gridMin) / cellSize));
    int3 g = int3(gridDim);
    c = clamp(c, int3(0, 0, 0), g - 1);
    return uint3(c);
}

uint cellHash(uint3 c)
{
    return (c.z * gridDim.y + c.y) * gridDim.x + c.x;
}

[numthreads(128, 1, 1)]
void CS_Predict(uint3 DTid : SV_DispatchThreadID)
{
    uint i = DTid.x;
    if (i >= numParticles)
        return;

    Particle p = gParticlesIn[i];

    // 予測位置
    p.v += gravity * dt;
    p.x_pred = p.x + p.v * dt;

    // 密度推定
    float rho = 0.0;
    uint3 cc = cellCoord(p.x_pred);

    [unroll]
    for (int dz = -1; dz <= 1; ++dz)
    [unroll]
        for (int dy = -1; dy <= 1; ++dy)
    [unroll]
            for (int dx = -1; dx <= 1; ++dx)
            {
                int3 off = int3(dx, dy, dz);
                int3 ci = int3(cc) + off;
                int3 g = int3(gridDim);
                if (any(ci < int3(0, 0, 0)) || any(ci >= g))
                    continue;

                uint3 nc = uint3(ci);
                uint hsh = cellHash(nc);
                uint s = gCellStart[hsh];
                uint e = gCellEnd[hsh];

                for (uint k = s; k < e; ++k)
                {
                    uint j = gSortedIndices[k];
                    float3 rij = p.x_pred - gParticlesIn[j].x_pred; // 初回は x でも可
                    float r = length(rij);
                    rho += mass * W_poly6(r, h);
                }
            }

    p.density = max(rho, 1e-6);
    p.lambda = 0.0;
    gParticlesOut[i] = p;
}

[numthreads(128, 1, 1)]
void CS_Lambda(uint3 DTid : SV_DispatchThreadID)
{
    uint i = DTid.x;
    if (i >= numParticles)
        return;

    Particle p = gParticlesOut[i];

    // C_i = ρ/ρ0 - 1
    float Ci = p.density / restDensity - 1.0;

    // ∑|∇C|^2
    float3 gradCi = float3(0, 0, 0);

    uint3 cc = cellCoord(p.x_pred);
    [unroll]
    for (int dz = -1; dz <= 1; ++dz)
    [unroll]
        for (int dy = -1; dy <= 1; ++dy)
    [unroll]
            for (int dx = -1; dx <= 1; ++dx)
            {
                int3 off = int3(dx, dy, dz);
                int3 ci = int3(cc) + off;
                int3 g = int3(gridDim);
                if (any(ci < int3(0, 0, 0)) || any(ci >= g))
                    continue;

                uint3 nc = uint3(ci);
                uint hsh = cellHash(nc);
                uint s = gCellStart[hsh];
                uint e = gCellEnd[hsh];

                for (uint k = s; k < e; ++k)
                {
                    uint j = gSortedIndices[k];
                    float3 rij = p.x_pred - gParticlesOut[j].x_pred;
                    float r = length(rij);
                    float3 gradW = (1.0 / restDensity) * gradW_spiky(rij, r, h);
                    gradCi += gradW;
                }
            }

    float sumGrad2 = dot(gradCi, gradCi);
    float lambda = -Ci / (sumGrad2 + epsilon);

    p.lambda = lambda;
    gParticlesOut[i] = p;
}

[numthreads(128, 1, 1)]
void CS_DeltaP(uint3 DTid : SV_DispatchThreadID)
{
    uint i = DTid.x;
    if (i >= numParticles)
        return;

    Particle p = gParticlesOut[i];
    float3 dp = float3(0, 0, 0);

    uint3 cc = cellCoord(p.x_pred);
    [unroll]
    for (int dz = -1; dz <= 1; ++dz)
    [unroll]
        for (int dy = -1; dy <= 1; ++dy)
    [unroll]
            for (int dx = -1; dx <= 1; ++dx)
            {
                int3 off = int3(dx, dy, dz);
                int3 ci = int3(cc) + off;
                int3 g = int3(gridDim);
                if (any(ci < int3(0, 0, 0)) || any(ci >= g))
                    continue;

                uint3 nc = uint3(ci);
                uint hsh = cellHash(nc);
                uint s = gCellStart[hsh];
                uint e = gCellEnd[hsh];

                for (uint k = s; k < e; ++k)
                {
                    uint j = gSortedIndices[k];
                    if (j == i)
                        continue;

                    Particle q = gParticlesOut[j];
                    float3 rij = p.x_pred - q.x_pred;
                    float r = length(rij);

            // s_corr（クラスタ抑制）
                    float corr = 0.0;
                    if (deltaQ > 0.0)
                    {
                        float w = W_poly6(r, h);
                        float wq = W_poly6(deltaQ, h);
                        corr = sCorrK * pow(w / max(wq, 1e-12), sCorrN);
                    }

                    float3 gradW = gradW_spiky(rij, r, h);
                    dp += (p.lambda + q.lambda + corr) * gradW;
                }
            }

    p.x_pred += dp;

    // 床SDF（y=0）
    if (p.x_pred.y < 0.0)
    {
        p.x_pred.y = 0.0;
    }

    gParticlesOut[i] = p;
}

[numthreads(128, 1, 1)]
void CS_Velocity(uint3 DTid : SV_DispatchThreadID)
{
    uint i = DTid.x;
    if (i >= numParticles)
        return;

    Particle p = gParticlesOut[i];

    // 速度
    float3 v = (p.x_pred - p.x) / dt;

    // XSPH
    float3 xsph = float3(0, 0, 0);
    uint3 cc = cellCoord(p.x_pred);

    [unroll]
    for (int dz = -1; dz <= 1; ++dz)
    [unroll]
        for (int dy = -1; dy <= 1; ++dy)
    [unroll]
            for (int dx = -1; dx <= 1; ++dx)
            {
                int3 off = int3(dx, dy, dz);
                int3 ci = int3(cc) + off;
                int3 g = int3(gridDim);
                if (any(ci < int3(0, 0, 0)) || any(ci >= g))
                    continue;

                uint3 nc = uint3(ci);
                uint hsh = cellHash(nc);
                uint s = gCellStart[hsh];
                uint e = gCellEnd[hsh];

                for (uint k = s; k < e; ++k)
                {
                    uint j = gSortedIndices[k];
                    float3 rij = p.x_pred - gParticlesOut[j].x_pred;
                    float r = length(rij);
                    float w = W_poly6(r, h);
                    xsph += (gParticlesOut[j].v - v) * w;
                }
            }

    v += xsphC * xsph;

    p.v = v;
    p.x = p.x_pred;
    gParticlesOut[i] = p;
}

[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= numParticles)
        return;

    // 段階をCBで切替（mainは常に同じ）
    if (passnum == 0)
    CS_Predict(DTid);
    else
    if (passnum == 1)
    CS_Lambda(DTid);
    else
    if (passnum == 2)
    CS_DeltaP(DTid);
    else
    if (passnum == 3)
    CS_Velocity(DTid);
}