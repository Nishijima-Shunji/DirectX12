#define PI 3.14159265358979323846f
#define MAX_PARTICLES_PER_CELL 64

// SPHParams 構造体（b0）
cbuffer SPHParams : register(b0) {
    float restDensity;   // 自然状態の密度
    float particleMass;  // 質量
    float viscosity;     // 粘性係数
    float stiffness;     // 剛性係数
    float radius;        // 影響半径
    float timeStep;      // 時間ステップ
    uint  particleCount; // 粒子数
    uint  pad0;
    float3 gridMin;
    float  pad1;
    uint3  gridDim;
    uint   pad2;
};

cbuffer ViewProjCB : register(b1) {
    float4x4 viewProj;   // ビュー投影行列
};

// GPU 上の粒子構造体
struct Particle {
    float3 position;
    float3 velocity;
};

struct ParticleMeta
{
    float3 pos; // ワールド空間位置
    float   r;  // 半径
};

// 前フレームの粒子読み込み（t0）
StructuredBuffer<Particle>    inParticles   : register(t0);
// 計算結果の書き込み（u0）
RWStructuredBuffer<Particle>  outParticles  : register(u0);
// メタボールに必要な情報に変換
RWStructuredBuffer<ParticleMeta> outMeta : register(u1);
RWStructuredBuffer<uint>  gridCount : register(u2);
RWStructuredBuffer<uint>  gridTable : register(u3);

// ========================================
//  メイン
// ========================================

[numthreads(256,1,1)]
void main(uint3 id : SV_DispatchThreadID)
{
    // powを事前計算
    float radius2 = radius * radius;
    float radius6 = radius2 * radius2 * radius2;
    float radius9 = radius6 * radius2 * radius;
    
    uint i = id.x;
    if (i >= particleCount) return;
    // ========================================
    //  密度計算 
    // ========================================
    float density = 0;
    int3 baseCell = int3(floor((inParticles[i].position - gridMin) / radius));
    for (int cx = baseCell.x - 1; cx <= baseCell.x + 1; ++cx) {
        if (cx < 0 || cx >= (int)gridDim.x) continue;
        for (int cy = baseCell.y - 1; cy <= baseCell.y + 1; ++cy) {
            if (cy < 0 || cy >= (int)gridDim.y) continue;
            for (int cz = baseCell.z - 1; cz <= baseCell.z + 1; ++cz) {
                if (cz < 0 || cz >= (int)gridDim.z) continue;
                uint cId = cx + gridDim.x * (cy + gridDim.y * cz);
                uint cnt = gridCount[cId];
                // gridCountはMAX_PARTICLES_PER_CELLを超える可能性があるため、
                // ここで飽和させて配列外アクセスを防ぐ。
                cnt = min(cnt, (uint)MAX_PARTICLES_PER_CELL);
                for (uint n = 0; n < cnt; ++n) {
                    uint j = gridTable[cId * MAX_PARTICLES_PER_CELL + n];
                    float3 rij = inParticles[i].position - inParticles[j].position;
                    float r2 = dot(rij, rij);
                    if (r2 < radius2) {
                        float x = radius2 - r2;
                        density += particleMass * (315.0/(64.0 * PI * radius9)) * x * x * x;
                    }
                }
            }
        }
    }
    
    density = max(density, 0.000001f); // 密度がゼロになるのを防ぎ、ゼロ除算を回避する
    float pressure = stiffness * (density - restDensity);

    // ========================================
    //  力計算
    // ========================================
    float3 force = float3(0, -9.8f * density, 0);
    // ループ変数を分けて記述し、HLSL側でのスコープ衝突を避ける
    for (int nx = baseCell.x - 1; nx <= baseCell.x + 1; ++nx) {
        if (nx < 0 || nx >= (int)gridDim.x) continue;
        for (int ny = baseCell.y - 1; ny <= baseCell.y + 1; ++ny) {
            if (ny < 0 || ny >= (int)gridDim.y) continue;
            for (int nz = baseCell.z - 1; nz <= baseCell.z + 1; ++nz) {
                if (nz < 0 || nz >= (int)gridDim.z) continue;
                uint neighborCellId = nx + gridDim.x * (ny + gridDim.y * nz);
                uint neighborCount = gridCount[neighborCellId];
                // 近傍セルについても同様に上限を守る。
                neighborCount = min(neighborCount, (uint)MAX_PARTICLES_PER_CELL);
                for (uint n = 0; n < neighborCount; ++n) {
                    uint j = gridTable[neighborCellId * MAX_PARTICLES_PER_CELL + n];
                    if (j == i) continue;
                    float3 rij = inParticles[i].position - inParticles[j].position;
                    float r2 = dot(rij, rij);
                    if (r2 > 0 && r2 < radius2) {
                        float r = sqrt(r2);
                        float coeff = -45.0/(PI * radius6) * (radius - r)*(radius - r);
                        float3 grad = coeff * (rij / r);
                        float pTerm = (pressure + stiffness * ( (density - restDensity) )) / (2*density);
                        force += -particleMass * pTerm * grad;
                        float lap = 45.0/(PI * radius6) * (radius - r);
                        float3 velocityDiff = inParticles[j].velocity - inParticles[i].velocity;
                        force += viscosity * particleMass * velocityDiff * (lap / density);
                    }
                }
            }
        }
    }

    // ========================================
    //  動き計算
    // ========================================
    Particle p = inParticles[i];
    float3 accel = force / density;
    p.velocity += accel * timeStep;
    p.position += p.velocity * timeStep;

    // ========================================
    //  境界処理
    // ========================================
    if (p.position.x < -1 || p.position.x > 1) p.velocity.x *= -0.1f;
    if (p.position.y < -1 || p.position.y > 5) p.velocity.y *= -0.1f;
    if (p.position.z < -1 || p.position.z > 1) p.velocity.z *= -0.1f;

    // 書き込み
    ParticleMeta m;
    m.pos = p.position;
    m.r   = radius;

    // ========================================
    //  出力
    // ========================================
    outParticles[i] = p;
    outMeta[i]      = m;
}
