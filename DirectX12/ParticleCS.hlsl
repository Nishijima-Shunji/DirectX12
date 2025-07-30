#define PI 3.14159265358979323846f

// SPHParams 構造体（b0）
cbuffer SPHParams : register(b0) {
    float restDensity;   // 自然状態の密度
    float particleMass;  // 質量
    float viscosity;     // 粘性係数
    float stiffness;     // 剛性係数
    float radius;        // 影響半径
    float timeStep;      // 時間ステップ
    uint  particleCount; // 粒子数
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
    float x, y, r, pad;
};

// 前フレームの粒子読み込み（t0）
StructuredBuffer<Particle>    inParticles   : register(t0);
// 計算結果の書き込み（u0）
RWStructuredBuffer<Particle>  outParticles  : register(u0);
// メタボールに必要な情報に変換
RWStructuredBuffer<ParticleMeta> outMeta : register(u1);

// ========================================
//  メイン
// ========================================

[numthreads(256,1,1)]
void CSMain(uint3 id : SV_DispatchThreadID)
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
    for (uint j = 0; j < particleCount; ++j) {
        float3 rij = inParticles[i].position - inParticles[j].position;
        float  r   = length(rij);
        if (r < radius) {
            float x = radius * radius - r * r;
            density += particleMass * (315.0/(64.0 * PI * radius9)) * x * x * x;
        }
    }
    float pressure = stiffness * (density - restDensity);

    // ========================================
    //  力計算
    // ========================================
    float3 force = float3(0, -9.8f * density, 0);
    for (uint j = 0; j < particleCount; ++j) {
        if (i == j) continue;
        float3 rij = inParticles[i].position - inParticles[j].position;
        float  r   = length(rij);
        if (r > 0 && r < radius) {
            // 圧力力
            float coeff = -45.0/(PI * radius6) * (radius - r)*(radius - r);
            float3 grad = coeff * (rij / r);
            float pTerm = (pressure + stiffness * ( (density - restDensity) )) / (2*density);
            force += -particleMass * pTerm * grad;
            // 粘性力
            float lap = 45.0/(PI * radius6) * (radius - r);
            force += viscosity * particleMass * (inParticles[j].velocity - inParticles[i].velocity) * (lap / density);
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

   // ワールド座標（float3）
    float3 worldPos = p.position;

    // ワールド → クリップ空間
    float4 clipPos = mul(float4(worldPos, 1.0f), viewProj);

    // NDC に正規化
    clipPos /= clipPos.w;

    // NDC → UV
    float2 uv;
    uv.x = clipPos.x * 0.5f + 0.5f;
    uv.y = -clipPos.y * 0.5f + 0.5f; // Y反転（左上原点のテクスチャに合わせる）

    // 書き込み
    ParticleMeta m;
    m.x = uv.x;
    m.y = uv.y;
    m.r = radius;
    m.pad = 0;

    // ========================================
    //  出力
    // ========================================
    outMeta[i] = m;
}
