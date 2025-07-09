#define PI 3.14159265358979323846f

// SPHParams �\���́ib0�j
cbuffer SPHParams : register(b0) {
    float restDensity;   // ���R��Ԃ̖��x
    float particleMass;  // ����
    float viscosity;     // �S���W��
    float stiffness;     // �����W��
    float radius;        // �e�����a
    float timeStep;      // ���ԃX�e�b�v
    uint  particleCount; // ���q��
};

cbuffer ViewProjCB : register(b1) {
    float4x4 viewProj;   // �r���[���e�s��
};

// GPU ��̗��q�\����
struct Particle {
    float3 position;
    float3 velocity;
};

struct ParticleMeta
{
    float x, y, r, pad;
};

// �O�t���[���̗��q�ǂݍ��݁it0�j
StructuredBuffer<Particle>    inParticles   : register(t0);
// �v�Z���ʂ̏������݁iu0�j
RWStructuredBuffer<Particle>  outParticles  : register(u0);
// ���^�{�[���ɕK�v�ȏ��ɕϊ�
RWStructuredBuffer<ParticleMeta> outMeta : register(u1);

// ========================================
//  ���C��
// ========================================

[numthreads(256,1,1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    // pow�����O�v�Z
    float radius2 = radius * radius;
    float radius6 = radius2 * radius2 * radius2;
    float radius9 = radius6 * radius2 * radius;
    
    uint i = id.x;
    if (i >= particleCount) return;
    // ========================================
    //  ���x�v�Z 
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
    //  �͌v�Z
    // ========================================
    float3 force = float3(0, -9.8f * density, 0);
    for (uint j = 0; j < particleCount; ++j) {
        if (i == j) continue;
        float3 rij = inParticles[i].position - inParticles[j].position;
        float  r   = length(rij);
        if (r > 0 && r < radius) {
            // ���͗�
            float coeff = -45.0/(PI * radius6) * (radius - r)*(radius - r);
            float3 grad = coeff * (rij / r);
            float pTerm = (pressure + stiffness * ( (density - restDensity) )) / (2*density);
            force += -particleMass * pTerm * grad;
            // �S����
            float lap = 45.0/(PI * radius6) * (radius - r);
            force += viscosity * particleMass * (inParticles[j].velocity - inParticles[i].velocity) * (lap / density);
        }
    }

    // ========================================
    //  �����v�Z
    // ========================================
    Particle p = inParticles[i];
    float3 accel = force / density;
    p.velocity += accel * timeStep;
    p.position += p.velocity * timeStep;

    // ========================================
    //  ���E����
    // ========================================
    if (p.position.x < -1 || p.position.x > 1) p.velocity.x *= -0.1f;
    if (p.position.y < -1 || p.position.y > 5) p.velocity.y *= -0.1f;
    if (p.position.z < -1 || p.position.z > 1) p.velocity.z *= -0.1f;

   // ���[���h���W�ifloat3�j
    float3 worldPos = p.position;

    // ���[���h �� �N���b�v���
    float4 clipPos = mul(float4(worldPos, 1.0f), viewProj);

    // NDC �ɐ��K��
    clipPos /= clipPos.w;

    // NDC �� UV
    float2 uv;
    uv.x = clipPos.x * 0.5f + 0.5f;
    uv.y = -clipPos.y * 0.5f + 0.5f; // Y���]�i���㌴�_�̃e�N�X�`���ɍ��킹��j

    // ��������
    ParticleMeta m;
    m.x = uv.x;
    m.y = uv.y;
    m.r = radius;
    m.pad = 0;

    // ========================================
    //  �o��
    // ========================================
    outMeta[i] = m;
}
