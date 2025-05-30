#define PI 3.14159265358979323846f

cbuffer SPHParams : register(b0) {
    float restDensity;
    float particleMass;
    float viscosity;
    float stiffness;
    float radius;
    float timeStep;
    uint  particleCount;
};

struct Particle {
    float3 position;
    float3 velocity;
    float  density;
    // float pad;       // 16�o�C�g���킹�p
    };

StructuredBuffer<Particle>    inParticles   : register(t0);
RWStructuredBuffer<Particle>  outParticles  : register(u0);

[numthreads(256,1,1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= particleCount) return;

    // --- ���t���[���̖��x���v�Z ---
    float density = 0;
    for (uint j = 0; j < particleCount; ++j) {
        float3 rij = inParticles[i].position - inParticles[j].position;
        float  r   = length(rij);
        if (r < radius) {
            float x = (radius*radius - r*r);
            density += particleMass * (315.0/(64.0*PI*pow(radius,9))) * x*x*x;
        }
    }

    // --- ���� ---
    float pressure = stiffness * (density - restDensity);

    // --- �͂̌v�Z ---
    float3 force = float3(0, -9.8f * density, 0);
    for (uint j = 0; j < particleCount; ++j) {
        if (i == j) continue;
        float3 rij = inParticles[i].position - inParticles[j].position;
        float  r   = length(rij);
        if (r > 0 && r < radius) {
            // neighbor �̖��x
            float density_j = inParticles[j].density;
            // ����
            float coeff = -45.0/(PI*pow(radius,6)) * (radius - r)*(radius - r);
            float3 grad = coeff * (rij / r);
            float  pTerm = (pressure + stiffness*(density_j - restDensity)) / (2*density);
            force += -particleMass * pTerm * grad;
            // �S����
            float lap = 45.0/(PI*pow(radius,6))*(radius - r);
            force += viscosity * particleMass * (inParticles[j].velocity - inParticles[i].velocity) * (lap / density);
        }
    }

    // --- �ϕ��E�o�E���X���� ---
    Particle p = inParticles[i];
    float3 accel = force / density;
    p.velocity += accel * timeStep;
    p.position += p.velocity * timeStep;
    if (p.position.y < -1) { p.position.y = -1; p.velocity.y *= -0.1; }

    // --- �V���� density ����������ŏI�� ---
    p.density = density;
    outParticles[i] = p;
}
