#define MAX_PARTICLES_PER_CELL 64
#define PI 3.14159265358979323846f

cbuffer SPHParams : register(b0)
{
    float restDensity;
    float particleMass;
    float viscosity;
    float stiffness;
    float radius;
    float timeStep;
    uint  particleCount;
    uint  pad0;
    float3 gridMin;
    float  pad1;
    uint3  gridDim;
    uint   pad2;
};

cbuffer ViewProjCB : register(b1)
{
    float4x4 viewProj;
};

struct Particle
{
    float3 position;
    float3 velocity;
};

StructuredBuffer<Particle> inParticles : register(t0);
RWStructuredBuffer<uint>   outGridCount : register(u2);
RWStructuredBuffer<uint>   outGridTable : register(u3);

[numthreads(256,1,1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint index = id.x;
    uint cellCount = gridDim.x * gridDim.y * gridDim.z;
    if (index < cellCount)
    {
        outGridCount[index] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (index >= particleCount) return;

    float3 pos = inParticles[index].position;
    int3 cell = int3(floor((pos - gridMin) / radius));
    cell = clamp(cell, int3(0,0,0), int3(gridDim) - 1);
    uint cellId = cell.x + gridDim.x * (cell.y + gridDim.y * cell.z);
    uint writeIdx;
    InterlockedAdd(outGridCount[cellId], 1, writeIdx);

    // 1セルが保持できる粒子数の上限を超えた場合は、
    // gridCountを飽和させて以降のアクセスが配列境界を越えないようにする。
    if (writeIdx >= MAX_PARTICLES_PER_CELL)
    {
        uint originalCount;
        // InterlockedExchangeは3つ目の引数に元の値を受け取る必要があるため、ダミーの変数を用意する。
        InterlockedExchange(outGridCount[cellId], MAX_PARTICLES_PER_CELL, originalCount);
        return;
    }

    outGridTable[cellId * MAX_PARTICLES_PER_CELL + writeIdx] = index;
}
