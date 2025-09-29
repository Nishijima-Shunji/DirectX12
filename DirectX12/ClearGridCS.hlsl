// グリッドの粒子カウントとテーブルをゼロ初期化するコンピュートシェーダー
#define MAX_PARTICLES_PER_CELL 64

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

RWStructuredBuffer<uint> gridCount : register(u2);
RWStructuredBuffer<uint> gridTable : register(u3);

[numthreads(256,1,1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint cellCount = gridDim.x * gridDim.y * gridDim.z;
    uint index = id.x;

    if (index < cellCount)
    {
        gridCount[index] = 0;
    }

    uint tableCount = cellCount * MAX_PARTICLES_PER_CELL;
    if (index < tableCount)
    {
        gridTable[index] = 0;
    }
}
