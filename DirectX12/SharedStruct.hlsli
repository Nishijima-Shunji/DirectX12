#ifndef SHARED_STRUCT_HLSLI
#define SHARED_STRUCT_HLSLI

// ƒJƒƒ‰Œn
cbuffer CameraCB : register(b0)
{
    float4x4 proj;
    float4x4 view;
    float2 screenSize;
    float nearZ;
    float farZ;
    float3 iorF0; // —á: (0.02,0.02,0.02)
    float absorb; // ‹zûŒW”
}

// —¬‘Ìƒpƒ‰ƒ[ƒ^
cbuffer SPHParamsCB : register(b1)
{
    float restDensity;
    float particleMass;
    float viscosity;
    float stiffness;
    float radius;
    float timeStep;
    uint particleCount;
    uint pad0;
    float3 gridMin;
    uint pad1;
    uint3 gridDim;
    uint pad2;
}

#endif