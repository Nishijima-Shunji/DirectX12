#ifndef SHARED_STRUCT_HLSLI
#define SHARED_STRUCT_HLSLI

    float2 framebufferSize; // UVvZt𑜓x֍킹
    float2 _pad;            // 萔obt@̃AC
// カメラ系
cbuffer CameraCB : register(b0)
{
    float4x4 proj;
    float4x4 view;
    float2 screenSize;
    float nearZ;
    float farZ;
    float3 iorF0; // 例: (0.02,0.02,0.02)
    float absorb; // 吸収係数
}

// 流体パラメータ
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