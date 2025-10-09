#ifndef SHARED_STRUCT_HLSLI
#define SHARED_STRUCT_HLSLI

// ※PS 側で CBV b0 のみを参照できるよう、framebufferSize を CameraCB 内へ整理（RS 警告対策）
cbuffer CameraCB : register(b0)
{
    float4x4 proj;
    float4x4 view;
    float4x4 world;            // ワールド行列を共有して描画位置のズレを防ぐ
    float2 screenSize;
    float nearZ;
    float farZ;
    float3 iorF0;                // 例: (0.02,0.02,0.02)
    float absorb;                // 吸収係数
    float2 framebufferSize;      // 合成用のフル解像度
    float2 bilateralSigma;       // バイラテラルフィルタの空間＆深度シグマ
    float2 bilateralNormalKernel;// 法線シグマとカーネル半径
    float2 _pad;                 // 16byte 境界を維持
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
