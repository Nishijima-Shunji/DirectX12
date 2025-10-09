#ifndef SHARED_STRUCT_HLSLI
#define SHARED_STRUCT_HLSLI

// ※PS 側で CBV b0 のみを参照できるよう、framebufferSize を CameraCB 内へ整理（RS 警告対策）
cbuffer CameraCB : register(b0)
{
    float4x4 proj;
    float4x4 view;
    float4x4 world;             // ワールド行列を共有して描画位置のズレを防ぐ
    float2 screenSize;          // 流体バッファの解像度（半解像度）
    float nearZ;                // カメラ近クリップ
    float farZ;                 // カメラ遠クリップ
    float3 iorF0;               // フレネル計算用の F0
    float absorb;               // Beer-Lambert の吸収係数
    float2 framebufferSize;     // フル解像度のフレームバッファ
    float refractionScale;      // 屈折オフセットの強さ
    float thicknessScale;       // 厚みに対する減衰係数
    float2 invScreenSize;       // 流体バッファの逆解像度
    float2 _pad;                // 16byte 境界を維持
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
