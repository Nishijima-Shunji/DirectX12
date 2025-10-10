// SSFR共通定数バッファ定義。MLS-MPM粒子描画をGPUパイプラインで完結させるため。
cbuffer CameraCB    : register(b0)
{
    float4x4 gView;
    float4x4 gProj;
    float2   gScreenSize;
    float2   gInvScreenSize;
    float    gNearZ;
    float    gFarZ;
    float2   _pad0;
};

cbuffer DrawCB      : register(b1)
{
    float gParticleRadius;
    float3 _pad1;
};

cbuffer BlurCB      : register(b2)
{
    float gWorldBlurRadius;
    float gDepthSigma;
    uint  gEnableBilateral;
    float _pad2;
};

cbuffer CompositeCB : register(b3)
{
    float  gRefractScale;
    float3 gAbsorbK;
    float3 gFluidColor;
    uint   gFlags;
};

SamplerState gSampLinearClamp : register(s0);
