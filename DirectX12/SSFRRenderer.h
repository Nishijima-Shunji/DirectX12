#pragma once
#include <d3d12.h>
#include <wrl.h>
#include <DirectXMath.h>
#include <cstdint>
#include "SSFRResources.h"

// MLS-MPM流体のスクリーンスペース表現を実装するレンダラ。
struct SSFRSettings {
    bool  enableBilateral   = true;
    bool  enableRefraction  = true;
    bool  enableBeerLambert = true;
    bool  enableFresnel     = true;
    bool  thicknessHalfRes  = true;
    float particleRadius    = 0.03f;
    float blurWorldRadius   = 0.05f;
    float refractScale      = 0.02f;
    float absorbK[3]        = {0.2f, 0.1f, 0.05f};
    float fluidColor[3]     = {0.2f, 0.4f, 0.9f};
};

struct SSFRCamera {
    DirectX::XMFLOAT4X4 view;
    DirectX::XMFLOAT4X4 proj;
    float nearZ;
    float farZ;
    uint32_t width;
    uint32_t height;
};

struct SSFRParticleInputSRV {
    D3D12_CPU_DESCRIPTOR_HANDLE positionsSrvCPU;
    uint32_t count;
};

class SSFRRenderer {
public:
    void Initialize(ID3D12Device* device, DXGI_FORMAT swapChainFormat);
    void Resize(ID3D12Device* device, uint32_t width, uint32_t height);
    void SetEnvironmentCube(ID3D12Resource* cubeSRV);
    void SetSettings(const SSFRSettings& s);

    void Render(
        ID3D12GraphicsCommandList* cmd,
        const SSFRCamera& cam,
        const SSFRParticleInputSRV& fluid,
        ID3D12DescriptorHeap* srvUavHeap,
        D3D12_CPU_DESCRIPTOR_HANDLE rtvBackbuffer,
        D3D12_CPU_DESCRIPTOR_HANDLE dsvDepth,
        ID3D12Resource* sceneColorSRV = nullptr,
        ID3D12Resource* sceneLinearDepthSRV = nullptr);

private:
    void CreatePipelines(ID3D12Device* device);
    void UpdateCameraCB(const SSFRCamera& cam);
    void UpdateDrawCB();
    void UpdateBlurCB(float depthSigma);
    void UpdateCompositeCB();

    void RenderDepthPass(ID3D12GraphicsCommandList* cmd, const SSFRParticleInputSRV& fluid);
    void BlurDepth(ID3D12GraphicsCommandList* cmd, uint32_t width, uint32_t height);
    void RenderThickness(ID3D12GraphicsCommandList* cmd, const SSFRParticleInputSRV& fluid);
    void BlurThickness(ID3D12GraphicsCommandList* cmd, uint32_t width, uint32_t height);
    void Composite(ID3D12GraphicsCommandList* cmd, D3D12_CPU_DESCRIPTOR_HANDLE rtvBackbuffer);

    void EnsureDescriptorHeap(ID3D12Device* device);
    void PrepareSceneDescriptors(ID3D12Device* device, ID3D12Resource* sceneColor, ID3D12Resource* sceneDepth);

    struct ConstantBufferData {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        uint8_t* mapped = nullptr;
        size_t size = 0;
    };

    ConstantBufferData mCameraCB;
    ConstantBufferData mDrawCB;
    ConstantBufferData mBlurCB;
    ConstantBufferData mCompositeCB;

    void AllocateCB(ID3D12Device* device, ConstantBufferData& cb, size_t size, const char* debugName);
    void WriteCB(ConstantBufferData& cb, const void* data, size_t size);

    SSFRSettings mSettings{};
    DXGI_FORMAT mSwapFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
    uint32_t mWidth = 0;
    uint32_t mHeight = 0;

    Microsoft::WRL::ComPtr<ID3D12Device> mDevice;

    SSFRResources mResources;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mSrvUavHeap;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> mRSDepth;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mPSODepth;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> mRSBlur;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mPSOBlurX;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mPSOBlurY;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> mRSThickness;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mPSOThickness;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> mRSThicknessBlur;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mPSOThicknessBlurX;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mPSOThicknessBlurY;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> mRSComposite;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mPSOComposite;

    Microsoft::WRL::ComPtr<ID3D12Resource> mEnvironmentCube;

    float mDepthSigma = 0.04f;
    float mCurrentNear = 0.1f;
    float mCurrentFar = 1000.0f;
    bool mHasSceneColor = false;
    bool mHasSceneDepth = false;
    bool mHasEnvironment = false;
};
