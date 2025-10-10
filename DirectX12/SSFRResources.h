#pragma once
#include <d3d12.h>
#include <wrl.h>
#include <cstdint>

// SSFR内部リソース（中間RTとディスクリプタ）の管理クラス。
class SSFRResources {
public:
    enum SrvSlot : uint32_t {
        SrvDepthRaw = 0,
        SrvDepthBlurX,
        SrvDepthBlurY,
        SrvThickness,
        SrvThicknessBlur,
        SrvSceneColor,
        SrvSceneDepth,
        SrvEnvironment,
        SrvParticlePositions,
        SrvCount
    };

    enum UavSlot : uint32_t {
        UavDepthBlurX = 0,
        UavDepthBlurY,
        UavGeneric,
        UavCount
    };

    void Initialize(ID3D12Device* device, DXGI_FORMAT swapFormat);
    void Resize(ID3D12Device* device, uint32_t width, uint32_t height, bool thicknessHalfRes);
    void SetSceneResources(ID3D12Device* device, ID3D12Resource* sceneColor, ID3D12Resource* sceneDepth, ID3D12Resource* envCube);

    D3D12_CPU_DESCRIPTOR_HANDLE GetSrvCPU(SrvSlot slot) const;
    D3D12_GPU_DESCRIPTOR_HANDLE GetSrvGPU(SrvSlot slot) const;
    D3D12_CPU_DESCRIPTOR_HANDLE GetUavCPU(UavSlot slot) const;
    D3D12_GPU_DESCRIPTOR_HANDLE GetUavGPU(UavSlot slot) const;

    D3D12_CPU_DESCRIPTOR_HANDLE GetRtvDepthRaw() const;
    D3D12_CPU_DESCRIPTOR_HANDLE GetRtvThickness() const;

    ID3D12DescriptorHeap* GetSrvUavHeap() const { return mSrvUavHeap.Get(); }

    ID3D12Resource* GetDepthRaw() const { return mDepthRaw.Get(); }
    ID3D12Resource* GetDepthBlurX() const { return mDepthBlurX.Get(); }
    ID3D12Resource* GetDepthBlurY() const { return mDepthBlurY.Get(); }
    ID3D12Resource* GetThickness() const { return mThickness.Get(); }
    ID3D12Resource* GetThicknessBlur() const { return mThicknessBlur.Get(); }

private:
    void CreateHeaps(ID3D12Device* device);
    void CreateInternalTextures(ID3D12Device* device);
    void CreateDescriptors(ID3D12Device* device);
    void CreateSceneSRV(ID3D12Device* device, ID3D12Resource* resource, SrvSlot slot);

    DXGI_FORMAT mSwapFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
    uint32_t mWidth = 0;
    uint32_t mHeight = 0;
    bool mThicknessHalfRes = true;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mSrvUavHeap;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mRtvHeap;

    UINT mSrvDescriptorSize = 0;
    UINT mRtvDescriptorSize = 0;

    Microsoft::WRL::ComPtr<ID3D12Resource> mDepthRaw;
    Microsoft::WRL::ComPtr<ID3D12Resource> mDepthBlurX;
    Microsoft::WRL::ComPtr<ID3D12Resource> mDepthBlurY;
    Microsoft::WRL::ComPtr<ID3D12Resource> mThickness;
    Microsoft::WRL::ComPtr<ID3D12Resource> mThicknessBlur;

    Microsoft::WRL::ComPtr<ID3D12Resource> mSceneColor;
    Microsoft::WRL::ComPtr<ID3D12Resource> mSceneDepth;
    Microsoft::WRL::ComPtr<ID3D12Resource> mEnvCube;
};
