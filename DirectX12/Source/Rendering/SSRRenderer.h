#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <memory>
#include <DirectXMath.h>
#include "ConstantBuffer.h"
#include "FluidWaterRenderer.h"

class Camera;

class SSRRenderer
{
public:
    SSRRenderer();
    bool Init(ID3D12Device* device, FluidWaterRenderer& fluid, UINT width, UINT height);
    void Resize(FluidWaterRenderer& fluid, UINT width, UINT height);
    void RenderSSR(ID3D12GraphicsCommandList* cmd, FluidWaterRenderer& fluid, const Camera& camera);

    ID3D12Resource* SSRTexture() const { return m_ssrTexture.Get(); }

private:
    void CreateTarget(ID3D12Device* device, FluidWaterRenderer& fluid, UINT width, UINT height);
    void CreatePipeline(ID3D12Device* device);
    void UpdateCameraCB(const Camera& camera, UINT ssrQuality);
    void Transition(ID3D12GraphicsCommandList* cmd, D3D12_RESOURCE_STATES after);

private:
    ID3D12Device* m_device = nullptr;
    UINT m_width = 0;
    UINT m_height = 0;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_ssrTexture;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pipelineState;
    std::unique_ptr<ConstantBuffer> m_cameraCB;
    D3D12_CPU_DESCRIPTOR_HANDLE m_ssrRTV{};
    D3D12_RESOURCE_STATES m_ssrState = D3D12_RESOURCE_STATE_COMMON;
};
