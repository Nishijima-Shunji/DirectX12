#pragma once
#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl.h>
#include <string>

class FullscreenPSO {
public:
    enum class Blend { Opaque, Alpha, Additive };

    FullscreenPSO() = default;

    void SetRootSignature(ID3D12RootSignature* rs) { rs_ = rs; }
    void SetShaders(const wchar_t* vsCsoPath, const wchar_t* psCsoPath);
    void SetFormats(DXGI_FORMAT rtv, DXGI_FORMAT dsv = DXGI_FORMAT_UNKNOWN) { rtv_ = rtv; dsv_ = dsv; }
    void SetBlend(Blend b) { blend_ = b; }
    void SetTopology(D3D12_PRIMITIVE_TOPOLOGY_TYPE type) { topo_ = type; }

    bool Create(ID3D12Device* dev);
    ID3D12PipelineState* Get() const { return pso_.Get(); }
    bool IsValid() const { return pso_ != nullptr; }

private:
    Microsoft::WRL::ComPtr<ID3DBlob> vs_, ps_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> pso_;
    ID3D12RootSignature* rs_ = nullptr;
    DXGI_FORMAT rtv_ = DXGI_FORMAT_R8G8B8A8_UNORM;
    DXGI_FORMAT dsv_ = DXGI_FORMAT_UNKNOWN;
    Blend blend_ = Blend::Opaque;
    D3D12_PRIMITIVE_TOPOLOGY_TYPE topo_ = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
};
