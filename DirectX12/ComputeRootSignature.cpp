#pragma once
#include "ComPtr.h"
#include <d3dx12.h>
#include <string>
#include "Engine.h"

class ComputePipelineState {
public:
    ComputePipelineState() { desc.NodeMask = 0; }
    void SetRootSignature(ID3D12RootSignature* rs) { desc.pRootSignature = rs; }
    // .cso Çì«Ç›çûÇﬁ
    void SetCS(const std::wstring& path) {
        HRESULT hr = D3DReadFileToBlob(path.c_str(),
            m_csBlob.GetAddressOf());
        if (FAILED(hr)) {
            wprintf(L"CSì«Ç›çûÇ›é∏îs: %ls\n", path.c_str());
            return;
        }
        desc.CS = CD3DX12_SHADER_BYTECODE(m_csBlob.Get());
    }
    void Create() {
        HRESULT hr = g_Engine->Device()->CreateComputePipelineState(
            &desc, IID_PPV_ARGS(&m_pso));
        if (FAILED(hr)) {
            wprintf(L"Compute PSOê∂ê¨é∏îs 0x%08X\n", hr);
        }
    }
    ID3D12PipelineState* Get() const { return m_pso.Get(); }

private:
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    ComPtr<ID3D12PipelineState>        m_pso;
    ComPtr<ID3DBlob>                   m_csBlob;
};
