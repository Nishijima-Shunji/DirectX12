// ComputePipelineState.h
#pragma once
#include "ComPtr.h"
#include <d3dx12.h>
#include <string>

class ComputePipelineState {
private:
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    ComPtr<ID3D12PipelineState>        m_pso;
    ComPtr<ID3DBlob>                   m_csBlob;

public:
    ComputePipelineState() { desc.NodeMask = 0; }
    void SetRootSignature(ID3D12RootSignature* rs) { desc.pRootSignature = rs; }
    // .cso Çì«Ç›çûÇﬁ
    void SetCS(const std::wstring& path);
    void Create();
    ID3D12PipelineState* Get() const { return m_pso.Get(); }

};
