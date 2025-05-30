#pragma once
#include "ComPtr.h"
#include <d3dx12.h>
#include <string>

class ComputePipelineState
{

private:
    bool m_isValid = false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
    ComPtr<ID3D12PipelineState> m_pPSO;
    ComPtr<ID3DBlob> m_csBlob;

public:
    ComputePipelineState();
    bool IsValid() const;
    void SetRootSignature(ID3D12RootSignature* rootSignature);
    void SetCS(const std::wstring& filePath);
    void Create();  // ComputePipelineState Çê∂ê¨
    ID3D12PipelineState* Get() const;
};

