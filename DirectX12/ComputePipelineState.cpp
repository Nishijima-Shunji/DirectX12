#include "ComputePipelineState.h"
#include <d3dcompiler.h>
#include "Engine.h"

// シェーダーの読み込み
void ComputePipelineState::SetCS(const std::wstring& path) {
    HRESULT hr = D3DReadFileToBlob(path.c_str(),
        m_csBlob.GetAddressOf());
    if (FAILED(hr)) {
        wprintf(L"CS読み込み失敗: %ls\n", path.c_str());
        return;
    }
    desc.CS = CD3DX12_SHADER_BYTECODE(m_csBlob.Get());
}

// 生成
void ComputePipelineState::Create() {
    HRESULT hr = g_Engine->Device()->CreateComputePipelineState(
        &desc, IID_PPV_ARGS(&m_pso));
    if (FAILED(hr)) {
        wprintf(L"Compute PSO生成失敗 0x%08X\n", hr);
    }
}