#include "ComputePipelineState.h"
#include <d3dcompiler.h>
#include "Engine.h"

// シェーダーの読み込み
void ComputePipelineState::SetCS(const std::wstring& path) {
    HRESULT hr = D3DReadFileToBlob(path.c_str(), m_csBlob.GetAddressOf());
    if (FAILED(hr)) {
        // Fallback: compile from .hlsl when .cso is missing
        std::wstring hlsl = path;
        size_t pos = hlsl.find_last_of(L'.');
        if (pos != std::wstring::npos) hlsl.replace(pos, std::wstring::npos, L".hlsl");

        UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
        flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
        ComPtr<ID3DBlob> err;
        hr = D3DCompileFromFile(hlsl.c_str(), nullptr, nullptr, "CSMain", "cs_5_0", flags, 0,
                               m_csBlob.GetAddressOf(), err.GetAddressOf());
        if (FAILED(hr)) {
            if (err) wprintf(L"CS compile error: %hs\n", (char*)err->GetBufferPointer());
            return;
        }
    }
    desc.CS = CD3DX12_SHADER_BYTECODE(m_csBlob.Get());
}

// 生成
void ComputePipelineState::Create() {
    if (!m_csBlob) {
        wprintf(L"[Error] Compute shader is not loaded.\n");
        return;
    }

    // desc.CS は SetCS() で設定済みだが、安全のため再設定しておく
    desc.CS = CD3DX12_SHADER_BYTECODE(m_csBlob.Get());

    HRESULT hr = m_device->CreateComputePipelineState(
        &desc, IID_PPV_ARGS(m_pso.ReleaseAndGetAddressOf()));

    if (FAILED(hr)) {
        wprintf(L"Compute PSO生成失敗 0x%08X\n", hr);
    }
}