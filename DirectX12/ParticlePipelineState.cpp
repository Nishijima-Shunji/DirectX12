#include "ParticlePipelineState.h"
#include "Engine.h"
#include <d3dx12.h>
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")

ParticlePipelineState::ParticlePipelineState()
{
    // パイプラインステートの初期設定
    desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    desc.DepthStencilState.DepthEnable = FALSE;
    desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    desc.SampleMask = UINT_MAX;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
}

bool ParticlePipelineState::IsValid()
{
    return m_IsValid;
}

void ParticlePipelineState::SetInputLayout(D3D12_INPUT_LAYOUT_DESC layout)
{
    desc.InputLayout = layout;
}

void ParticlePipelineState::SetRootSignature(ID3D12RootSignature* rootSignature)
{
    desc.pRootSignature = rootSignature;
}

static HRESULT LoadOrCompileShader(const std::wstring& csoPath,
                                   const std::wstring& hlslPath,
                                   const char* entry,
                                   const char* target,
                                   ComPtr<ID3DBlob>& outBlob)
{
    HRESULT hr = D3DReadFileToBlob(csoPath.c_str(), outBlob.ReleaseAndGetAddressOf());
    if (FAILED(hr)) {
        UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
        flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
        ComPtr<ID3DBlob> err;
        hr = D3DCompileFromFile(hlslPath.c_str(), nullptr, nullptr,
                                entry, target, flags, 0,
                                outBlob.ReleaseAndGetAddressOf(), err.GetAddressOf());
        if (FAILED(hr)) {
            if (err) {
                printf("Shader compile error: %s\n", (char*)err->GetBufferPointer());
            }
        }
    }
    return hr;
}

void ParticlePipelineState::SetVS(std::wstring filePath)
{
    std::wstring hlsl = filePath;
    size_t pos = hlsl.find_last_of(L'.');
    if (pos != std::wstring::npos) hlsl.replace(pos, std::wstring::npos, L".hlsl");

    const char* entry = "VSMain";
    if (filePath.find(L"SimpleVS") != std::wstring::npos) entry = "vert";

    if (SUCCEEDED(LoadOrCompileShader(filePath, hlsl, entry, "vs_5_0", m_pVsBlob))) {
        desc.VS = CD3DX12_SHADER_BYTECODE(m_pVsBlob.Get());
    }
}

void ParticlePipelineState::SetPS(std::wstring filePath)
{
    std::wstring hlsl = filePath;
    size_t pos = hlsl.find_last_of(L'.');
    if (pos != std::wstring::npos) hlsl.replace(pos, std::wstring::npos, L".hlsl");

    const char* entry = "PSMain";
    if (filePath.find(L"SimplePS") != std::wstring::npos) entry = "pixel";

    if (SUCCEEDED(LoadOrCompileShader(filePath, hlsl, entry, "ps_5_0", m_pPSBlob))) {
        desc.PS = CD3DX12_SHADER_BYTECODE(m_pPSBlob.Get());
    }
}

void ParticlePipelineState::Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE type)
{
    desc.PrimitiveTopologyType = type;
    HRESULT hr = g_Engine->Device()->CreateGraphicsPipelineState(
        &desc, IID_PPV_ARGS(m_pPipelineState.ReleaseAndGetAddressOf()));
    if (FAILED(hr)) {
        printf("ParticlePipelineState creation failed HRESULT=0x%08X\n", hr);
        return;
    }

    m_IsValid = true;
}

ID3D12PipelineState* ParticlePipelineState::Get()
{
    return m_pPipelineState.Get();
}
