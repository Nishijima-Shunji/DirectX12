#include "PipelineState.h"
#include "Engine.h"
#include <d3dx12.h>
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")

PipelineState::PipelineState()
{
    // パイプラインステートの初期設定
    desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    desc.SampleMask = UINT_MAX;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
}

bool PipelineState::IsValid()
{
    return m_IsValid;
}

void PipelineState::SetInputLayout(D3D12_INPUT_LAYOUT_DESC layout)
{
    desc.InputLayout = layout;
}

void PipelineState::SetRootSignature(ID3D12RootSignature* rootSignature)
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

void PipelineState::SetVS(std::wstring filePath)
{
    std::wstring hlsl = filePath;
    size_t pos = hlsl.find_last_of(L'.');
    if (pos != std::wstring::npos) hlsl.replace(pos, std::wstring::npos, L".hlsl");

    const char* entry = "main";
    if (filePath.find(L"SimpleVS") != std::wstring::npos) entry = "vert";
    else if (filePath.find(L"ParticleVS") != std::wstring::npos) entry = "VSMain";

    if (SUCCEEDED(LoadOrCompileShader(filePath, hlsl, entry, "vs_5_0", m_pVsBlob))) {
        desc.VS = CD3DX12_SHADER_BYTECODE(m_pVsBlob.Get());
    }
}

void PipelineState::SetPS(std::wstring filePath)
{
    std::wstring hlsl = filePath;
    size_t pos = hlsl.find_last_of(L'.');
    if (pos != std::wstring::npos) hlsl.replace(pos, std::wstring::npos, L".hlsl");

    const char* entry = "main";
    if (filePath.find(L"SimplePS") != std::wstring::npos) entry = "pixel";
    else if (filePath.find(L"ParticlePS") != std::wstring::npos) entry = "PSMain";

    if (SUCCEEDED(LoadOrCompileShader(filePath, hlsl, entry, "ps_5_0", m_pPSBlob))) {
        desc.PS = CD3DX12_SHADER_BYTECODE(m_pPSBlob.Get());
    }
}

void PipelineState::SetDepthStencilFormat(DXGI_FORMAT format)
{
    desc.DSVFormat = format; // フレームバッファの深度バッファ形式とPSOを揃える
}

void PipelineState::Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE type)
{
    desc.PrimitiveTopologyType = type;

    // ==============SSFRの為に一時的にコメントアウト=========
    // 
    //D3D12_BLEND_DESC blendDesc = {};
    //blendDesc.AlphaToCoverageEnable = FALSE;
    //blendDesc.IndependentBlendEnable = FALSE;

    //auto& rtBlendDesc = blendDesc.RenderTarget[0];
    //rtBlendDesc.BlendEnable = TRUE;
    //rtBlendDesc.SrcBlend = D3D12_BLEND_SRC_ALPHA;
    //rtBlendDesc.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    //rtBlendDesc.BlendOp = D3D12_BLEND_OP_ADD;
    //rtBlendDesc.SrcBlendAlpha = D3D12_BLEND_ONE;
    //rtBlendDesc.DestBlendAlpha = D3D12_BLEND_ZERO;
    //rtBlendDesc.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    //rtBlendDesc.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

    //desc.BlendState = blendDesc;
    HRESULT hr = g_Engine->Device()->CreateGraphicsPipelineState(
        &desc, IID_PPV_ARGS(m_pPipelineState.ReleaseAndGetAddressOf()));
    if (FAILED(hr)) {
        printf("PipelineState creation failed HRESULT=0x%08X\n", hr);
        return;
    }

    m_IsValid = true;
}

ID3D12PipelineState* PipelineState::Get()
{
    return m_pPipelineState.Get();
}
