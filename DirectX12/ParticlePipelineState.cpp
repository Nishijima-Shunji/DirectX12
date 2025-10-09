#include "ParticlePipelineState.h"
#include "Engine.h"
#include <d3dx12.h>
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")

ParticlePipelineState::ParticlePipelineState()
{
    // パイプラインステートの初期設定
    desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE; // 背面を描かない粒子なのでカリング無効にするため
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    desc.DepthStencilState.DepthEnable = FALSE;                     // スプラットは深度テスト不要のため無効化
    desc.DepthStencilState.StencilEnable = FALSE;                    // ステンシルも使用しないため無効化
    desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    desc.SampleMask = UINT_MAX;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 2;                                      // 深度と厚みを同時出力するためRTVを2つ使用
    desc.RTVFormats[0] = DXGI_FORMAT_R32_FLOAT;                     // 前面線形深度を保持するフォーマット
    desc.RTVFormats[1] = DXGI_FORMAT_R16_FLOAT;                     // 粒子厚みを蓄積するフォーマット
    desc.DSVFormat = DXGI_FORMAT_UNKNOWN;                           // 深度バッファを使わないためDSVは未使用
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;

    // RTV0: 最前面の深度を取得するため最小値ブレンドを設定
    auto& r0 = desc.BlendState.RenderTarget[0];
    r0.BlendEnable = TRUE;
    r0.SrcBlend = D3D12_BLEND_ONE;
    r0.DestBlend = D3D12_BLEND_ONE;
    r0.BlendOp = D3D12_BLEND_OP_MIN;

    // RTV1: 粒子厚みを積算するため加算ブレンドを設定
    auto& r1 = desc.BlendState.RenderTarget[1];
    r1.BlendEnable = TRUE;
    r1.SrcBlend = D3D12_BLEND_ONE;
    r1.DestBlend = D3D12_BLEND_ONE;
    r1.BlendOp = D3D12_BLEND_OP_ADD;
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
