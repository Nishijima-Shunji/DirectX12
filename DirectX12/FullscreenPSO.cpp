#include "FullScreenPSO.h"
#include <stdexcept>

using Microsoft::WRL::ComPtr;

void FullscreenPSO::SetShaders(const wchar_t* vsCsoPath, const wchar_t* psCsoPath)
{
    HRESULT hr = D3DReadFileToBlob(vsCsoPath, &vs_);
    if (FAILED(hr)) throw std::runtime_error("VScso load failed");
    hr = D3DReadFileToBlob(psCsoPath, &ps_);
    if (FAILED(hr)) throw std::runtime_error("PScso load failed");
}

bool FullscreenPSO::Create(ID3D12Device* dev)
{
    if (!dev || !rs_ || !vs_ || !ps_) return false;

    D3D12_GRAPHICS_PIPELINE_STATE_DESC p = {};
    p.pRootSignature = rs_;
    p.VS = { vs_->GetBufferPointer(), vs_->GetBufferSize() };
    p.PS = { ps_->GetBufferPointer(), ps_->GetBufferSize() };

    // フルスクリーン向けの定形
    p.InputLayout.pInputElementDescs = nullptr;
    p.InputLayout.NumElements = 0;
    p.PrimitiveTopologyType = topo_;
    p.SampleMask = 0xFFFFFFFF;

    // ラスタライザ（D3D12デフォルト相当・ヘルパー不使用）
    D3D12_RASTERIZER_DESC rast = {};
    rast.FillMode = D3D12_FILL_MODE_SOLID;
    rast.CullMode = D3D12_CULL_MODE_BACK;
    rast.FrontCounterClockwise = FALSE;
    rast.DepthBias = 0;
    rast.DepthBiasClamp = 0.0f;
    rast.SlopeScaledDepthBias = 0.0f;
    rast.DepthClipEnable = TRUE;
    rast.MultisampleEnable = FALSE;
    rast.AntialiasedLineEnable = FALSE;
    rast.ForcedSampleCount = 0;
    rast.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
    p.RasterizerState = rast;

    // 深度完全OFF＋DSV不要
    D3D12_DEPTH_STENCIL_DESC ds = {};
    ds.DepthEnable = FALSE;
    ds.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    ds.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    ds.StencilEnable = FALSE;
    ds.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
    ds.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
    ds.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
    ds.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    ds.BackFace = ds.FrontFace;
    p.DepthStencilState = ds;
    p.DSVFormat = dsv_; // 既定 UNKNOWN

    // ブレンドプリセット
    D3D12_BLEND_DESC blend = {};
    blend.AlphaToCoverageEnable = FALSE;
    blend.IndependentBlendEnable = FALSE;
    D3D12_RENDER_TARGET_BLEND_DESC rt = {};
    rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

    switch (blend_) {
    case Blend::Opaque:
        rt.BlendEnable = FALSE;
        break;
    case Blend::Alpha:
        rt.BlendEnable = TRUE;
        rt.SrcBlend = D3D12_BLEND_SRC_ALPHA;
        rt.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        rt.BlendOp = D3D12_BLEND_OP_ADD;
        rt.SrcBlendAlpha = D3D12_BLEND_ONE;
        rt.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
        rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        break;
    case Blend::Additive:
        rt.BlendEnable = TRUE;
        rt.SrcBlend = D3D12_BLEND_ONE;
        rt.DestBlend = D3D12_BLEND_ONE;
        rt.BlendOp = D3D12_BLEND_OP_ADD;
        rt.SrcBlendAlpha = D3D12_BLEND_ONE;
        rt.DestBlendAlpha = D3D12_BLEND_ONE;
        rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        break;
    }
    blend.RenderTarget[0] = rt;
    p.BlendState = blend;

    p.NumRenderTargets = 1;
    p.RTVFormats[0] = rtv_;
    p.SampleDesc.Count = 1;
    p.SampleDesc.Quality = 0;

    HRESULT hr = dev->CreateGraphicsPipelineState(&p, IID_PPV_ARGS(pso_.ReleaseAndGetAddressOf()));
    return SUCCEEDED(hr);
}
