#include "MetaBallPipelineState.h"
#include <d3dcompiler.h>
#include "CD3DX12.h"

namespace graphics {

    void MetaBallPipeline::CreateRootSignature(
        ID3D12Device* device,
        ComPtr<ID3D12RootSignature>& outRootSig)
    {
        // SRV(t0) + CBV(b0)
        CD3DX12_DESCRIPTOR_RANGE ranges[1];
        ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);

        CD3DX12_ROOT_PARAMETER params[2];
        params[0].InitAsDescriptorTable(1, &ranges[0], D3D12_SHADER_VISIBILITY_PIXEL);
        params[1].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_PIXEL);

        CD3DX12_ROOT_SIGNATURE_DESC desc(
            _countof(params), params,
            0, nullptr,
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
        );

        ComPtr<ID3DBlob> blob, error;
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error);
        device->CreateRootSignature(
            0,
            blob->GetBufferPointer(),
            blob->GetBufferSize(),
            IID_PPV_ARGS(&outRootSig)
        );
    }

    void MetaBallPipeline::CreatePipelineState(
        ID3D12Device* device,
        ID3D12RootSignature* rootSig,
        DXGI_FORMAT rtvFormat,
        ComPtr<ID3D12PipelineState>& outPSO)
    {
        // シェーダーコンパイル
        auto vs = CompileShader(L"MetaBallVS.hlsl", nullptr, "main", "vs_5_0");
        auto ps = CompileShader(L"MetaBallPS.hlsl", nullptr, "main", "ps_5_0");

        // PSO 設定
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = rootSig;
        desc.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
        desc.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
        desc.InputLayout = { nullptr, 0 };
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = rtvFormat;
        desc.SampleMask = UINT_MAX;
        desc.SampleDesc.Count = 1;
        desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);

        device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&outPSO));
    }

} // namespace graphics
