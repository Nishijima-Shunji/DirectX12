#include "SSRRenderer.h"
#include "Camera.h"
#include "Engine.h"
#include <d3dx12.h>
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;
using namespace DirectX;

SSRRenderer::SSRRenderer() = default;

bool SSRRenderer::Init(ID3D12Device* device, FluidWaterRenderer& fluid, UINT width, UINT height)
{
    if (!device)
    {
        return false;
    }
    m_device = device;
    m_cameraCB = std::make_unique<ConstantBuffer>(sizeof(FluidCameraConstants));
    m_width = width;
    m_height = height;

    CreatePipeline(device);
    CreateTarget(device, fluid, width, height);
    return true;
}

void SSRRenderer::Resize(FluidWaterRenderer& fluid, UINT width, UINT height)
{
    if (!m_device || (width == m_width && height == m_height))
    {
        return;
    }
    m_width = width;
    m_height = height;
    CreateTarget(m_device, fluid, width, height);
}

void SSRRenderer::RenderSSR(ID3D12GraphicsCommandList* cmd, FluidWaterRenderer& fluid, const Camera& camera)
{
    if (!cmd || !m_ssrTexture)
    {
        return;
    }

    UpdateCameraCB(camera, fluid.SSRQuality());
    Transition(cmd, D3D12_RESOURCE_STATE_RENDER_TARGET);

    cmd->OMSetRenderTargets(1, &m_ssrRTV, FALSE, nullptr);
    float clearColor[4] = { 0,0,0,0 };
    cmd->ClearRenderTargetView(m_ssrRTV, clearColor, 0, nullptr);

    D3D12_VIEWPORT viewport{ 0.0f, 0.0f, static_cast<float>(m_width), static_cast<float>(m_height), 0.0f, 1.0f };
    D3D12_RECT scissor{ 0, 0, static_cast<LONG>(m_width), static_cast<LONG>(m_height) };
    cmd->RSSetViewports(1, &viewport);
    cmd->RSSetScissorRects(1, &scissor);

    ID3D12DescriptorHeap* heaps[] = { fluid.DescriptorHeap(), fluid.SamplerHeap() };
    cmd->SetDescriptorHeaps(2, heaps);

    cmd->SetGraphicsRootSignature(m_rootSignature.Get());
    cmd->SetPipelineState(m_pipelineState.Get());
    cmd->SetGraphicsRootConstantBufferView(0, m_cameraCB->GetAddress());
    cmd->SetGraphicsRootDescriptorTable(1, fluid.GpuHandle(3));
    cmd->SetGraphicsRootDescriptorTable(2, fluid.GpuHandle(4));
    cmd->SetGraphicsRootDescriptorTable(3, fluid.GpuHandle(7));
    cmd->SetGraphicsRootDescriptorTable(4, fluid.SamplerHeap()->GetGPUDescriptorHandleForHeapStart());

    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd->DrawInstanced(3, 1, 0, 0);

    Transition(cmd, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
}

void SSRRenderer::CreateTarget(ID3D12Device* device, FluidWaterRenderer& fluid, UINT width, UINT height)
{
    CD3DX12_DESCRIPTOR_HEAP_DESC rtvDesc(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1);
    device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(m_rtvHeap.ReleaseAndGetAddressOf()));
    m_ssrRTV = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();

    CD3DX12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_R11G11B10_FLOAT, width, height, 1, 1, 1, 0,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    D3D12_CLEAR_VALUE clear{};
    clear.Format = DXGI_FORMAT_R11G11B10_FLOAT;
    clear.Color[0] = clear.Color[1] = clear.Color[2] = 0.0f;
    clear.Color[3] = 1.0f;
    CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_DEFAULT);
    device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &clear,
        IID_PPV_ARGS(m_ssrTexture.ReleaseAndGetAddressOf()));
    m_ssrState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    device->CreateRenderTargetView(m_ssrTexture.Get(), nullptr, m_ssrRTV);
    fluid.UpdateSSRColorSRV(m_ssrTexture.Get());
}

void SSRRenderer::CreatePipeline(ID3D12Device* device)
{
    if (!m_rootSignature)
    {
        CD3DX12_DESCRIPTOR_RANGE ranges[3];
        ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0); // SceneColor
        ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1); // SceneDepth
        ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2); // Normal

        CD3DX12_DESCRIPTOR_RANGE samplerRange;
        samplerRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0);

        CD3DX12_ROOT_PARAMETER params[5];
        params[0].InitAsConstantBufferView(0);
        params[1].InitAsDescriptorTable(1, &ranges[0]);
        params[2].InitAsDescriptorTable(1, &ranges[1]);
        params[3].InitAsDescriptorTable(1, &ranges[2]);
        params[4].InitAsDescriptorTable(1, &samplerRange);

        D3D12_ROOT_SIGNATURE_DESC desc{};
        desc.NumParameters = _countof(params);
        desc.pParameters = params;
        desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        ComPtr<ID3DBlob> blob;
        ComPtr<ID3DBlob> error;
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, blob.GetAddressOf(), error.GetAddressOf());
        device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(m_rootSignature.ReleaseAndGetAddressOf()));
    }

    ComPtr<ID3DBlob> vs;
    ComPtr<ID3DBlob> ps;
    D3DCompileFromFile(L"FullscreenVS.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "main", "vs_5_1", 0, 0, vs.GetAddressOf(), nullptr);
    D3DCompileFromFile(L"Shaders/SSR/SSR_PS.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "PSMain", "ps_5_1", 0, 0, ps.GetAddressOf(), nullptr);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = m_rootSignature.Get();
    pso.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
    pso.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
    pso.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    pso.SampleMask = UINT_MAX;
    pso.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    pso.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    pso.DepthStencilState.DepthEnable = FALSE;
    pso.NumRenderTargets = 1;
    pso.RTVFormats[0] = DXGI_FORMAT_R11G11B10_FLOAT;
    pso.SampleDesc.Count = 1;
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(m_pipelineState.ReleaseAndGetAddressOf()));
}

void SSRRenderer::UpdateCameraCB(const Camera& camera, UINT ssrQuality)
{
    FluidCameraConstants* cb = m_cameraCB->GetPtr<FluidCameraConstants>();
    XMMATRIX viewMatrix = camera.GetViewMatrix();
    XMMATRIX projMatrix = camera.GetProjMatrix();
    XMStoreFloat4x4(&cb->View, XMMatrixTranspose(viewMatrix));
    XMStoreFloat4x4(&cb->Proj, XMMatrixTranspose(projMatrix));
    cb->ScreenSize = XMFLOAT2(static_cast<float>(m_width), static_cast<float>(m_height));
    cb->InvScreenSize = XMFLOAT2(1.0f / m_width, 1.0f / m_height);
    cb->NearZ = camera.GetNearClip();
    cb->FarZ = camera.GetFarClip();
    cb->IorF0 = XMFLOAT3(0.02f, 0.02f, 0.02f);
    cb->Absorb = 1.2f;
    cb->Options = XMUINT4(0, ssrQuality, 0, 0);
}

void SSRRenderer::Transition(ID3D12GraphicsCommandList* cmd, D3D12_RESOURCE_STATES after)
{
    if (!m_ssrTexture || m_ssrState == after)
    {
        return;
    }
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(m_ssrTexture.Get(), m_ssrState, after);
    cmd->ResourceBarrier(1, &barrier);
    m_ssrState = after;
}
