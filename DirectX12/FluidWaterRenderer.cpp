#include "FluidWaterRenderer.h"
#include "Camera.h"
#include "FluidSystem.h"
#include "Engine.h"
#include <d3dx12.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <algorithm>

#pragma comment(lib, "d3dcompiler.lib")

using namespace DirectX;
using Microsoft::WRL::ComPtr;

namespace
{
    // ================================
    // HLSLコンパイルのユーティリティ
    // ================================
    bool CompileShader(const std::wstring& path, const char* entry, const char* target, ComPtr<ID3DBlob>& blob)
    {
        UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
    #if defined(_DEBUG)
        flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
    #endif
        ComPtr<ID3DBlob> errors;
        HRESULT hr = D3DCompileFromFile(path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
            entry, target, flags, 0, blob.GetAddressOf(), errors.GetAddressOf());
        if (FAILED(hr))
        {
            if (errors)
            {
                OutputDebugStringA((char*)errors->GetBufferPointer());
            }
            return false;
        }
        return true;
    }

    inline D3D12_STATIC_SAMPLER_DESC LinearClampSampler()
    {
        D3D12_STATIC_SAMPLER_DESC desc{};
        desc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        desc.MaxLOD = D3D12_FLOAT32_MAX;
        desc.ShaderRegister = 0;
        desc.RegisterSpace = 0;
        desc.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        return desc;
    }
}

FluidWaterRenderer::FluidWaterRenderer() = default;

bool FluidWaterRenderer::Init(ID3D12Device* device, UINT width, UINT height)
{
    if (!device || m_initialized)
    {
        return false;
    }

    m_device = device;
    m_width = width;
    m_height = height;

    CreateHeaps(device);
    CreateConstantBuffers(device);
    CreateRenderTargets(device);
    CreateRootSignatures(device);
    CreatePipelineStates(device);
    CreateSamplers(device);

    m_initialized = true;
    return true;
}

void FluidWaterRenderer::Resize(UINT width, UINT height)
{
    if (!m_initialized || (width == m_width && height == m_height))
    {
        return;
    }

    m_width = width;
    m_height = height;

    CreateRenderTargets(m_device);
}

void FluidWaterRenderer::SetDownsample(uint32_t step)
{
    step = std::max<uint32_t>(1u, std::min<uint32_t>(4u, step));
    if (m_downsample == step)
    {
        return;
    }
    m_downsample = step;
    CreateRenderTargets(m_device);
}

void FluidWaterRenderer::BeginSceneRender(ID3D12GraphicsCommandList* cmd)
{
    if (!cmd)
    {
        return;
    }

    ID3D12DescriptorHeap* heaps[] = { m_srvUavHeap.Get(), m_samplerHeap.Get() };
    cmd->SetDescriptorHeaps(2, heaps);

    // ダウンサンプル解像度のビューポート／シザーを設定
    UINT targetWidth = std::max<UINT>(1u, m_width / m_downsample);
    UINT targetHeight = std::max<UINT>(1u, m_height / m_downsample);
    D3D12_VIEWPORT viewport{ 0.0f, 0.0f, static_cast<float>(targetWidth), static_cast<float>(targetHeight), 0.0f, 1.0f };
    D3D12_RECT scissor{ 0, 0, static_cast<LONG>(targetWidth), static_cast<LONG>(targetHeight) };
    cmd->RSSetViewports(1, &viewport);
    cmd->RSSetScissorRects(1, &scissor);

    // シーンカラー／深度をレンダーターゲットに遷移
    Transition(cmd, m_sceneColor.Get(), m_sceneColorState, D3D12_RESOURCE_STATE_RENDER_TARGET);
    Transition(cmd, m_sceneDepthBuffer.Get(), m_sceneDepthState, D3D12_RESOURCE_STATE_DEPTH_WRITE);

    // RTV/DSVを設定
    cmd->OMSetRenderTargets(1, &m_sceneColorRTV, FALSE, &m_sceneDepthDSV);

    const float clearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    cmd->ClearRenderTargetView(m_sceneColorRTV, clearColor, 0, nullptr);
    cmd->ClearDepthStencilView(m_sceneDepthDSV, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
}

void FluidWaterRenderer::EndSceneRender(ID3D12GraphicsCommandList* cmd)
{
    if (!cmd)
    {
        return;
    }

    // 描画結果をシェーダ読み込み可能に戻す
    Transition(cmd, m_sceneColor.Get(), m_sceneColorState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    Transition(cmd, m_sceneDepthBuffer.Get(), m_sceneDepthState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    // バックバッファへ戻す
    auto backBuffer = g_Engine->CurrentBackBufferView();
    auto defaultDSV = g_Engine->DepthStencilView();
    cmd->OMSetRenderTargets(1, &backBuffer, FALSE, &defaultDSV);

    // 元の解像度にビューポート／シザーを戻す
    D3D12_VIEWPORT viewport{ 0.0f, 0.0f, static_cast<float>(g_Engine->FrameBufferWidth()), static_cast<float>(g_Engine->FrameBufferHeight()), 0.0f, 1.0f };
    D3D12_RECT scissor{ 0, 0, static_cast<LONG>(g_Engine->FrameBufferWidth()), static_cast<LONG>(g_Engine->FrameBufferHeight()) };
    cmd->RSSetViewports(1, &viewport);
    cmd->RSSetScissorRects(1, &scissor);
}

void FluidWaterRenderer::RenderDepthThickness(ID3D12GraphicsCommandList* cmd, const Camera& camera, FluidSystem& fluid)
{
    if (!m_initialized || !cmd)
    {
        return;
    }

    UpdateCameraConstants(camera);
    UpdateFluidConstants(fluid);

    auto particleSRV = fluid.ActiveMetaSRV();
    if (particleSRV)
    {
        m_device->CopyDescriptorsSimple(1, CpuHandle(2), particleSRV->HandleCPU, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

    // FluidDepth/Thicknessをレンダーターゲットへ遷移
    Transition(cmd, m_fluidDepth.Get(), m_fluidDepthState, D3D12_RESOURCE_STATE_RENDER_TARGET);
    Transition(cmd, m_fluidThickness.Get(), m_fluidThicknessState, D3D12_RESOURCE_STATE_RENDER_TARGET);
    Transition(cmd, m_fluidDepthStencil.Get(), m_fluidDepthStencilState, D3D12_RESOURCE_STATE_DEPTH_WRITE);

    ID3D12DescriptorHeap* heaps[] = { m_srvUavHeap.Get(), m_samplerHeap.Get() };
    cmd->SetDescriptorHeaps(2, heaps);

    cmd->SetGraphicsRootSignature(m_depthThicknessRS.Get());
    cmd->SetPipelineState(m_depthThicknessPSO.Get());

    UINT targetWidth = std::max<UINT>(1u, m_width / m_downsample);
    UINT targetHeight = std::max<UINT>(1u, m_height / m_downsample);
    D3D12_VIEWPORT viewport{ 0.0f, 0.0f, static_cast<float>(targetWidth), static_cast<float>(targetHeight), 0.0f, 1.0f };
    D3D12_RECT scissor{ 0, 0, static_cast<LONG>(targetWidth), static_cast<LONG>(targetHeight) };
    cmd->RSSetViewports(1, &viewport);
    cmd->RSSetScissorRects(1, &scissor);

    D3D12_CPU_DESCRIPTOR_HANDLE rtHandles[2] = { m_fluidThicknessRTV, m_fluidDepthRTV };
    cmd->OMSetRenderTargets(2, rtHandles, FALSE, &m_fluidDSV);

    float clearThick[4] = { 0,0,0,0 };
    cmd->ClearRenderTargetView(m_fluidThicknessRTV, clearThick, 0, nullptr);
    cmd->ClearRenderTargetView(m_fluidDepthRTV, clearThick, 0, nullptr);
    cmd->ClearDepthStencilView(m_fluidDSV, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    cmd->SetGraphicsRootConstantBufferView(0, m_cameraCB->GetAddress());
    cmd->SetGraphicsRootConstantBufferView(1, m_fluidCB->GetAddress());
    cmd->SetGraphicsRootDescriptorTable(2, GpuHandle(2));
    cmd->SetGraphicsRootDescriptorTable(3, m_samplerHeap->GetGPUDescriptorHandleForHeapStart());

    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd->DrawInstanced(3, 1, 0, 0);

    // 描画後はSRV状態に戻す
    Transition(cmd, m_fluidThickness.Get(), m_fluidThicknessState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    Transition(cmd, m_fluidDepth.Get(), m_fluidDepthState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
}

void FluidWaterRenderer::BlurAndNormal(ID3D12GraphicsCommandList* cmd)
{
    if (!m_initialized || !cmd)
    {
        return;
    }

    ID3D12DescriptorHeap* heaps[] = { m_srvUavHeap.Get(), m_samplerHeap.Get() };
    cmd->SetDescriptorHeaps(2, heaps);

    // Blur X : Depth -> DepthBlur
    D3D12_UNORDERED_ACCESS_VIEW_DESC blurUav{};
    blurUav.Format = DXGI_FORMAT_R32_FLOAT;
    blurUav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(m_fluidDepthBlur.Get(), nullptr, &blurUav, CpuHandle(11));
    Transition(cmd, m_fluidDepth.Get(), m_fluidDepthState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Transition(cmd, m_fluidDepthBlur.Get(), m_fluidDepthBlurState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    cmd->SetComputeRootSignature(m_blurRS.Get());
    cmd->SetPipelineState(m_blurXPSO.Get());
    cmd->SetComputeRootDescriptorTable(0, GpuHandle(5));
    cmd->SetComputeRootDescriptorTable(1, m_samplerHeap->GetGPUDescriptorHandleForHeapStart());
    cmd->SetComputeRootDescriptorTable(2, GpuHandle(11));

    UINT dispatchX = (m_width / m_downsample + 7) / 8;
    UINT dispatchY = (m_height / m_downsample + 7) / 8;
    cmd->Dispatch(dispatchX, dispatchY, 1);

    auto uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_fluidDepthBlur.Get());
    cmd->ResourceBarrier(1, &uavBarrier);

    Transition(cmd, m_fluidDepthBlur.Get(), m_fluidDepthBlurState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Transition(cmd, m_fluidDepth.Get(), m_fluidDepthState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Blur Y : DepthBlur -> Depth
    D3D12_SHADER_RESOURCE_VIEW_DESC blurSrv{};
    blurSrv.Format = DXGI_FORMAT_R32_FLOAT;
    blurSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    blurSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    blurSrv.Texture2D.MipLevels = 1;
    m_device->CreateShaderResourceView(m_fluidDepthBlur.Get(), &blurSrv, CpuHandle(11));

    cmd->SetPipelineState(m_blurYPSO.Get());
    cmd->SetComputeRootDescriptorTable(0, GpuHandle(11));
    cmd->SetComputeRootDescriptorTable(1, m_samplerHeap->GetGPUDescriptorHandleForHeapStart());
    cmd->SetComputeRootDescriptorTable(2, GpuHandle(9));
    cmd->Dispatch(dispatchX, dispatchY, 1);

    uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_fluidDepth.Get());
    cmd->ResourceBarrier(1, &uavBarrier);

    Transition(cmd, m_fluidDepth.Get(), m_fluidDepthState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // Normal reconstruct
    Transition(cmd, m_fluidNormal.Get(), m_fluidNormalState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    cmd->SetComputeRootSignature(m_normalRS.Get());
    cmd->SetPipelineState(m_normalPSO.Get());
    cmd->SetComputeRootDescriptorTable(0, GpuHandle(5));
    cmd->SetComputeRootDescriptorTable(1, GpuHandle(12));
    cmd->Dispatch(dispatchX, dispatchY, 1);

    uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_fluidNormal.Get());
    cmd->ResourceBarrier(1, &uavBarrier);

    Transition(cmd, m_fluidNormal.Get(), m_fluidNormalState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    Transition(cmd, m_fluidDepth.Get(), m_fluidDepthState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
}

void FluidWaterRenderer::Composite(ID3D12GraphicsCommandList* cmd)
{
    if (!m_initialized || !cmd)
    {
        return;
    }

    ID3D12DescriptorHeap* heaps[] = { m_srvUavHeap.Get(), m_samplerHeap.Get() };
    cmd->SetDescriptorHeaps(2, heaps);

    // シーンカラーテクスチャと深度をピクセルシェーダで読む準備
    Transition(cmd, m_sceneColor.Get(), m_sceneColorState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    Transition(cmd, m_sceneDepthBuffer.Get(), m_sceneDepthState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    Transition(cmd, m_fluidDepth.Get(), m_fluidDepthState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    Transition(cmd, m_fluidThickness.Get(), m_fluidThicknessState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    Transition(cmd, m_fluidNormal.Get(), m_fluidNormalState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    auto backBuffer = g_Engine->CurrentBackBufferView();
    auto defaultDSV = g_Engine->DepthStencilView();
    cmd->OMSetRenderTargets(1, &backBuffer, FALSE, &defaultDSV);

    D3D12_VIEWPORT viewport{ 0.0f, 0.0f, static_cast<float>(g_Engine->FrameBufferWidth()), static_cast<float>(g_Engine->FrameBufferHeight()), 0.0f, 1.0f };
    D3D12_RECT scissor{ 0, 0, static_cast<LONG>(g_Engine->FrameBufferWidth()), static_cast<LONG>(g_Engine->FrameBufferHeight()) };
    cmd->RSSetViewports(1, &viewport);
    cmd->RSSetScissorRects(1, &scissor);

    cmd->SetGraphicsRootSignature(m_compositeRS.Get());
    cmd->SetPipelineState(m_compositePSO.Get());
    cmd->SetGraphicsRootConstantBufferView(0, m_cameraCB->GetAddress());
    cmd->SetGraphicsRootDescriptorTable(1, GpuHandle(3));
    cmd->SetGraphicsRootDescriptorTable(2, GpuHandle(4));
    cmd->SetGraphicsRootDescriptorTable(3, GpuHandle(5));
    cmd->SetGraphicsRootDescriptorTable(4, GpuHandle(6));
    cmd->SetGraphicsRootDescriptorTable(5, GpuHandle(7));
    cmd->SetGraphicsRootDescriptorTable(6, GpuHandle(8));
    cmd->SetGraphicsRootDescriptorTable(7, m_samplerHeap->GetGPUDescriptorHandleForHeapStart());

    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd->DrawInstanced(3, 1, 0, 0);
}

D3D12_GPU_DESCRIPTOR_HANDLE FluidWaterRenderer::GpuHandle(UINT index) const
{
    D3D12_GPU_DESCRIPTOR_HANDLE handle = m_srvUavGpuStart;
    handle.ptr += static_cast<UINT64>(index) * m_srvUavIncrement;
    return handle;
}

D3D12_CPU_DESCRIPTOR_HANDLE FluidWaterRenderer::CpuHandle(UINT index) const
{
    D3D12_CPU_DESCRIPTOR_HANDLE handle = m_srvUavCpuStart;
    handle.ptr += static_cast<UINT64>(index) * m_srvUavIncrement;
    return handle;
}

void FluidWaterRenderer::UpdateSSRColorSRV(ID3D12Resource* resource)
{
    if (!resource)
    {
        return;
    }

    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = DXGI_FORMAT_R11G11B10_FLOAT;
    desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.Texture2D.MipLevels = 1;
    m_device->CreateShaderResourceView(resource, &desc, CpuHandle(8));
}

void FluidWaterRenderer::CreateHeaps(ID3D12Device* device)
{
    // CBV/SRV/UAVヒープ（32スロット）
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
    heapDesc.NumDescriptors = 32;
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(m_srvUavHeap.ReleaseAndGetAddressOf()));
    m_srvUavIncrement = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_srvUavCpuStart = m_srvUavHeap->GetCPUDescriptorHandleForHeapStart();
    m_srvUavGpuStart = m_srvUavHeap->GetGPUDescriptorHandleForHeapStart();

    // サンプラーヒープ（1スロット）
    D3D12_DESCRIPTOR_HEAP_DESC samplerDesc{};
    samplerDesc.NumDescriptors = 1;
    samplerDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    samplerDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    device->CreateDescriptorHeap(&samplerDesc, IID_PPV_ARGS(m_samplerHeap.ReleaseAndGetAddressOf()));

    // RTVヒープ（シーンカラー + 厚み + 深度）
    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc{};
    rtvDesc.NumDescriptors = 3;
    rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(m_rtvHeap.ReleaseAndGetAddressOf()));
    m_rtvIncrement = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    // DSVヒープ（シーン深度 / FluidDS）
    D3D12_DESCRIPTOR_HEAP_DESC dsvDesc{};
    dsvDesc.NumDescriptors = 2;
    dsvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    device->CreateDescriptorHeap(&dsvDesc, IID_PPV_ARGS(m_dsvHeap.ReleaseAndGetAddressOf()));
    m_dsvIncrement = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);

    m_sceneColorRTV = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    m_fluidThicknessRTV = { m_sceneColorRTV.ptr + m_rtvIncrement };
    m_fluidDepthRTV = { m_fluidThicknessRTV.ptr + m_rtvIncrement };

    m_sceneDepthDSV = m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
    m_fluidDSV = { m_sceneDepthDSV.ptr + m_dsvIncrement };
}

void FluidWaterRenderer::CreateConstantBuffers(ID3D12Device* device)
{
    m_cameraCB = std::make_unique<ConstantBuffer>(sizeof(FluidCameraConstants));
    m_fluidCB = std::make_unique<ConstantBuffer>(sizeof(FluidParamConstants));

    auto cameraView = m_cameraCB->ViewDesc();
    auto fluidView = m_fluidCB->ViewDesc();
    device->CreateConstantBufferView(&cameraView, CpuHandle(0));
    device->CreateConstantBufferView(&fluidView, CpuHandle(1));
}

void FluidWaterRenderer::CreateRenderTargets(ID3D12Device* device)
{
    UINT width = std::max<UINT>(1u, m_width / m_downsample);
    UINT height = std::max<UINT>(1u, m_height / m_downsample);

    CD3DX12_HEAP_PROPERTIES defaultHeap(D3D12_HEAP_TYPE_DEFAULT);

    // シーンカラー
    CD3DX12_RESOURCE_DESC sceneColorDesc = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_R16G16B16A16_FLOAT, width, height, 1, 1, 1, 0,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    D3D12_CLEAR_VALUE sceneClear{};
    sceneClear.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    sceneClear.Color[0] = 0;
    sceneClear.Color[1] = 0;
    sceneClear.Color[2] = 0;
    sceneClear.Color[3] = 1;
    device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &sceneColorDesc,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &sceneClear,
        IID_PPV_ARGS(m_sceneColor.ReleaseAndGetAddressOf()));
    m_sceneColorState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    device->CreateRenderTargetView(m_sceneColor.Get(), nullptr, m_sceneColorRTV);

    D3D12_SHADER_RESOURCE_VIEW_DESC sceneColorSrv{};
    sceneColorSrv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    sceneColorSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sceneColorSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sceneColorSrv.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(m_sceneColor.Get(), &sceneColorSrv, CpuHandle(3));

    // シーン深度（TypelessでDSV/SRV併用）
    CD3DX12_RESOURCE_DESC depthDesc = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_R32_TYPELESS, width, height, 1, 1, 1, 0,
        D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    D3D12_CLEAR_VALUE depthClear{};
    depthClear.Format = DXGI_FORMAT_D32_FLOAT;
    depthClear.DepthStencil.Depth = 1.0f;
    depthClear.DepthStencil.Stencil = 0;
    device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &depthDesc,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &depthClear,
        IID_PPV_ARGS(m_sceneDepthBuffer.ReleaseAndGetAddressOf()));
    m_sceneDepthState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
    dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    device->CreateDepthStencilView(m_sceneDepthBuffer.Get(), &dsvDesc, m_sceneDepthDSV);

    D3D12_SHADER_RESOURCE_VIEW_DESC depthSrv{};
    depthSrv.Format = DXGI_FORMAT_R32_FLOAT;
    depthSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    depthSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    depthSrv.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(m_sceneDepthBuffer.Get(), &depthSrv, CpuHandle(4));

    // Fluid厚み（R16_FLOAT）
    CD3DX12_RESOURCE_DESC thickDesc = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_R16_TYPELESS, width, height, 1, 1, 1, 0,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    D3D12_CLEAR_VALUE thickClear{};
    thickClear.Format = DXGI_FORMAT_R16_FLOAT;
    thickClear.Color[0] = 0;
    device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &thickDesc,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &thickClear,
        IID_PPV_ARGS(m_fluidThickness.ReleaseAndGetAddressOf()));
    m_fluidThicknessState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    D3D12_RENDER_TARGET_VIEW_DESC thickRTV{};
    thickRTV.Format = DXGI_FORMAT_R16_FLOAT;
    thickRTV.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    device->CreateRenderTargetView(m_fluidThickness.Get(), &thickRTV, m_fluidThicknessRTV);

    D3D12_SHADER_RESOURCE_VIEW_DESC thickSrv{};
    thickSrv.Format = DXGI_FORMAT_R16_FLOAT;
    thickSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    thickSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    thickSrv.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(m_fluidThickness.Get(), &thickSrv, CpuHandle(6));

    D3D12_UNORDERED_ACCESS_VIEW_DESC thickUav{};
    thickUav.Format = DXGI_FORMAT_R16_FLOAT;
    thickUav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(m_fluidThickness.Get(), nullptr, &thickUav, CpuHandle(10));

    // Fluid深度（R32_FLOAT）
    CD3DX12_RESOURCE_DESC depthLinearDesc = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_R32_TYPELESS, width, height, 1, 1, 1, 0,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    D3D12_CLEAR_VALUE depthLinearClear{};
    depthLinearClear.Format = DXGI_FORMAT_R32_FLOAT;
    depthLinearClear.Color[0] = 1.0f;
    device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &depthLinearDesc,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &depthLinearClear,
        IID_PPV_ARGS(m_fluidDepth.ReleaseAndGetAddressOf()));
    m_fluidDepthState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    D3D12_RENDER_TARGET_VIEW_DESC depthLinearRTV{};
    depthLinearRTV.Format = DXGI_FORMAT_R32_FLOAT;
    depthLinearRTV.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    device->CreateRenderTargetView(m_fluidDepth.Get(), &depthLinearRTV, m_fluidDepthRTV);

    D3D12_SHADER_RESOURCE_VIEW_DESC depthLinearSrv{};
    depthLinearSrv.Format = DXGI_FORMAT_R32_FLOAT;
    depthLinearSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    depthLinearSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    depthLinearSrv.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(m_fluidDepth.Get(), &depthLinearSrv, CpuHandle(5));

    D3D12_UNORDERED_ACCESS_VIEW_DESC depthLinearUav{};
    depthLinearUav.Format = DXGI_FORMAT_R32_FLOAT;
    depthLinearUav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(m_fluidDepth.Get(), nullptr, &depthLinearUav, CpuHandle(9));

    // Blur用の中間バッファ
    device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &depthLinearDesc,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &depthLinearClear,
        IID_PPV_ARGS(m_fluidDepthBlur.ReleaseAndGetAddressOf()));
    m_fluidDepthBlurState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    // Fluid法線（R11G11B10_FLOAT）
    CD3DX12_RESOURCE_DESC normalDesc = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_R11G11B10_FLOAT, width, height, 1, 1, 1, 0,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &normalDesc,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr,
        IID_PPV_ARGS(m_fluidNormal.ReleaseAndGetAddressOf()));
    m_fluidNormalState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    D3D12_SHADER_RESOURCE_VIEW_DESC normalSrv{};
    normalSrv.Format = DXGI_FORMAT_R11G11B10_FLOAT;
    normalSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    normalSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    normalSrv.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(m_fluidNormal.Get(), &normalSrv, CpuHandle(7));

    D3D12_UNORDERED_ACCESS_VIEW_DESC normalUav{};
    normalUav.Format = DXGI_FORMAT_R11G11B10_FLOAT;
    normalUav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(m_fluidNormal.Get(), nullptr, &normalUav, CpuHandle(12));

    // Fluid用DSV
    D3D12_RESOURCE_DESC fluidDSDesc = depthDesc;
    fluidDSDesc.Format = DXGI_FORMAT_R24G8_TYPELESS;
    D3D12_CLEAR_VALUE fluidClear{};
    fluidClear.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    fluidClear.DepthStencil.Depth = 1.0f;
    fluidClear.DepthStencil.Stencil = 0;
    device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &fluidDSDesc,
        D3D12_RESOURCE_STATE_DEPTH_WRITE, &fluidClear,
        IID_PPV_ARGS(m_fluidDepthStencil.ReleaseAndGetAddressOf()));
    m_fluidDepthStencilState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    D3D12_DEPTH_STENCIL_VIEW_DESC fluidDSVDesc{};
    fluidDSVDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    fluidDSVDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    device->CreateDepthStencilView(m_fluidDepthStencil.Get(), &fluidDSVDesc, m_fluidDSV);
}

void FluidWaterRenderer::CreateRootSignatures(ID3D12Device* device)
{
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> error;

    // DepthThickness RS
    CD3DX12_DESCRIPTOR_RANGE depthSrvRange;
    depthSrvRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
    CD3DX12_DESCRIPTOR_RANGE samplerRange;
    samplerRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0);

    CD3DX12_ROOT_PARAMETER depthParams[4];
    depthParams[0].InitAsConstantBufferView(0);
    depthParams[1].InitAsConstantBufferView(1);
    depthParams[2].InitAsDescriptorTable(1, &depthSrvRange);
    depthParams[3].InitAsDescriptorTable(1, &samplerRange);

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = _countof(depthParams);
    rsDesc.pParameters = depthParams;
    rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, blob.GetAddressOf(), error.GetAddressOf());
    device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(m_depthThicknessRS.ReleaseAndGetAddressOf()));

    // Blur RS
    CD3DX12_DESCRIPTOR_RANGE blurSrvRange;
    blurSrvRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
    CD3DX12_DESCRIPTOR_RANGE blurUavRange;
    blurUavRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);

    CD3DX12_ROOT_PARAMETER blurParams[3];
    blurParams[0].InitAsDescriptorTable(1, &blurSrvRange);
    blurParams[1].InitAsDescriptorTable(1, &samplerRange);
    blurParams[2].InitAsDescriptorTable(1, &blurUavRange);

    rsDesc = {};
    rsDesc.NumParameters = _countof(blurParams);
    rsDesc.pParameters = blurParams;
    rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, blob.ReleaseAndGetAddressOf(), error.ReleaseAndGetAddressOf());
    device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(m_blurRS.ReleaseAndGetAddressOf()));

    // Normal RS
    CD3DX12_DESCRIPTOR_RANGE normalSrvRange;
    normalSrvRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
    CD3DX12_DESCRIPTOR_RANGE normalUavRange;
    normalUavRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);

    CD3DX12_ROOT_PARAMETER normalParams[2];
    normalParams[0].InitAsDescriptorTable(1, &normalSrvRange);
    normalParams[1].InitAsDescriptorTable(1, &normalUavRange);

    rsDesc = {};
    rsDesc.NumParameters = _countof(normalParams);
    rsDesc.pParameters = normalParams;
    rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, blob.ReleaseAndGetAddressOf(), error.ReleaseAndGetAddressOf());
    device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(m_normalRS.ReleaseAndGetAddressOf()));

    // Composite RS
    CD3DX12_DESCRIPTOR_RANGE compRange[6];
    for (UINT i = 0; i < 6; ++i)
    {
        compRange[i].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, i);
    }

    CD3DX12_ROOT_PARAMETER compParams[8];
    compParams[0].InitAsConstantBufferView(0);
    for (int i = 0; i < 6; ++i)
    {
        compParams[1 + i].InitAsDescriptorTable(1, &compRange[i]);
    }
    compParams[7].InitAsDescriptorTable(1, &samplerRange);

    rsDesc = {};
    rsDesc.NumParameters = _countof(compParams);
    rsDesc.pParameters = compParams;
    rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, blob.ReleaseAndGetAddressOf(), error.ReleaseAndGetAddressOf());
    device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(m_compositeRS.ReleaseAndGetAddressOf()));
}

void FluidWaterRenderer::CreatePipelineStates(ID3D12Device* device)
{
    ComPtr<ID3DBlob> vs;
    ComPtr<ID3DBlob> ps;
    ComPtr<ID3DBlob> cs;

    // シェーダーファイルはプロジェクト直下に配置しているため、ファイル名のみで指定する
    // DepthThickness PSO
    CompileShader(L"FluidDepthThicknessVS.hlsl", "VSMain", "vs_5_1", vs);
    CompileShader(L"FluidDepthThicknessPS.hlsl", "PSMain", "ps_5_1", ps);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = m_depthThicknessRS.Get();
    desc.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
    desc.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
    desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    desc.BlendState.RenderTarget[0].BlendEnable = TRUE;
    desc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_ONE;
    desc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_ONE;
    desc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
    desc.SampleMask = UINT_MAX;
    desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    desc.DepthStencilState.DepthEnable = TRUE;
    desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    desc.NumRenderTargets = 2;
    desc.RTVFormats[0] = DXGI_FORMAT_R16_FLOAT;
    desc.RTVFormats[1] = DXGI_FORMAT_R32_FLOAT;
    desc.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
    desc.SampleDesc.Count = 1;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(m_depthThicknessPSO.ReleaseAndGetAddressOf()));

    // Blur X
    CompileShader(L"BilateralBlurX.hlsl", "CSMain", "cs_5_1", cs);
    D3D12_COMPUTE_PIPELINE_STATE_DESC cdesc{};
    cdesc.pRootSignature = m_blurRS.Get();
    cdesc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    device->CreateComputePipelineState(&cdesc, IID_PPV_ARGS(m_blurXPSO.ReleaseAndGetAddressOf()));

    // Blur Y
    CompileShader(L"BilateralBlurY.hlsl", "CSMain", "cs_5_1", cs);
    cdesc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    device->CreateComputePipelineState(&cdesc, IID_PPV_ARGS(m_blurYPSO.ReleaseAndGetAddressOf()));

    // Normal reconstruct
    CompileShader(L"ReconstructNormalCS.hlsl", "CSMain", "cs_5_1", cs);
    cdesc.pRootSignature = m_normalRS.Get();
    cdesc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    device->CreateComputePipelineState(&cdesc, IID_PPV_ARGS(m_normalPSO.ReleaseAndGetAddressOf()));

    // Composite
    CompileShader(L"FullscreenVS.hlsl", "main", "vs_5_1", vs);
    CompileShader(L"CompositeWaterPS.hlsl", "PSMain", "ps_5_1", ps);
    desc = {};
    desc.pRootSignature = m_compositeRS.Get();
    desc.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
    desc.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
    desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    desc.SampleMask = UINT_MAX;
    desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    desc.DepthStencilState.DepthEnable = FALSE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(m_compositePSO.ReleaseAndGetAddressOf()));
}

void FluidWaterRenderer::CreateSamplers(ID3D12Device* device)
{
    D3D12_SAMPLER_DESC desc{};
    desc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    desc.MaxLOD = D3D12_FLOAT32_MAX;
    device->CreateSampler(&desc, m_samplerHeap->GetCPUDescriptorHandleForHeapStart());
}

void FluidWaterRenderer::UpdateCameraConstants(const Camera& camera)
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
    cb->Options = XMUINT4(static_cast<UINT>(m_debugView), m_ssrQuality, m_usePlanarReflection ? 1u : 0u, m_showGrid ? 1u : 0u);
}

void FluidWaterRenderer::UpdateFluidConstants(const FluidSystem& fluid)
{
    FluidParamConstants* cb = m_fluidCB->GetPtr<FluidParamConstants>();
    cb->Radius = fluid.RenderParticleRadius();
    cb->ThicknessScale = 1.0f;
    cb->ParticleCount = fluid.ActiveParticleCount();
    cb->Downsample = static_cast<float>(m_downsample);
}

void FluidWaterRenderer::Transition(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource, D3D12_RESOURCE_STATES& state, D3D12_RESOURCE_STATES after)
{
    if (!resource || state == after)
    {
        return;
    }
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(resource, state, after);
    cmd->ResourceBarrier(1, &barrier);
    state = after;
}
