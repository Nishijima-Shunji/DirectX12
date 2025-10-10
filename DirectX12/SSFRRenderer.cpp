#include "SSFRRenderer.h"
#include <d3dx12.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <windows.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cwchar>

using Microsoft::WRL::ComPtr;
using namespace DirectX;

namespace {
    // 変更理由: 各CBの内容を明示し、設定値変更時の転送を確実にするためのローカル構造体。
    struct CameraCBData {
        XMFLOAT4X4 view;
        XMFLOAT4X4 proj;
        XMFLOAT2 screenSize;
        XMFLOAT2 invScreenSize;
        float nearZ;
        float farZ;
        XMFLOAT2 pad;
    };

    struct DrawCBData {
        float particleRadius;
        float pad[3];
    };

    struct BlurCBData {
        float worldBlurRadius;
        float depthSigma;
        uint32_t enableBilateral;
        float pad;
    };

    struct CompositeCBData {
        float refractScale;
        XMFLOAT3 absorbK;
        XMFLOAT3 fluidColor;
        uint32_t flags;
    };

    ComPtr<ID3DBlob> CompileShader(const wchar_t* path, const char* entry, const char* target)
    {
        UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(_DEBUG)
        flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
        ComPtr<ID3DBlob> shader;
        ComPtr<ID3DBlob> error;
        HRESULT hr = D3DCompileFromFile(path, nullptr, nullptr, entry, target, flags, 0, shader.GetAddressOf(), error.GetAddressOf());
        if (FAILED(hr) && error)
        {
            OutputDebugStringA((char*)error->GetBufferPointer());
        }
        return shader;
    }
}

void SSFRRenderer::Initialize(ID3D12Device* device, DXGI_FORMAT swapChainFormat)
{
    mDevice = device;
    mSwapFormat = swapChainFormat;
    mResources.Initialize(device, swapChainFormat);

    AllocateCB(device, mCameraCB, sizeof(CameraCBData), "SSFR_CameraCB");
    AllocateCB(device, mDrawCB, sizeof(DrawCBData), "SSFR_DrawCB");
    AllocateCB(device, mBlurCB, sizeof(BlurCBData), "SSFR_BlurCB");
    AllocateCB(device, mCompositeCB, sizeof(CompositeCBData), "SSFR_CompositeCB");

    CreatePipelines(device);
    UpdateDrawCB();
    UpdateBlurCB(mDepthSigma);
    UpdateCompositeCB();
}

void SSFRRenderer::Resize(ID3D12Device* device, uint32_t width, uint32_t height)
{
    mWidth = width;
    mHeight = height;
    mResources.Resize(device, width, height, mSettings.thicknessHalfRes);
}

void SSFRRenderer::SetEnvironmentCube(ID3D12Resource* cubeSRV)
{
    mEnvironmentCube = cubeSRV;
}

void SSFRRenderer::SetSettings(const SSFRSettings& s)
{
    mSettings = s;
    UpdateDrawCB();
    UpdateBlurCB(mDepthSigma);
    UpdateCompositeCB();
    if (mDevice)
    {
        mResources.Resize(mDevice.Get(), mWidth, mHeight, mSettings.thicknessHalfRes);
    }
}

void SSFRRenderer::Render(
    ID3D12GraphicsCommandList* cmd,
    const SSFRCamera& cam,
    const SSFRParticleInputSRV& fluid,
    ID3D12DescriptorHeap* /*srvUavHeap*/,
    D3D12_CPU_DESCRIPTOR_HANDLE rtvBackbuffer,
    D3D12_CPU_DESCRIPTOR_HANDLE /*dsvDepth*/,
    ID3D12Resource* sceneColorSRV,
    ID3D12Resource* sceneLinearDepthSRV)
{
    // 画面解像度変化や外部リソース更新に追従しつつ、粒子→深度→厚み→合成の順で描画する。
    if (mWidth != cam.width || mHeight != cam.height)
    {
        Resize(mDevice.Get(), cam.width, cam.height);
    }

    PrepareSceneDescriptors(mDevice.Get(), sceneColorSRV, sceneLinearDepthSRV);

    UpdateCameraCB(cam);

    mDevice->CopyDescriptorsSimple(1,
        mResources.GetSrvCPU(SSFRResources::SrvParticlePositions),
        fluid.positionsSrvCPU,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    ID3D12DescriptorHeap* heaps[] = { mResources.GetSrvUavHeap() };
    cmd->SetDescriptorHeaps(1, heaps);

    RenderDepthPass(cmd, fluid);
    BlurDepth(cmd, cam.width, cam.height);

    SSFRCamera thicknessCam = cam;
    uint32_t thicknessW = mSettings.thicknessHalfRes ? (cam.width / 2) : cam.width;
    uint32_t thicknessH = mSettings.thicknessHalfRes ? (cam.height / 2) : cam.height;
    thicknessCam.width = thicknessW;
    thicknessCam.height = thicknessH;
    UpdateCameraCB(thicknessCam);

    RenderThickness(cmd, fluid);
    BlurThickness(cmd, thicknessW, thicknessH);

    UpdateCameraCB(cam);

    Composite(cmd, rtvBackbuffer);
}

void SSFRRenderer::CreatePipelines(ID3D12Device* device)
{
    D3D12_STATIC_SAMPLER_DESC sampler = {};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    // 深度パスルートシグネチャ
    {
        CD3DX12_ROOT_PARAMETER1 params[3];
        params[0].InitAsConstantBufferView(0);
        params[1].InitAsConstantBufferView(1);
        CD3DX12_DESCRIPTOR_RANGE1 range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 8);
        params[2].InitAsDescriptorTable(1, &range);

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc = {};
        desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        desc.Desc_1_1.NumParameters = _countof(params);
        desc.Desc_1_1.pParameters = params;
        desc.Desc_1_1.NumStaticSamplers = 1;
        desc.Desc_1_1.pStaticSamplers = &sampler;
        desc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        ComPtr<ID3DBlob> blob;
        ComPtr<ID3DBlob> error;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&desc, blob.GetAddressOf(), error.GetAddressOf());
        if (FAILED(hr) && error)
        {
            OutputDebugStringA((char*)error->GetBufferPointer());
        }
        device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(mRSDepth.ReleaseAndGetAddressOf()));

        ComPtr<ID3DBlob> vs = CompileShader(L"ParticlesDepthVS.hlsl", "VSMain", "vs_5_1");
        ComPtr<ID3DBlob> ps = CompileShader(L"ParticlesDepthPS.hlsl", "PSMain", "ps_5_1");

        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = mRSDepth.Get();
        psoDesc.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
        psoDesc.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R32_FLOAT;
        psoDesc.SampleDesc.Count = 1;
        device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(mPSODepth.ReleaseAndGetAddressOf()));
    }

    // ブラーパス
    {
        CD3DX12_ROOT_PARAMETER1 params[3];
        params[0].InitAsConstantBufferView(2);
        CD3DX12_DESCRIPTOR_RANGE1 srvRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
        params[1].InitAsDescriptorTable(1, &srvRange);
        CD3DX12_DESCRIPTOR_RANGE1 uavRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
        params[2].InitAsDescriptorTable(1, &uavRange);

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc = {};
        desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        desc.Desc_1_1.NumParameters = _countof(params);
        desc.Desc_1_1.pParameters = params;
        desc.Desc_1_1.NumStaticSamplers = 1;
        desc.Desc_1_1.pStaticSamplers = &sampler;

        ComPtr<ID3DBlob> blob;
        ComPtr<ID3DBlob> error;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&desc, blob.GetAddressOf(), error.GetAddressOf());
        if (FAILED(hr) && error)
        {
            OutputDebugStringA((char*)error->GetBufferPointer());
        }
        device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(mRSBlur.ReleaseAndGetAddressOf()));

        ComPtr<ID3DBlob> csX = CompileShader(L"BilateralBlurCS_X.hlsl", "CSMain", "cs_5_1");
        D3D12_COMPUTE_PIPELINE_STATE_DESC descX = {};
        descX.pRootSignature = mRSBlur.Get();
        descX.CS = { csX->GetBufferPointer(), csX->GetBufferSize() };
        device->CreateComputePipelineState(&descX, IID_PPV_ARGS(mPSOBlurX.ReleaseAndGetAddressOf()));

        ComPtr<ID3DBlob> csY = CompileShader(L"BilateralBlurCS_Y.hlsl", "CSMain", "cs_5_1");
        D3D12_COMPUTE_PIPELINE_STATE_DESC descY = {};
        descY.pRootSignature = mRSBlur.Get();
        descY.CS = { csY->GetBufferPointer(), csY->GetBufferSize() };
        device->CreateComputePipelineState(&descY, IID_PPV_ARGS(mPSOBlurY.ReleaseAndGetAddressOf()));
    }

    // 厚み生成
    {
        CD3DX12_ROOT_PARAMETER1 params[3];
        params[0].InitAsConstantBufferView(0);
        params[1].InitAsConstantBufferView(1);
        CD3DX12_DESCRIPTOR_RANGE1 range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 8);
        params[2].InitAsDescriptorTable(1, &range);

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc = {};
        desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        desc.Desc_1_1.NumParameters = _countof(params);
        desc.Desc_1_1.pParameters = params;
        desc.Desc_1_1.NumStaticSamplers = 1;
        desc.Desc_1_1.pStaticSamplers = &sampler;
        desc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        ComPtr<ID3DBlob> blob;
        ComPtr<ID3DBlob> error;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&desc, blob.GetAddressOf(), error.GetAddressOf());
        if (FAILED(hr) && error)
        {
            OutputDebugStringA((char*)error->GetBufferPointer());
        }
        device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(mRSThickness.ReleaseAndGetAddressOf()));

        ComPtr<ID3DBlob> vs = CompileShader(L"ThicknessVS.hlsl", "VSMain", "vs_5_1");
        ComPtr<ID3DBlob> ps = CompileShader(L"ThicknessPS.hlsl", "PSMain", "ps_5_1");

        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = mRSThickness.Get();
        psoDesc.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
        psoDesc.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        psoDesc.BlendState.RenderTarget[0].BlendEnable = TRUE;
        psoDesc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_ONE;
        psoDesc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_ONE;
        psoDesc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
        psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R16_FLOAT;
        psoDesc.SampleDesc.Count = 1;
        device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(mPSOThickness.ReleaseAndGetAddressOf()));
    }

    // 厚みブラー
    {
        CD3DX12_ROOT_PARAMETER1 params[3];
        params[0].InitAsConstantBufferView(2);
        CD3DX12_DESCRIPTOR_RANGE1 srvRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
        params[1].InitAsDescriptorTable(1, &srvRange);
        CD3DX12_DESCRIPTOR_RANGE1 uavRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
        params[2].InitAsDescriptorTable(1, &uavRange);

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc = {};
        desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        desc.Desc_1_1.NumParameters = _countof(params);
        desc.Desc_1_1.pParameters = params;
        desc.Desc_1_1.NumStaticSamplers = 1;
        desc.Desc_1_1.pStaticSamplers = &sampler;

        ComPtr<ID3DBlob> blob;
        ComPtr<ID3DBlob> error;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&desc, blob.GetAddressOf(), error.GetAddressOf());
        if (FAILED(hr) && error)
        {
            OutputDebugStringA((char*)error->GetBufferPointer());
        }
        device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(mRSThicknessBlur.ReleaseAndGetAddressOf()));

        ComPtr<ID3DBlob> csX = CompileShader(L"ThicknessPS.hlsl", "ThicknessBlurCS_X", "cs_5_1");
        D3D12_COMPUTE_PIPELINE_STATE_DESC descX = {};
        descX.pRootSignature = mRSThicknessBlur.Get();
        descX.CS = { csX->GetBufferPointer(), csX->GetBufferSize() };
        device->CreateComputePipelineState(&descX, IID_PPV_ARGS(mPSOThicknessBlurX.ReleaseAndGetAddressOf()));

        ComPtr<ID3DBlob> csY = CompileShader(L"ThicknessPS.hlsl", "ThicknessBlurCS_Y", "cs_5_1");
        D3D12_COMPUTE_PIPELINE_STATE_DESC descY = {};
        descY.pRootSignature = mRSThicknessBlur.Get();
        descY.CS = { csY->GetBufferPointer(), csY->GetBufferSize() };
        device->CreateComputePipelineState(&descY, IID_PPV_ARGS(mPSOThicknessBlurY.ReleaseAndGetAddressOf()));
    }

    // 合成
    {
        CD3DX12_ROOT_PARAMETER1 params[4];
        params[0].InitAsConstantBufferView(0);
        params[1].InitAsConstantBufferView(3);
        CD3DX12_DESCRIPTOR_RANGE1 rangeDepth(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
        params[2].InitAsDescriptorTable(1, &rangeDepth);
        CD3DX12_DESCRIPTOR_RANGE1 rangeOthers[2];
        rangeOthers[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
        rangeOthers[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 2, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND);
        params[3].InitAsDescriptorTable(2, rangeOthers);

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc = {};
        desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        desc.Desc_1_1.NumParameters = _countof(params);
        desc.Desc_1_1.pParameters = params;
        desc.Desc_1_1.NumStaticSamplers = 1;
        desc.Desc_1_1.pStaticSamplers = &sampler;

        ComPtr<ID3DBlob> blob;
        ComPtr<ID3DBlob> error;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&desc, blob.GetAddressOf(), error.GetAddressOf());
        if (FAILED(hr) && error)
        {
            OutputDebugStringA((char*)error->GetBufferPointer());
        }
        device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(mRSComposite.ReleaseAndGetAddressOf()));

        ComPtr<ID3DBlob> vs = CompileShader(L"CompositePS.hlsl", "VSMain", "vs_5_1");
        ComPtr<ID3DBlob> ps = CompileShader(L"CompositePS.hlsl", "PSMain", "ps_5_1");

        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = mRSComposite.Get();
        psoDesc.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
        psoDesc.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = mSwapFormat;
        psoDesc.SampleDesc.Count = 1;
        device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(mPSOComposite.ReleaseAndGetAddressOf()));
    }
}

void SSFRRenderer::UpdateCameraCB(const SSFRCamera& cam)
{
    CameraCBData data{};
    data.view = cam.view;
    data.proj = cam.proj;
    data.screenSize = XMFLOAT2(static_cast<float>(cam.width), static_cast<float>(cam.height));
    data.invScreenSize = XMFLOAT2(1.0f / std::max(1u, cam.width), 1.0f / std::max(1u, cam.height));
    data.nearZ = cam.nearZ;
    data.farZ = cam.farZ;
    WriteCB(mCameraCB, &data, sizeof(data));
    mCurrentNear = cam.nearZ;
    mCurrentFar = cam.farZ;
}

void SSFRRenderer::UpdateDrawCB()
{
    DrawCBData data{};
    data.particleRadius = mSettings.particleRadius;
    WriteCB(mDrawCB, &data, sizeof(data));
}

void SSFRRenderer::UpdateBlurCB(float depthSigma)
{
    BlurCBData data{};
    data.worldBlurRadius = mSettings.blurWorldRadius;
    data.depthSigma = depthSigma;
    data.enableBilateral = mSettings.enableBilateral ? 1u : 0u;
    WriteCB(mBlurCB, &data, sizeof(data));
}

void SSFRRenderer::UpdateCompositeCB()
{
    CompositeCBData data{};
    data.refractScale = mSettings.refractScale;
    data.absorbK = XMFLOAT3(mSettings.absorbK[0], mSettings.absorbK[1], mSettings.absorbK[2]);
    data.fluidColor = XMFLOAT3(mSettings.fluidColor[0], mSettings.fluidColor[1], mSettings.fluidColor[2]);
    data.flags = 0;
    if (mSettings.enableRefraction) data.flags |= 0x1;
    if (mSettings.enableBeerLambert) data.flags |= 0x2;
    if (mSettings.enableFresnel) data.flags |= 0x4;
    if (mHasSceneColor) data.flags |= 0x8;
    if (mHasSceneDepth) data.flags |= 0x10;
    if (mHasEnvironment) data.flags |= 0x20;
    WriteCB(mCompositeCB, &data, sizeof(data));
}

void SSFRRenderer::RenderDepthPass(ID3D12GraphicsCommandList* cmd, const SSFRParticleInputSRV& fluid)
{
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetDepthRaw(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
    cmd->ResourceBarrier(1, &barrier);

    FLOAT clear[4] = { mCurrentFar, 0, 0, 0 };
    auto depthRtv = mResources.GetRtvDepthRaw();
    cmd->ClearRenderTargetView(depthRtv, clear, 0, nullptr);

    D3D12_VIEWPORT viewport = { 0.0f, 0.0f, static_cast<float>(mWidth), static_cast<float>(mHeight), 0.0f, 1.0f };
    D3D12_RECT scissor = { 0, 0, static_cast<LONG>(mWidth), static_cast<LONG>(mHeight) };

    cmd->RSSetViewports(1, &viewport);
    cmd->RSSetScissorRects(1, &scissor);

    cmd->SetPipelineState(mPSODepth.Get());
    cmd->SetGraphicsRootSignature(mRSDepth.Get());
    cmd->SetGraphicsRootConstantBufferView(0, mCameraCB.resource->GetGPUVirtualAddress());
    cmd->SetGraphicsRootConstantBufferView(1, mDrawCB.resource->GetGPUVirtualAddress());
    cmd->SetGraphicsRootDescriptorTable(2, mResources.GetSrvGPU(SSFRResources::SrvParticlePositions));

    cmd->OMSetRenderTargets(1, &depthRtv, FALSE, nullptr);
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    cmd->DrawInstanced(4, fluid.count, 0, 0);

    barrier = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetDepthRaw(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &barrier);
}

void SSFRRenderer::BlurDepth(ID3D12GraphicsCommandList* cmd, uint32_t width, uint32_t height)
{
    width = std::max(1u, width);
    height = std::max(1u, height);
    auto barrierToUAV = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetDepthBlurX(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->ResourceBarrier(1, &barrierToUAV);

    cmd->SetPipelineState(mPSOBlurX.Get());
    cmd->SetComputeRootSignature(mRSBlur.Get());
    cmd->SetComputeRootConstantBufferView(0, mBlurCB.resource->GetGPUVirtualAddress());
    cmd->SetComputeRootDescriptorTable(1, mResources.GetSrvGPU(SSFRResources::SrvDepthRaw));
    cmd->SetComputeRootDescriptorTable(2, mResources.GetUavGPU(SSFRResources::UavDepthBlurX));

    uint32_t dispatchX = (width + 127) / 128;
    cmd->Dispatch(dispatchX, height, 1);

    auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(mResources.GetDepthBlurX());
    cmd->ResourceBarrier(1, &barrier);

    auto barrierToSRV = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetDepthBlurX(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &barrierToSRV);

    barrierToUAV = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetDepthBlurY(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->ResourceBarrier(1, &barrierToUAV);

    cmd->SetPipelineState(mPSOBlurY.Get());
    cmd->SetComputeRootSignature(mRSBlur.Get());
    cmd->SetComputeRootConstantBufferView(0, mBlurCB.resource->GetGPUVirtualAddress());
    cmd->SetComputeRootDescriptorTable(1, mResources.GetSrvGPU(SSFRResources::SrvDepthBlurX));
    cmd->SetComputeRootDescriptorTable(2, mResources.GetUavGPU(SSFRResources::UavDepthBlurY));

    cmd->Dispatch(dispatchX, height, 1);

    barrier = CD3DX12_RESOURCE_BARRIER::UAV(mResources.GetDepthBlurY());
    cmd->ResourceBarrier(1, &barrier);

    barrierToSRV = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetDepthBlurY(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &barrierToSRV);
}

void SSFRRenderer::RenderThickness(ID3D12GraphicsCommandList* cmd, const SSFRParticleInputSRV& fluid)
{
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetThickness(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
    cmd->ResourceBarrier(1, &barrier);

    FLOAT clear[4] = { 0,0,0,0 };
    auto thicknessRtv = mResources.GetRtvThickness();
    cmd->ClearRenderTargetView(thicknessRtv, clear, 0, nullptr);

    uint32_t width = mSettings.thicknessHalfRes ? std::max(1u, mWidth / 2) : mWidth;
    uint32_t height = mSettings.thicknessHalfRes ? std::max(1u, mHeight / 2) : mHeight;

    D3D12_VIEWPORT viewport = { 0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f };
    D3D12_RECT scissor = { 0, 0, static_cast<LONG>(width), static_cast<LONG>(height) };

    cmd->RSSetViewports(1, &viewport);
    cmd->RSSetScissorRects(1, &scissor);

    cmd->SetPipelineState(mPSOThickness.Get());
    cmd->SetGraphicsRootSignature(mRSThickness.Get());
    cmd->SetGraphicsRootConstantBufferView(0, mCameraCB.resource->GetGPUVirtualAddress());
    cmd->SetGraphicsRootConstantBufferView(1, mDrawCB.resource->GetGPUVirtualAddress());
    cmd->SetGraphicsRootDescriptorTable(2, mResources.GetSrvGPU(SSFRResources::SrvParticlePositions));

    cmd->OMSetRenderTargets(1, &thicknessRtv, FALSE, nullptr);
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    cmd->DrawInstanced(4, fluid.count, 0, 0);

    barrier = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetThickness(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &barrier);
}

void SSFRRenderer::BlurThickness(ID3D12GraphicsCommandList* cmd, uint32_t width, uint32_t height)
{
    width = std::max(1u, width);
    height = std::max(1u, height);
    auto barrierToUAV = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetThicknessBlur(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->ResourceBarrier(1, &barrierToUAV);

    cmd->SetPipelineState(mPSOThicknessBlurX.Get());
    cmd->SetComputeRootSignature(mRSThicknessBlur.Get());
    cmd->SetComputeRootConstantBufferView(0, mBlurCB.resource->GetGPUVirtualAddress());
    cmd->SetComputeRootDescriptorTable(1, mResources.GetSrvGPU(SSFRResources::SrvThickness));
    cmd->SetComputeRootDescriptorTable(2, mResources.GetUavGPU(SSFRResources::UavGeneric));

    uint32_t dispatchX = (width + 127) / 128;
    cmd->Dispatch(dispatchX, height, 1);

    auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(mResources.GetThicknessBlur());
    cmd->ResourceBarrier(1, &barrier);

    auto barrierToSRV = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetThicknessBlur(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &barrierToSRV);

    barrierToUAV = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetThickness(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->ResourceBarrier(1, &barrierToUAV);

    // UavGenericを厚みRTに更新
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_R16_FLOAT;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    mDevice->CreateUnorderedAccessView(mResources.GetThickness(), nullptr, &uavDesc, mResources.GetUavCPU(SSFRResources::UavGeneric));

    cmd->SetPipelineState(mPSOThicknessBlurY.Get());
    cmd->SetComputeRootSignature(mRSThicknessBlur.Get());
    cmd->SetComputeRootConstantBufferView(0, mBlurCB.resource->GetGPUVirtualAddress());
    cmd->SetComputeRootDescriptorTable(1, mResources.GetSrvGPU(SSFRResources::SrvThicknessBlur));
    cmd->SetComputeRootDescriptorTable(2, mResources.GetUavGPU(SSFRResources::UavGeneric));

    cmd->Dispatch(dispatchX, height, 1);

    barrier = CD3DX12_RESOURCE_BARRIER::UAV(mResources.GetThickness());
    cmd->ResourceBarrier(1, &barrier);

    barrierToSRV = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetThickness(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &barrierToSRV);

    // 厚み結果をThicknessBlurへコピー（Compositeで参照するスロットの整合性確保）。
    auto barrierCopySrc = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetThickness(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    auto barrierCopyDst = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetThicknessBlur(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    cmd->ResourceBarrier(1, &barrierCopySrc);
    cmd->ResourceBarrier(1, &barrierCopyDst);

    cmd->CopyResource(mResources.GetThicknessBlur(), mResources.GetThickness());

    barrierCopySrc = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetThickness(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    barrierCopyDst = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetThicknessBlur(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &barrierCopySrc);
    cmd->ResourceBarrier(1, &barrierCopyDst);

    // UavGenericを元に戻す
    uavDesc.Format = DXGI_FORMAT_R16_FLOAT;
    mDevice->CreateUnorderedAccessView(mResources.GetThicknessBlur(), nullptr, &uavDesc, mResources.GetUavCPU(SSFRResources::UavGeneric));
}

void SSFRRenderer::Composite(ID3D12GraphicsCommandList* cmd, D3D12_CPU_DESCRIPTOR_HANDLE rtvBackbuffer)
{
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetDepthBlurY(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &barrier);
    barrier = CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetThicknessBlur(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &barrier);

    D3D12_VIEWPORT viewport = { 0.0f, 0.0f, static_cast<float>(mWidth), static_cast<float>(mHeight), 0.0f, 1.0f };
    D3D12_RECT scissor = { 0, 0, static_cast<LONG>(mWidth), static_cast<LONG>(mHeight) };
    cmd->RSSetViewports(1, &viewport);
    cmd->RSSetScissorRects(1, &scissor);

    cmd->SetPipelineState(mPSOComposite.Get());
    cmd->SetGraphicsRootSignature(mRSComposite.Get());
    cmd->SetGraphicsRootConstantBufferView(0, mCameraCB.resource->GetGPUVirtualAddress());
    cmd->SetGraphicsRootConstantBufferView(1, mCompositeCB.resource->GetGPUVirtualAddress());
    cmd->SetGraphicsRootDescriptorTable(2, mResources.GetSrvGPU(SSFRResources::SrvDepthBlurY));
    cmd->SetGraphicsRootDescriptorTable(3, mResources.GetSrvGPU(SSFRResources::SrvThicknessBlur));

    cmd->OMSetRenderTargets(1, &rtvBackbuffer, FALSE, nullptr);
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd->DrawInstanced(3, 1, 0, 0);

    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetDepthBlurY(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON));
    barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetThicknessBlur(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON));
    barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetDepthRaw(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON));
    barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetDepthBlurX(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON));
    barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(mResources.GetThickness(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON));
    cmd->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
}

void SSFRRenderer::PrepareSceneDescriptors(ID3D12Device* device, ID3D12Resource* sceneColor, ID3D12Resource* sceneDepth)
{
    // シーン側リソースの有無に応じてフラグを更新し、合成シェーダの条件分岐に反映する。
    mHasSceneColor = sceneColor != nullptr;
    mHasSceneDepth = sceneDepth != nullptr;
    mHasEnvironment = mEnvironmentCube != nullptr;
    mResources.SetSceneResources(device, sceneColor, sceneDepth, mEnvironmentCube.Get());
    UpdateCompositeCB();
}

void SSFRRenderer::AllocateCB(ID3D12Device* device, ConstantBufferData& cb, size_t size, const char* debugName)
{
    size_t align = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
    size_t aligned = (size + (align - 1)) & ~(align - 1);

    CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(aligned);

    device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(cb.resource.ReleaseAndGetAddressOf()));
    cb.resource->Map(0, nullptr, reinterpret_cast<void**>(&cb.mapped));
    cb.size = aligned;
    if (debugName)
    {
        wchar_t wname[128];
        swprintf(wname, 128, L"%S", debugName);
        cb.resource->SetName(wname);
    }
}

void SSFRRenderer::WriteCB(ConstantBufferData& cb, const void* data, size_t size)
{
    memcpy(cb.mapped, data, size);
}
