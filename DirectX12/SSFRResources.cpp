#include "SSFRResources.h"
#include <d3dx12.h>

using Microsoft::WRL::ComPtr;

namespace {
    constexpr UINT kSrvCount = SSFRResources::SrvCount;
    constexpr UINT kUavCount = SSFRResources::UavCount;
    constexpr DXGI_FORMAT kDepthFormat = DXGI_FORMAT_R32_FLOAT;
    constexpr DXGI_FORMAT kThicknessFormat = DXGI_FORMAT_R16_FLOAT;
}

void SSFRResources::Initialize(ID3D12Device* device, DXGI_FORMAT swapFormat)
{
    mSwapFormat = swapFormat;
    CreateHeaps(device);
}

void SSFRResources::Resize(ID3D12Device* device, uint32_t width, uint32_t height, bool thicknessHalfRes)
{
    mWidth = width;
    mHeight = height;
    mThicknessHalfRes = thicknessHalfRes;
    CreateInternalTextures(device);
    CreateDescriptors(device);
}

void SSFRResources::SetSceneResources(ID3D12Device* device, ID3D12Resource* sceneColor, ID3D12Resource* sceneDepth, ID3D12Resource* envCube)
{
    mSceneColor = sceneColor;
    mSceneDepth = sceneDepth;
    mEnvCube = envCube;
    CreateSceneSRV(device, sceneColor, SrvSceneColor);
    CreateSceneSRV(device, sceneDepth, SrvSceneDepth);
    CreateSceneSRV(device, envCube, SrvEnvironment);
}

D3D12_CPU_DESCRIPTOR_HANDLE SSFRResources::GetSrvCPU(SrvSlot slot) const
{
    D3D12_CPU_DESCRIPTOR_HANDLE handle = mSrvUavHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += slot * mSrvDescriptorSize;
    return handle;
}

D3D12_GPU_DESCRIPTOR_HANDLE SSFRResources::GetSrvGPU(SrvSlot slot) const
{
    D3D12_GPU_DESCRIPTOR_HANDLE handle = mSrvUavHeap->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += slot * mSrvDescriptorSize;
    return handle;
}

D3D12_CPU_DESCRIPTOR_HANDLE SSFRResources::GetUavCPU(UavSlot slot) const
{
    D3D12_CPU_DESCRIPTOR_HANDLE handle = mSrvUavHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += (kSrvCount + slot) * mSrvDescriptorSize;
    return handle;
}

D3D12_GPU_DESCRIPTOR_HANDLE SSFRResources::GetUavGPU(UavSlot slot) const
{
    D3D12_GPU_DESCRIPTOR_HANDLE handle = mSrvUavHeap->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += (kSrvCount + slot) * mSrvDescriptorSize;
    return handle;
}

D3D12_CPU_DESCRIPTOR_HANDLE SSFRResources::GetRtvDepthRaw() const
{
    D3D12_CPU_DESCRIPTOR_HANDLE handle = mRtvHeap->GetCPUDescriptorHandleForHeapStart();
    return handle;
}

D3D12_CPU_DESCRIPTOR_HANDLE SSFRResources::GetRtvThickness() const
{
    D3D12_CPU_DESCRIPTOR_HANDLE handle = mRtvHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += mRtvDescriptorSize;
    return handle;
}

void SSFRResources::CreateHeaps(ID3D12Device* device)
{
    D3D12_DESCRIPTOR_HEAP_DESC srvDesc = {};
    srvDesc.NumDescriptors = kSrvCount + kUavCount;
    srvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    device->CreateDescriptorHeap(&srvDesc, IID_PPV_ARGS(mSrvUavHeap.ReleaseAndGetAddressOf()));
    mSrvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc = {};
    rtvDesc.NumDescriptors = 2;
    rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(mRtvHeap.ReleaseAndGetAddressOf()));
    mRtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
}

void SSFRResources::CreateInternalTextures(ID3D12Device* device)
{
    CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_DEFAULT);

    auto createRT = [&](ComPtr<ID3D12Resource>& target, DXGI_FORMAT format, uint32_t w, uint32_t h, D3D12_RESOURCE_FLAGS flags)
    {
        if (w == 0 || h == 0)
        {
            target.Reset();
            return;
        }
        CD3DX12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Tex2D(format, w, h, 1, 1, 1, 0, flags);
        D3D12_CLEAR_VALUE clear = {};
        clear.Format = format;
        clear.Color[0] = 0.0f;
        clear.Color[1] = 0.0f;
        clear.Color[2] = 0.0f;
        clear.Color[3] = 0.0f;
        device->CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            (flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) ? &clear : nullptr,
            IID_PPV_ARGS(target.ReleaseAndGetAddressOf()));
    };

    createRT(mDepthRaw, kDepthFormat, mWidth, mHeight, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    createRT(mDepthBlurX, kDepthFormat, mWidth, mHeight, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    createRT(mDepthBlurY, kDepthFormat, mWidth, mHeight, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    uint32_t thicknessW = mThicknessHalfRes ? (mWidth / 2) : mWidth;
    uint32_t thicknessH = mThicknessHalfRes ? (mHeight / 2) : mHeight;
    thicknessW = thicknessW > 0 ? thicknessW : 1;
    thicknessH = thicknessH > 0 ? thicknessH : 1;
    createRT(mThickness, kThicknessFormat, thicknessW, thicknessH, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    createRT(mThicknessBlur, kThicknessFormat, thicknessW, thicknessH, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
}

void SSFRResources::CreateDescriptors(ID3D12Device* device)
{
    if (!mSrvUavHeap)
    {
        CreateHeaps(device);
    }

    auto srvHandle = mSrvUavHeap->GetCPUDescriptorHandleForHeapStart();

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;

    srvDesc.Format = kDepthFormat;
    device->CreateShaderResourceView(mDepthRaw.Get(), &srvDesc, srvHandle);

    srvHandle.ptr += mSrvDescriptorSize;
    device->CreateShaderResourceView(mDepthBlurX.Get(), &srvDesc, srvHandle);

    srvHandle.ptr += mSrvDescriptorSize;
    device->CreateShaderResourceView(mDepthBlurY.Get(), &srvDesc, srvHandle);

    srvHandle.ptr += mSrvDescriptorSize;
    srvDesc.Format = kThicknessFormat;
    device->CreateShaderResourceView(mThickness.Get(), &srvDesc, srvHandle);

    srvHandle.ptr += mSrvDescriptorSize;
    device->CreateShaderResourceView(mThicknessBlur.Get(), &srvDesc, srvHandle);

    srvHandle.ptr += mSrvDescriptorSize;
    CreateSceneSRV(device, mSceneColor.Get(), SrvSceneColor);

    srvHandle.ptr += mSrvDescriptorSize;
    CreateSceneSRV(device, mSceneDepth.Get(), SrvSceneDepth);

    srvHandle.ptr += mSrvDescriptorSize;
    CreateSceneSRV(device, mEnvCube.Get(), SrvEnvironment);

    srvHandle.ptr += mSrvDescriptorSize;
    device->CreateShaderResourceView(nullptr, nullptr, srvHandle);

    auto uavHandle = mSrvUavHeap->GetCPUDescriptorHandleForHeapStart();
    uavHandle.ptr += kSrvCount * mSrvDescriptorSize;

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uavDesc.Format = kDepthFormat;
    device->CreateUnorderedAccessView(mDepthBlurX.Get(), nullptr, &uavDesc, uavHandle);

    uavHandle.ptr += mSrvDescriptorSize;
    device->CreateUnorderedAccessView(mDepthBlurY.Get(), nullptr, &uavDesc, uavHandle);

    uavHandle.ptr += mSrvDescriptorSize;
    uavDesc.Format = kThicknessFormat;
    device->CreateUnorderedAccessView(mThicknessBlur.Get(), nullptr, &uavDesc, uavHandle);

    auto rtvHandle = mRtvHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_RENDER_TARGET_VIEW_DESC rtvDesc = {};
    rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    rtvDesc.Format = kDepthFormat;
    device->CreateRenderTargetView(mDepthRaw.Get(), &rtvDesc, rtvHandle);

    rtvHandle.ptr += mRtvDescriptorSize;
    rtvDesc.Format = kThicknessFormat;
    device->CreateRenderTargetView(mThickness.Get(), &rtvDesc, rtvHandle);
}

void SSFRResources::CreateSceneSRV(ID3D12Device* device, ID3D12Resource* resource, SrvSlot slot)
{
    D3D12_CPU_DESCRIPTOR_HANDLE handle = GetSrvCPU(slot);
    if (!resource)
    {
        device->CreateShaderResourceView(nullptr, nullptr, handle);
        return;
    }

    D3D12_RESOURCE_DESC desc = resource->GetDesc();

    if (slot == SrvEnvironment && desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
        if (desc.DepthOrArraySize >= 6)
        {
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
            srv.TextureCube.MipLevels = desc.MipLevels;
        }
        else
        {
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Texture2D.MipLevels = desc.MipLevels;
        }
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Format = desc.Format;
        device->CreateShaderResourceView(resource, &srv, handle);
        return;
    }

    if (slot == SrvEnvironment && desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Format = desc.Format;
        srv.TextureCube.MipLevels = desc.MipLevels;
        device->CreateShaderResourceView(resource, &srv, handle);
        return;
    }

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Format = desc.Format;
    srvDesc.Texture2D.MipLevels = desc.MipLevels;
    device->CreateShaderResourceView(resource, &srvDesc, handle);
}
