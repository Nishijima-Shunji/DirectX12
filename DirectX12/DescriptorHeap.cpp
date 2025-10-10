#include "DescriptorHeap.h"
#include "Texture2D.h"
#include <d3dx12.h>
#include "Engine.h"

const UINT HANDLE_MAX = 512;

DescriptorHeap::DescriptorHeap()
{
        m_Handles.clear();
        m_Handles.reserve(HANDLE_MAX);

        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        desc.NodeMask = 1;
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        desc.NumDescriptors = HANDLE_MAX;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

        auto device = g_Engine->Device();
        if (!device)
        {
                // foCXƈȍ~̃\[Xo^j]邽߁AsɂB
                m_IsValid = false;
                return;
        }

        const HRESULT hr = device->CreateDescriptorHeap(
                &desc,
                IID_PPV_ARGS(m_pHeap.ReleaseAndGetAddressOf()));

        if (FAILED(hr))
        {
                // q[vsƑSĂ SRV/UAV o^ɂȂ邽߁Aŏł؂B
                m_IsValid = false;
                return;
        }

        m_IncrementSize = device->GetDescriptorHandleIncrementSize(desc.Type); // fBXNv^ 1 ̃oCgoĂ
        m_IsValid = true;
}

ID3D12DescriptorHeap* DescriptorHeap::GetHeap()
{
        return m_pHeap.Get();
}

DescriptorHandle* DescriptorHeap::Register(Texture2D* texture)
{
        if (!texture)
        {
                return nullptr;
        }
        return Register(texture->Resource(), texture->ViewDesc());
}

DescriptorHandle* DescriptorHeap::Register(ID3D12Resource* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC& desc)
{
        if (!resource)
        {
                return nullptr;
        }

        auto device = g_Engine->Device();
        if (!device)
        {
                return nullptr;
        }

        DescriptorHandle* handle = AllocateHandle();
        if (!handle)
        {
                return nullptr;
        }

        device->CreateShaderResourceView(resource, &desc, handle->HandleCPU);

        return handle;
}

DescriptorHandle* DescriptorHeap::RegisterBuffer(
        ID3D12Resource* resource,
        UINT            numElements,
        UINT            stride)
{
        if (!resource)
        {
                return nullptr;
        }

        auto device = g_Engine->Device();
        if (!device)
        {
                return nullptr;
        }

        DescriptorHandle* handle = AllocateHandle();
        if (!handle)
        {
                return nullptr;
        }

        D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc.Buffer.NumElements = numElements;
        desc.Buffer.StructureByteStride = stride;
        desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

        device->CreateShaderResourceView(resource, &desc, handle->HandleCPU);

        return handle;
}

DescriptorHandle* DescriptorHeap::RegisterBufferUAV(
        ID3D12Resource* resource,
        UINT            numElements,
        UINT            stride)
{
        if (!resource)
        {
                return nullptr;
        }

        auto device = g_Engine->Device();
        if (!device)
        {
                return nullptr;
        }

        DescriptorHandle* handle = AllocateHandle();
        if (!handle)
        {
                return nullptr;
        }

        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        desc.Buffer.NumElements = numElements;
        desc.Buffer.StructureByteStride = stride;
        desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

        device->CreateUnorderedAccessView(
                resource, nullptr, &desc, handle->HandleCPU);

        return handle;
}

DescriptorHandle* DescriptorHeap::RegisterTextureUAV(
        ID3D12Resource* resource,
        DXGI_FORMAT      format)
{
        if (!resource)
        {
                return nullptr;
        }

        auto device = g_Engine->Device();
        if (!device)
        {
                return nullptr;
        }

        DescriptorHandle* handle = AllocateHandle();
        if (!handle)
        {
                return nullptr;
        }

        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
        desc.Format = format;
        desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        desc.Texture2D.MipSlice = 0;
        desc.Texture2D.PlaneSlice = 0;

        device->CreateUnorderedAccessView(
                resource, nullptr, &desc, handle->HandleCPU);

        return handle;
}

DescriptorHandle* DescriptorHeap::AllocateHandle()
{
        if (!m_IsValid || !m_pHeap)
        {
                return nullptr;
        }

        const size_t index = m_Handles.size();
        if (index >= HANDLE_MAX)
        {
                // 𒴂ƓfBXNv^̈Ă܂߁AŊ蓖Ă~߂B
                return nullptr;
        }

        DescriptorHandle& handle = m_Handles.emplace_back();

        auto cpu = m_pHeap->GetCPUDescriptorHandleForHeapStart();
        cpu.ptr += m_IncrementSize * index;
        auto gpu = m_pHeap->GetGPUDescriptorHandleForHeapStart();
        gpu.ptr += m_IncrementSize * index;

        handle.HandleCPU = cpu;
        handle.HandleGPU = gpu;

        return &handle;
}
