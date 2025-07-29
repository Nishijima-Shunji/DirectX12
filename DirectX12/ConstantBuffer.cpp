#include "ConstantBuffer.h"
#include "Engine.h"

ConstantBuffer::ConstantBuffer(size_t size)
{
    size_t align = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
    UINT64 sizeAligned = (size + (align - 1)) & ~(align - 1); // alignに切り上げる.

    auto prop = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD); // ヒーププロパティ
    auto desc = CD3DX12_RESOURCE_DESC::Buffer(sizeAligned); // リソースの設定

    // リソースを生成
    auto hr = g_Engine->Device()->CreateCommittedResource(
        &prop,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(m_pBuffer.GetAddressOf()));
    if (FAILED(hr))
    {
        printf("定数バッファリソースの生成に失敗\n");
        return;
    }
    else
    {
        printf("定数バッファリソースの生成 成功\n");
    }


    hr = m_pBuffer->Map(0, nullptr, &m_pMappedPtr);
    if (!m_pBuffer) {
        printf("m_pBufferが空です\n");
        return;
    }

    auto addr = m_pBuffer->GetGPUVirtualAddress();

    m_Desc = {};
    m_Desc.BufferLocation = m_pBuffer->GetGPUVirtualAddress();
    m_Desc.SizeInBytes = UINT(sizeAligned);

    m_IsValid = true;
}

bool ConstantBuffer::IsValid()
{
    return m_IsValid;
}

D3D12_GPU_VIRTUAL_ADDRESS ConstantBuffer::GetAddress() const
{
    return m_Desc.BufferLocation;
}

D3D12_CONSTANT_BUFFER_VIEW_DESC ConstantBuffer::ViewDesc()
{
    return m_Desc;
}

void* ConstantBuffer::GetPtr() const
{
    return m_pMappedPtr;
}

ConstantBuffer::~ConstantBuffer()
{
    if (m_pBuffer) {
        m_pBuffer->Unmap(0, nullptr);
    }
    m_pBuffer.Reset();
}
