#include "ConstantBuffer.h"
#include "Engine.h"

#include <stdexcept>

ConstantBuffer::ConstantBuffer(size_t size)
{
    const size_t align = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
    const UINT64 sizeAligned = (size + (align - 1)) & ~(align - 1); // alignɐ؂グ.

    const HRESULT createResult = g_Engine->Device()->CreateCommittedResource(

    if (FAILED(createResult))
        // sԂƌŖobt@փANZXĕsɂȂ邽߁AOŖB
        throw std::runtime_error("ConstantBuffer resource creation failed");
    const HRESULT mapResult = m_pBuffer->Map(0, nullptr, &m_pMappedPtr);
    if (FAILED(mapResult) || !m_pMappedPtr)
    {
        // }bvsɃ\[XȂƃ[N邽߁AĂO𓊂B
        m_pBuffer.Reset();
        throw std::runtime_error("ConstantBuffer map failed");
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

