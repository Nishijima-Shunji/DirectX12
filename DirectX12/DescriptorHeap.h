#pragma once
#include "ComPtr.h"
#include <d3dx12.h>
#include <vector>

class ConstantBuffer;
class Texture2D;

class DescriptorHandle
{
public:
        D3D12_CPU_DESCRIPTOR_HANDLE HandleCPU;
        D3D12_GPU_DESCRIPTOR_HANDLE HandleGPU;
};

class DescriptorHeap
{
public:
        DescriptorHeap(); // RXgN^ŃfBXNv^q[vmۂ
        ID3D12DescriptorHeap* GetHeap(); // fBXNv^q[vԂ
        DescriptorHandle* Register(Texture2D* texture); // eNX`[fBXNv^q[vɓo^nhԂ
    DescriptorHandle* Register(ID3D12Resource* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC& desc);
    DescriptorHandle* RegisterBuffer(
            ID3D12Resource* resource,
            UINT            numElements,
            UINT            stride
    );

    DescriptorHandle* RegisterBufferUAV(
            ID3D12Resource* resource,
            UINT            numElements,
            UINT            stride
    );

    DescriptorHandle* RegisterTextureUAV(
            ID3D12Resource* resource,
            DXGI_FORMAT      format
    );

private:
        DescriptorHandle* AllocateHandle(); // nh̊蓖ďʉďdR[hh
        bool m_IsValid = false; // ɐǂ
        UINT m_IncrementSize = 0;
        ComPtr<ID3D12DescriptorHeap> m_pHeap = nullptr; // fBXNv^q[v{
        std::vector<DescriptorHandle> m_Handles; // lŕێ new/delete ̊Ǘsvɂ

};
