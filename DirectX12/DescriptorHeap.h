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
	DescriptorHeap(); // コンストラクタで生す
	ID3D12DescriptorHeap* GetHeap(); // ィスクリプタヒプを返す
	DescriptorHandle* Register(Texture2D* texture); // クスチャーをディスクリプタヒプに登録しハンドルを返す
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
	bool m_IsValid = false; // 生に成功したかど
	UINT m_IncrementSize = 0;
	ComPtr<ID3D12DescriptorHeap> m_pHeap = nullptr; // ィスクリプタヒプ本
	std::vector<DescriptorHandle*> m_pHandles; // 登録されてるハンドル

};
