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
        DescriptorHandle* Register(Texture2D* texture); // eNX`[fBXNv^q[vɓo^AnhԂ
        ~DescriptorHeap();
	ID3D12DescriptorHeap* GetHeap(); // ディスクリプタヒープを返す
	DescriptorHandle* Register(Texture2D* texture); // テクスチャーをディスクリプタヒープに登録し、ハンドルを返す
	DescriptorHandle* RegisterBuffer(
		ID3D12Resource* resource,
		UINT            numElements,
		UINT            stride
	);

private:
	bool m_IsValid = false; // 生成に成功したかどうか
	UINT m_IncrementSize = 0;
	ComPtr<ID3D12DescriptorHeap> m_pHeap = nullptr; // ディスクリプタヒープ本体
	std::vector<DescriptorHandle*> m_pHandles; // 登録されているハンドル

};
