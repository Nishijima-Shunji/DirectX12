#include "DescriptorHeap.h"
#include "Texture2D.h"
#include <d3dx12.h>
#include "Engine.h"

const UINT HANDLE_MAX = 512;

DescriptorHeap::DescriptorHeap()
{
	m_pHandles.clear();
	m_pHandles.reserve(HANDLE_MAX);

	D3D12_DESCRIPTOR_HEAP_DESC desc{};
	desc.NodeMask = 1;
	desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	desc.NumDescriptors = HANDLE_MAX;
	desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

	auto device = g_Engine->Device();

	// �f�B�X�N���v�^�q�[�v�𐶐�
	auto hr = device->CreateDescriptorHeap(
		&desc,
		IID_PPV_ARGS(m_pHeap.ReleaseAndGetAddressOf()));

	if (FAILED(hr))
	{
		m_IsValid = false;
		return;
	}

	m_IncrementSize = device->GetDescriptorHandleIncrementSize(desc.Type); // �f�B�X�N���v�^�q�[�v1�̃������T�C�Y��Ԃ�
	m_IsValid = true;
}

ID3D12DescriptorHeap* DescriptorHeap::GetHeap()
{
	return m_pHeap.Get();
}

DescriptorHandle* DescriptorHeap::Register(Texture2D* texture)
{
	auto count = m_pHandles.size();
	if (HANDLE_MAX <= count)
	{
		return nullptr;
	}

	DescriptorHandle* pHandle = new DescriptorHandle();

	auto handleCPU = m_pHeap->GetCPUDescriptorHandleForHeapStart(); // �f�B�X�N���v�^�q�[�v�̍ŏ��̃A�h���X
	handleCPU.ptr += m_IncrementSize * count; // �ŏ��̃A�h���X����count�Ԗڂ�����ǉ����ꂽ���\�[�X�̃n���h��

	auto handleGPU = m_pHeap->GetGPUDescriptorHandleForHeapStart(); // �f�B�X�N���v�^�q�[�v�̍ŏ��̃A�h���X
	handleGPU.ptr += m_IncrementSize * count; // �ŏ��̃A�h���X����count�Ԗڂ�����ǉ����ꂽ���\�[�X�̃n���h��

	pHandle->HandleCPU = handleCPU;
	pHandle->HandleGPU = handleGPU;

	auto device = g_Engine->Device();
	auto resource = texture->Resource();
	auto desc = texture->ViewDesc();
	device->CreateShaderResourceView(resource, &desc, pHandle->HandleCPU); // �V�F�[�_�[���\�[�X�r���[�쐬

	m_pHandles.push_back(pHandle);
	return pHandle; // �n���h����Ԃ�
}

DescriptorHandle* DescriptorHeap::RegisterBuffer(
	ID3D12Resource* resource,
	UINT            numElements,
	UINT            stride)
{
	auto count = m_pHandles.size();
	if (HANDLE_MAX <= count) return nullptr;

	auto pHandle = new DescriptorHandle();

	// CPU/GPU �����̃n���h�����擾
	auto cpu = m_pHeap->GetCPUDescriptorHandleForHeapStart();
	cpu.ptr += m_IncrementSize * count;
	auto gpu = m_pHeap->GetGPUDescriptorHandleForHeapStart();
	gpu.ptr += m_IncrementSize * count;

	pHandle->HandleCPU = cpu;
	pHandle->HandleGPU = gpu;

	// SRV �f�X�N���v�^���쐬
	D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
	desc.Format = DXGI_FORMAT_UNKNOWN;
	desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	desc.Buffer.NumElements = numElements;
	desc.Buffer.StructureByteStride = stride;
	desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

	// ���ۂ� GPU �փr���[����������
	g_Engine->Device()->CreateShaderResourceView(
		resource, &desc, cpu);

	m_pHandles.push_back(pHandle);
	return pHandle;
}
