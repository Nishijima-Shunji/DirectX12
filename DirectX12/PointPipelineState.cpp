#include "PointPipelineState.h"
#include "Engine.h"
#include <d3dx12.h>
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")

PointPipelineState::PointPipelineState()
{
	// �p�C�v���C���X�e�[�g�̐ݒ�
	desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);		// ���X�^���C�U�[�̓f�t�H���g
	desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;				// �J�����O�͂Ȃ�
	desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);				// �u�����h�X�e�[�g���f�t�H���g
	desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT); // �[�x�X�e���V���̓f�t�H���g���g��
	desc.SampleMask = UINT_MAX;
	desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;	// �_��`��
	desc.NumRenderTargets = 1;											// �`��Ώۂ�1
	desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
	desc.SampleDesc.Count = 1;											// �T���v���[��1
	desc.SampleDesc.Quality = 0;
}

bool PointPipelineState::IsValid()
{
	return m_IsValid;
}

void PointPipelineState::SetInputLayout(D3D12_INPUT_LAYOUT_DESC layout)
{
	desc.InputLayout = layout;
}

void PointPipelineState::SetRootSignature(ID3D12RootSignature* rootSignature)
{
	desc.pRootSignature = rootSignature;
}

void PointPipelineState::SetVS(std::wstring filePath)
{
	// ���_�V�F�[�_�[�ǂݍ���
	auto hr = D3DReadFileToBlob(filePath.c_str(), m_pVsBlob.GetAddressOf());
	if (FAILED(hr))
	{
		printf("���_�V�F�[�_�[�̓ǂݍ��݂Ɏ��s\n");
		return;
	}

	desc.VS = CD3DX12_SHADER_BYTECODE(m_pVsBlob.Get());
}

void PointPipelineState::SetPS(std::wstring filePath)
{
	// �s�N�Z���V�F�[�_�[�ǂݍ���
	auto hr = D3DReadFileToBlob(filePath.c_str(), m_pPSBlob.GetAddressOf());
	if (FAILED(hr))
	{
		printf("�s�N�Z���V�F�[�_�[�̓ǂݍ��݂Ɏ��s\n");
		return;
	}

	desc.PS = CD3DX12_SHADER_BYTECODE(m_pPSBlob.Get());
}

void PointPipelineState::Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE type)
{
	desc.PrimitiveTopologyType = type;

	printf("VS size: %zu, PS size: %zu\n", desc.VS.BytecodeLength, desc.PS.BytecodeLength);
	printf("InputLayout.Elements: %u\n", desc.InputLayout.NumElements);
	printf("SemanticName: %s\n", desc.InputLayout.pInputElementDescs[0].SemanticName);
	// �p�C�v���C���X�e�[�g�𐶐�
	auto hr = g_Engine->Device()->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(m_pPipelineState.ReleaseAndGetAddressOf()));
	if (FAILED(hr))
	{
		printf("�p�C�v���C���X�e�[�g�̐����Ɏ��s HRESULT=0x%08X\n", hr);
		return;
	}

	m_IsValid = true;
}

ID3D12PipelineState* PointPipelineState::Get()
{
	return m_pPipelineState.Get();
}
