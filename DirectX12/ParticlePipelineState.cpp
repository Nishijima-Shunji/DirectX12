#include "ParticlePipelineState.h"
#include "Engine.h"
#include <d3dx12.h>
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")

ParticlePipelineState::ParticlePipelineState()
{
	// �p�C�v���C���X�e�[�g�̐ݒ�
	
	// ���X�^���C�U�[�̐ݒ�
	desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);			// ���X�^���C�U�[�̓f�t�H���g
	desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;					// �J�����O�͂Ȃ�
	desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;					// �h��Ԃ��̓\���b�h

	// �[�x�X�e���V���̐ݒ�
	desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);		// �[�x�X�e���V���̓f�t�H���g
	desc.DepthStencilState.DepthEnable = TRUE;								// �[�x�X�e���V���𖳌�
	desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;	// �[�x�������݂͖���
	desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;		// �[�x��r�֐��͏�ɒʉ�

	desc.SampleMask = UINT_MAX;
	desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;	// �`����@
	desc.NumRenderTargets = 1;												// �`��Ώۂ�1
	desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;

	// �T���v�����̐ݒ�
	desc.SampleDesc.Count = 1;												// �T���v���[��1
	desc.SampleDesc.Quality = 0;

	// �u�����h�X�e�[�g�̐ݒ�
	desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);					// �u�����h�X�e�[�g���f�t�H���g
	desc.BlendState.RenderTarget[0].BlendEnable = TRUE;						// �u�����h�͗L��
	desc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;		// �\�[�X�u�����h�̓A���t�@
	desc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;	// �f�X�g�u�����h�͋t�A���t�@
	desc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;			// �u�����h���Z�͉��Z
	desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL; // �S�Ă̐F�������������݉\
}

bool ParticlePipelineState::IsValid()
{
	return m_IsValid;
}

void ParticlePipelineState::SetInputLayout(D3D12_INPUT_LAYOUT_DESC layout)
{
	desc.InputLayout = layout;
}

void ParticlePipelineState::SetRootSignature(ID3D12RootSignature* rootSignature)
{
	desc.pRootSignature = rootSignature;
}

void ParticlePipelineState::SetVS(std::wstring filePath)
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

void ParticlePipelineState::SetPS(std::wstring filePath)
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

void ParticlePipelineState::Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE type)
{
	desc.PrimitiveTopologyType = type;
	// �p�C�v���C���X�e�[�g�𐶐�
	auto hr = g_Engine->Device()->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(m_pPipelineState.ReleaseAndGetAddressOf()));
	if (FAILED(hr))
	{
		printf("�p�C�v���C���X�e�[�g�̐����Ɏ��s HRESULT=0x%08X\n", hr);
		return;
	}

	m_IsValid = true;
}

ID3D12PipelineState* ParticlePipelineState::Get()
{
	return m_pPipelineState.Get();
}
