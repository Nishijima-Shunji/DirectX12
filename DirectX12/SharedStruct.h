#pragma once
#include <d3dx12.h>
#include <DirectXMath.h>
#include "ComPtr.h"
#include <vector>

struct Vertex
{
	DirectX::XMFLOAT3 Position; // �ʒu���W
	DirectX::XMFLOAT3 Normal;   // �@��
	DirectX::XMFLOAT2 UV;       // uv���W
	DirectX::XMFLOAT3 Tangent;  // �ڋ��
	DirectX::XMFLOAT4 Color;    // ���_�F

	static const D3D12_INPUT_LAYOUT_DESC InputLayout;

private:
	static const int InputElementCount = 5;
	static const D3D12_INPUT_ELEMENT_DESC InputElements[InputElementCount];
};

struct alignas(256) Transform
{	
	DirectX::XMMATRIX World;    // ���[���h�s��
	DirectX::XMMATRIX View;     // �r���[�s��
	DirectX::XMMATRIX Proj;     // ���e�s��
};

struct Mesh {
	std::vector<Vertex> Vertices;	// ���_�f�[�^�̔z��
	std::vector<uint32_t> Indices;	// �C���f�b�N�X�̔z��
	std::wstring DiffuseMap;		// �e�N�X�`���̃t�@�C���p�X
};



struct ParticleVertex {
	DirectX::XMFLOAT3 position;

	static const D3D12_INPUT_ELEMENT_DESC ParticleInputElements[5];
	static const D3D12_INPUT_LAYOUT_DESC ParticleInputLayout;
	static constexpr UINT InputElementCount = _countof(ParticleInputElements);
};

_declspec(align(16))
struct MetaballCB
{
	float threshold;      // HLSL b0 : threshold
	float eps;            // b0 : eps
	float maxSum;         // b0 : maxSum
	float _pad0;          // �p�f�B���O�i16�o�C�g���E���킹�j

	DirectX::XMFLOAT4 color;	// b0 : float4 color
	UINT    particleCount;		// b0 : uint particleCount
	DirectX::XMFLOAT3 _pad1;	// �p�f�B���O�i16�o�C�g���E���킹�j
};