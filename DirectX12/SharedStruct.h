#pragma once
#include <d3dx12.h>
#include <DirectXMath.h>
#include "ComPtr.h"
#include <vector>

struct Vertex
{
	DirectX::XMFLOAT3 Position;
	DirectX::XMFLOAT3 Normal;
	DirectX::XMFLOAT2 UV;
	DirectX::XMFLOAT3 Tangent;
	DirectX::XMFLOAT4 Color;

	static const D3D12_INPUT_LAYOUT_DESC InputLayout;

private:
	static const int InputElementCount = 5;
	static const D3D12_INPUT_ELEMENT_DESC InputElements[InputElementCount];
};

struct alignas(256) Transform
{
	DirectX::XMMATRIX World;
	DirectX::XMMATRIX View;
	DirectX::XMMATRIX Proj;
};

struct Mesh {
	std::vector<Vertex> Vertices;
	std::vector<uint32_t> Indices;
	std::wstring DiffuseMap;
};

struct ParticleVertex {
	DirectX::XMFLOAT3 position;

	static const D3D12_INPUT_ELEMENT_DESC InputElements[1];
	static const D3D12_INPUT_LAYOUT_DESC InputLayout;
	static constexpr UINT InputElementCount = _countof(InputElements);
};
//struct ParticleParam {
//        DirectX::XMFLOAT3 position;
//        DirectX::XMFLOAT3 velocity;
//};

// メタボール用の粒子情報
struct ParticleMeta
{
	DirectX::XMFLOAT3 pos;   // ワールド空間位置
	float               r;    // 半径
};