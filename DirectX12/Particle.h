#pragma once
#include "Object.h"
#include "Camera.h"
#include <directxMath.h>
#include "Engine.h"
#include "VertexBuffer.h"
#include "RootSignature.h"
#include "PipelineState.h"
#include "ConstantBuffer.h"

struct Point {
	DirectX::XMFLOAT3 position = {};
	DirectX::XMFLOAT3 velocity = {};
};

class Particle : public Object
{
private:
	std::vector<Point>		m_Particles;
	VertexBuffer* m_VertexBuffer = nullptr;
	RootSignature* m_RootSignature = nullptr;
	PipelineState* m_PipelineState = nullptr;
	ConstantBuffer* m_ConstantBuffer[Engine::FRAME_BUFFER_COUNT] = {};

	// グラフィックス描画用
	ComPtr<ID3D12RootSignature>    m_graphicsRS;
	PipelineState*				   m_graphicsPS;
	ComPtr<ID3D12Resource>         m_graphicsCB;

	// コンピュート用
	ComPtr<ID3D12RootSignature>    m_computeRS;
	PipelineState*				   m_computePS;
	ComPtr<ID3D12Resource>         m_computeUAV;
	ComPtr<ID3D12DescriptorHeap>   m_uavHeap;
	ComPtr<ID3D12Resource>         m_particleBuffer;


	Camera* camera;
public:
	Particle(Camera* cam);
	bool Init();
	void Update(float deltaTime);
	void Draw();

	void UpdateParticles();
	void UpdateVertexBuffer();
};