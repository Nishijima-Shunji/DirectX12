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
	VertexBuffer*			m_VertexBuffer	= nullptr;
	RootSignature*			m_RootSignature = nullptr;
	PipelineState*			m_PipelineState = nullptr;
	ConstantBuffer*			m_ConstantBuffer[Engine::FRAME_BUFFER_COUNT] = {};
	Camera* camera;

public:
	Particle(Camera* cam);
	bool Init();
	void Update();
	void Draw();

	void UpdateParticles();
	void UpdateVertexBuffer();
};


