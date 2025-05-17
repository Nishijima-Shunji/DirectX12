#pragma once
#include "BaseScene.h"
#include <DirectXMath.h>
#include <vector>
#include <d3dx12.h>
#include "Engine.h"
#include "VertexBuffer.h"
#include "RootSignature.h"
#include "PipelineState.h"
#include "ConstantBuffer.h"

struct Particle {
	DirectX::XMFLOAT3 position;
	DirectX::XMFLOAT3 velocity;
};

class GameScene : public BaseScene
{
private:
	float rotateY = 0.0f;
	float posX = 0.0f;
	float posY = 0.0f;

	DirectX::XMVECTOR eyePos;
	DirectX::XMVECTOR targetPos;
	DirectX::XMVECTOR upward;
	float fov;

	std::vector<Particle> m_Particles;
	VertexBuffer* m_VertexBuffer = nullptr;
	RootSignature* m_RootSignature = nullptr;
	PipelineState* m_PipelineState = nullptr;
	ConstantBuffer* m_ConstantBuffer[Engine::FRAME_BUFFER_COUNT] = {};


	void InitParticles();
	void UpdateParticles();
	void DrawParticles();
	void UpdateVertexBuffer();

public:
	GameScene(Game* game);
	~GameScene();
	bool Init() override; 
	void Update() override;
	void Draw() override;
};
