#pragma once
#include "BaseScene.h"
#include <DirectXMath.h>
#include <vector>
#include <d3dx12.h>
#include "Camera.h"

#include "Particle.h"

class GameScene : public BaseScene
{
private:
	ConstantBuffer* constantBuffer[Engine::FRAME_BUFFER_COUNT];

	std::unique_ptr<Particle> particle;
public:
	GameScene(Game* game);
	~GameScene();
	bool Init() override;
	void Update() override;
	void Draw() override;
};
