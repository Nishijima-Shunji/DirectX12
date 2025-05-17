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
	std::unique_ptr<Particle> particle;
	Camera* cam;

public:
	GameScene(Game* game);
	~GameScene();
	bool Init() override; 
	void Update() override;
	void Draw() override;
};
