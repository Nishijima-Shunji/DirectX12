#pragma once
#include "BaseScene.h"
#include <DirectXMath.h>
#include "Camera.h"

class Scene : public BaseScene
{
private:
	Camera* camera = nullptr;


public:
	Scene(Game* game);
	bool Init();	// ‰Šú‰»
	void Update(float deltaTime);	// XVˆ—
	void Draw();	// •`‰æˆ—
};

extern Scene* g_Scene;

