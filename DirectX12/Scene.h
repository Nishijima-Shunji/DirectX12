#pragma once
#include "BaseScene.h"
#include <DirectXMath.h>

class Scene : public BaseScene
{
private:
	float rotateY = 0.0f;
	float posX = 0.0f;
	float posY = 0.0f;

	DirectX::XMVECTOR eyePos;
	DirectX::XMVECTOR targetPos;
	DirectX::XMVECTOR upward;
	float fov;


public:
	Scene(Game* game);
	bool Init();	// ‰Šú‰»
	void Update();	// XVˆ—
	void Draw();	// •`‰æˆ—
};

extern Scene* g_Scene;

