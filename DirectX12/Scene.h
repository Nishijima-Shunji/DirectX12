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
	bool Init();	// 初期化
	void Update(float deltaTime);	// 更新処理
	void Draw();	// 描画処理
};

extern Scene* g_Scene;

