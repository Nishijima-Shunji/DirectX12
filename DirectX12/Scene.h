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
	bool Init();	// ������
	void Update(float deltaTime);	// �X�V����
	void Draw();	// �`�揈��
};

extern Scene* g_Scene;

