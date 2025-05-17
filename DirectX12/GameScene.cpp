#include "Game.h"
#include "SceneManager.h"
#include "GameScene.h"

#include <windows.h>

GameScene::GameScene(Game* game) : BaseScene(game) {
	printf("GameScene¶¬\n");
	Init();
}

GameScene::~GameScene() {
	printf("GameScene”jŠü\n");
}

bool GameScene::Init() {
	Camera* cam = m_game->GetCamera();
	particle = std::make_unique<Particle>(cam);

	return true;
}

void GameScene::Update() {
	particle->Update();

	if (GetAsyncKeyState(VK_UP)) {
		m_game->ChangeScene("Scene");
	}
}

void GameScene::Draw() {
	particle->Draw();
}