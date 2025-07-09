#include "Game.h"
#include "SceneManager.h"
#include "GameScene.h"
#include <DirectXMath.h>
#include "SharedStruct.h"
#include "App.h"

#include <windows.h>

using namespace DirectX;

GameScene::GameScene(Game* game) : BaseScene(game) {
	printf("GameScene¶¬\n");
	Init();
}

GameScene::~GameScene() {
	printf("GameScene”jŠü\n");
}

bool GameScene::Init() {
	Camera* camera = new Camera();
	camera->Init();
	g_Engine->RegisterObj<Camera>("Camera", camera);
	particle = std::make_unique<Particle>(camera);
	
	return true;
}

void GameScene::Update() {
	g_Engine->GetObj<Camera>("Camera")->Update();
	particle->Update();

	if (GetAsyncKeyState('Q')) {
		m_game->ChangeScene("Scene");
	}
}

void GameScene::Draw() {
	particle->Draw();
}