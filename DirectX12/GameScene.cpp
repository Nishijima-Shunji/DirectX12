#include "Game.h"
#include "SceneManager.h"
#include "GameScene.h"
#include <DirectXMath.h>
#include "SharedStruct.h"
#include "App.h"

#include <windows.h>

using namespace DirectX;

GameScene::GameScene(Game* game) : BaseScene(game) {
	printf("GameScene生成\n");
	Init();
}

GameScene::~GameScene() {
	printf("GameScene破棄\n");
	m_objects.clear();
}

bool GameScene::Init() {
	g_pCurrentScene = this;

	Camera* camera = new Camera();
	camera->Init();
	g_Engine->RegisterObj<Camera>("Camera", camera);

	particle = std::make_unique<Particle>(camera);
	
	// 流体システムの初期化
	auto device = g_Engine->Device();
	auto rtvFormat = g_Engine->GetSwapChainFormat();

	const UINT maxParticles = 1024;
	const UINT threadGroupCount = 16;
	m_fluid.Init(device, rtvFormat, maxParticles, threadGroupCount);

	m_fluid.UseGPU(true); // GPU でシミュレーションを行かどうか


	return true;
}

void GameScene::Update() {
	g_Engine->GetObj<Camera>("Camera")->Update();
	particle->Update();

	// 全体のupdate
	for (auto& actor : m_objects) {
		if (actor->IsAlive)
			actor->Update(/*dt*/);
	}

	m_objects.erase(
		std::remove_if(
			m_objects.begin(), m_objects.end(),
			[](const std::unique_ptr<IActor>& uptr) {
				return !uptr->IsAlive;
			}
		),
		m_objects.end()
	);

	if (GetAsyncKeyState('Q')) {
		m_game->ChangeScene("Scene");
	}
}

void GameScene::Draw() {
	commandList = g_Engine->CommandList(); // コマンドリストを取得


	particle->Draw();

	for (auto& actor : m_objects) {
		if (actor->IsAlive)
			actor->Render(commandList);
	}
}