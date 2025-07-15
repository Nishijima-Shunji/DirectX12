#include "Game.h"
#include "SceneManager.h"
#include "GameScene.h"
#include <DirectXMath.h>
#include "SharedStruct.h"
#include "App.h"

#include <windows.h>

using namespace DirectX;

GameScene::GameScene(Game* game) : BaseScene(game) {
	printf("GameScene����\n");
	Init();
}

GameScene::~GameScene() {
	printf("GameScene�j��\n");
	m_objects.clear();
}

bool GameScene::Init() {
	g_pCurrentScene = this;

	Camera* camera = new Camera();
	camera->Init();
	g_Engine->RegisterObj<Camera>("Camera", camera);

	particle = std::make_unique<Particle>(camera);
	
	// ���̃V�X�e���̏�����
	auto device = g_Engine->Device();
	auto rtvFormat = g_Engine->GetSwapChainFormat();

	const UINT maxParticles = 1024;
	const UINT threadGroupCount = 16;
	m_fluid.Init(device, rtvFormat, maxParticles, threadGroupCount);

	m_fluid.UseGPU(true); // GPU �ŃV�~�����[�V�������s���ǂ���


	return true;
}

void GameScene::Update() {
	g_Engine->GetObj<Camera>("Camera")->Update();
	particle->Update();

	// �S�̂�update
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
	commandList = g_Engine->CommandList(); // �R�}���h���X�g���擾


	particle->Draw();

	for (auto& actor : m_objects) {
		if (actor->IsAlive)
			actor->Render(commandList);
	}
}