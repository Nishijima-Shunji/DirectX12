#include "Game.h"
#include "SceneManager.h"
#include "GameScene.h"
#include <DirectXMath.h>
#include "SharedStruct.h"
#include "App.h"
#include "TestCube.h" // テスト用立方体

#include <windows.h>

using namespace DirectX;

GameScene* GameScene::g_pCurrentScene = nullptr;

GameScene::GameScene(Game* game) : BaseScene(game) {
	if (g_pCurrentScene == this) {
		g_pCurrentScene = nullptr;
	}
	printf("GameScene Create\n");
	Init();
}

GameScene::~GameScene() {
	printf("GameScene Release\n");
	m_objects.clear();
	if (g_pCurrentScene == this) {
		g_pCurrentScene = nullptr;
	}
}

bool GameScene::Init() {
	g_pCurrentScene = this;

	Camera* camera = new Camera();
	camera->Init();
	g_Engine->RegisterObj<Camera>("Camera", camera);

	//particle = std::make_unique<Particle>(camera);

	// 流体システムの初期化
	auto device = g_Engine->Device();
	const DXGI_FORMAT rtvFormat = DXGI_FORMAT_R8G8B8A8_UNORM;

        const UINT maxParticles = 50;
        m_fluid.Init(device, rtvFormat, maxParticles, 0);
        m_fluid.SetSpatialCellSize(0.1f); // 計算範囲
<<<<<<< HEAD
        m_fluid.UseGPU(true); // GPU でシミュレーションをするかどうか

        // テスト用の立方体をシーンに追加
        Spawn<TestCube>();
=======

        m_fluid.UseGPU(false); // GPU でシミュレーションを行かどうか
>>>>>>> parent of 18b5a9b (一旦上げ)


	return true;
}

void GameScene::Update(float deltaTime) {
        g_Engine->GetObj<Camera>("Camera")->Update(deltaTime);
        //particle->Update(deltaTime);
        auto cmd = g_Engine->CommandList();
        m_fluid.Simulate(cmd, deltaTime);

        // マウスによる粒子ドラッグ
        static bool dragging = false;
        POINT pt; GetCursorPos(&pt); ScreenToClient(g_hWnd, &pt);
        auto cam = g_Engine->GetObj<Camera>("Camera");
        if (GetAsyncKeyState(VK_LBUTTON) & 0x8000) {
                if (!dragging) {
                        m_fluid.StartDrag(pt.x, pt.y, cam);
                        dragging = true;
                }
                else {
                        m_fluid.Drag(pt.x, pt.y, cam);
                }
        }
        else {
                if (dragging) {
                        m_fluid.EndDrag();
                        dragging = false;
                }
        }

	// 全体のupdate
	for (auto& obj : m_objects) {
		if (obj->IsAlive)
			obj->Update(deltaTime);
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

	if (GetAsyncKeyState('L')) {
		m_game->ChangeScene("Scene");
	}
}

void GameScene::Draw() {
        commandList = g_Engine->CommandList(); // コマンドリストを取得
        auto cmd = g_Engine->CommandList();
        auto invViewProj = g_Engine->GetObj<Camera>("Camera")->GetInvViewProj();
        auto cameraPos = g_Engine->GetObj<Camera>("Camera")->GetPosition();
        // 画面効果用の定数バッファを更新
        m_fluid.UpdateSSAConstantBuffers(cmd);
        // 流体を描画
        m_fluid.Render(cmd, invViewProj, cameraPos, 1.0f);

	//particle->Draw();

	for (auto& actor : m_objects) {
		if (actor->IsAlive)
			actor->Render(commandList);
	}
}