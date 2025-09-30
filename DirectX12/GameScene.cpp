#include "Game.h"
#include "SceneManager.h"
#include "GameScene.h"
#include <DirectXMath.h>
#include "SharedStruct.h"
#include "App.h"
#include "DebugCube.h" // デバッグ用キューブ

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

        const UINT maxParticles = 5; // 初期負荷を抑えて安定動作させるため控えめに設定
        m_fluid.Init(device, rtvFormat, maxParticles, 0);
        m_fluid.SetWaterAppearance(
                XMFLOAT3(0.32f, 0.7f, 0.95f),  // 浅瀬カラー
                XMFLOAT3(0.04f, 0.18f, 0.32f), // 深場カラー
                0.28f,                         // 吸収係数
                0.42f,                         // 泡しきい値
                0.55f,                         // 泡強度
                0.65f,                         // 反射割合
                96.0f);                        // ハイライトの鋭さ
        m_fluid.SpawnParticlesSphere(XMFLOAT3(0.0f, 0.6f, 0.0f), 1.1f, maxParticles);
        m_fluid.UseGPU(true);


	// 描画確認用のキューブを生成
	Spawn<DebugCube>();

	return true;
}

void GameScene::Update(float deltaTime) {
	g_Engine->GetObj<Camera>("Camera")->Update(deltaTime);
	//particle->Update(deltaTime);
	auto cmd = g_Engine->CommandList();
	m_fluid.Simulate(cmd, deltaTime);

	m_lastDeltaTime = deltaTime;          // 合成パス用に直近のデルタタイムを保存
	++m_frameCounter;                    // フレーム番号をインクリメント

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
	auto camObj = g_Engine->GetObj<Camera>("Camera");
        auto invViewProj = camObj->GetInvViewProj();
        XMMATRIX viewMat = camObj->GetViewMatrix();
        XMMATRIX projMat = camObj->GetProjMatrix();
        XMMATRIX viewProjMat = XMMatrixMultiply(viewMat, projMat);
        XMFLOAT4X4 viewProj;
        XMStoreFloat4x4(&viewProj, viewProjMat);
        XMFLOAT4X4 view;
        XMStoreFloat4x4(&view, viewMat);
        XMFLOAT4X4 proj;
        XMStoreFloat4x4(&proj, projMat);
        auto cameraPos = camObj->GetPosition();

	//particle->Draw();

	m_fluid.Render(cmd, invViewProj, viewProj, cameraPos, 1.0f); // まず流体用の中間テクスチャを構築

	for (auto& actor : m_objects) {
		if (actor->IsAlive)
			actor->Render(commandList);
	}

	auto mainRtv = g_Engine->CurrentBackBufferView();
	auto mainDsv = g_Engine->DepthStencilView();
	m_fluid.Composite(
		cmd,
		mainRtv,
		mainDsv,
		m_fluid.DepthTextureHandle(),
		m_fluid.NormalTextureHandle(),
		m_fluid.ColorTextureHandle(),
		view,
		proj,
		cameraPos,
		m_lastDeltaTime,
		static_cast<UINT>(m_frameCounter)
	);
}
