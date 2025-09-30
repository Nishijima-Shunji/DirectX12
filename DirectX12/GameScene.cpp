#include "Game.h"
#include "SceneManager.h"
#include "GameScene.h"
#include "Engine.h" // エンジンの各種リソースへアクセスするためのヘッダーを追加
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
        XMMATRIX viewMat = camObj->GetViewMatrix();
        XMMATRIX projMat = camObj->GetProjMatrix();
        XMMATRIX viewProjMat = XMMatrixMultiply(viewMat, projMat);
        XMFLOAT4X4 viewMatrixFloat;
        XMFLOAT4X4 projMatrixFloat;
        XMFLOAT4X4 viewProjFloat;
        XMStoreFloat4x4(&viewMatrixFloat, viewMat);
        XMStoreFloat4x4(&projMatrixFloat, projMat);
        XMStoreFloat4x4(&viewProjFloat, viewProjMat);
        auto cameraPos = camObj->GetPosition();

        // 1. 流体の中間テクスチャを更新（粒子深度・平滑化・法線など）
        // ピクセルシェーダーで参照する行列やカメラ位置を流体レンダリングへ渡す
        m_fluid.Render(cmd, viewMatrixFloat, projMatrixFloat, viewProjFloat, cameraPos, 1.0f);

        // 2. シーンジオメトリを通常のRTV/DSVへ描画
        for (auto& actor : m_objects) {
                if (actor->IsAlive)
                        actor->Render(commandList);
        }

        // 3. シーンカラー／深度と流体情報を合成して最終カラーを出力
        // レンダーターゲットと深度バッファを用いて流体の最終合成を実行
        m_fluid.Composite(
                cmd,
                g_Engine->CurrentRenderTargetResource(),
                g_Engine->DepthStencilBuffer(),
                g_Engine->CurrentBackBufferView());
}