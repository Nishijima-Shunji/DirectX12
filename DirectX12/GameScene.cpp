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

        // 流体システムの初期化
        auto device = g_Engine->Device();
        const DXGI_FORMAT rtvFormat = DXGI_FORMAT_R8G8B8A8_UNORM;

        // 描画面積全体を満たす程度の粒子数を確保しておく。
        // 粒子が多いほど水塊らしい振る舞いになるが、あまり増やし過ぎるとGPU負荷が増大するため
        // デフォルトでは 512 粒子でバランスを取っている。
        m_maxParticles = 512;
        m_fluid.Init(device, rtvFormat, m_maxParticles, 0);
        m_fluid.SetWaterAppearance(
                XMFLOAT3(0.32f, 0.7f, 0.95f),  // 浅瀬カラー
                XMFLOAT3(0.04f, 0.18f, 0.32f), // 深場カラー
                0.28f,                         // 吸収係数
                0.42f,                         // 泡しきい値
                0.55f,                         // 泡強度
                0.65f,                         // 反射割合
                96.0f);                        // ハイライトの鋭さ
        m_fluid.SpawnParticlesSphere(XMFLOAT3(0.0f, 0.6f, 0.0f), 1.1f, m_maxParticles);
        m_fluid.UseGPU(true);


	// 描画確認用のキューブを生成
	Spawn<DebugCube>();

	return true;
}

void GameScene::Update(float deltaTime) {
        g_Engine->GetObj<Camera>("Camera")->Update(deltaTime);
        auto cmd = g_Engine->CommandList();

        // 毎フレームキー入力に応じて外力操作を設定し直す。
        // Gather（Gキー）：水塊を中央へ引き寄せる。
        // Splash（Fキー）：水面を押し広げる。
        m_fluid.ClearDynamicOperations();
        if (GetAsyncKeyState('G') & 0x8000) {
                m_fluid.QueueGather(XMFLOAT3(0.0f, 0.6f, 0.0f), 0.9f, 12.0f);
        }
        if (GetAsyncKeyState('F') & 0x8000) {
                m_fluid.QueueSplash(XMFLOAT3(0.0f, 0.6f, 0.0f), 0.8f, 6.0f);
        }

        // 登録した操作を考慮して流体を1ステップ進める。
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