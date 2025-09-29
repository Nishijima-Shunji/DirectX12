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

	const UINT maxParticles = 5;
	m_fluid.Init(device, rtvFormat, maxParticles, 0);
	m_fluid.SpawnParticlesSphere(XMFLOAT3(0.0f,0.0f,0.0f),10.0f,50);
	m_fluid.UseGPU(true);
        m_fluidRenderer = std::make_unique<FluidWaterRenderer>();
        m_fluidRenderer->Init(device, g_Engine->FrameBufferWidth(), g_Engine->FrameBufferHeight());
        m_ssrRenderer = std::make_unique<SSRRenderer>();
        m_ssrRenderer->Init(device, *m_fluidRenderer, g_Engine->FrameBufferWidth(), g_Engine->FrameBufferHeight());


	// 描画確認用のキューブを生成
	Spawn<DebugCube>();

	return true;
}

void GameScene::Update(float deltaTime) {
	g_Engine->GetObj<Camera>("Camera")->Update(deltaTime);
	//particle->Update(deltaTime);
	auto cmd = g_Engine->CommandList();
	m_fluid.Simulate(cmd, deltaTime);

        if (m_fluidRenderer) {
                if (GetAsyncKeyState(VK_F1) & 0x0001) { m_currentDebug = FluidDebugView::Depth; }
                if (GetAsyncKeyState(VK_F2) & 0x0001) { m_currentDebug = FluidDebugView::Thickness; }
                if (GetAsyncKeyState(VK_F3) & 0x0001) { m_currentDebug = FluidDebugView::Normal; }
                if (GetAsyncKeyState(VK_F4) & 0x0001) { m_currentDebug = FluidDebugView::Composite; }
                if (GetAsyncKeyState(VK_F5) & 0x0001) { m_ssrQuality = (m_ssrQuality % 3) + 1; }
                if (GetAsyncKeyState(VK_F6) & 0x0001) { m_usePlanarReflection = !m_usePlanarReflection; }
                if (GetAsyncKeyState(VK_F7) & 0x0001) { m_downsampleStep = (m_downsampleStep == 4) ? 1 : (m_downsampleStep == 2 ? 4 : 2); }
                if (GetAsyncKeyState(VK_F8) & 0x0001) { m_gridDebug = !m_gridDebug; }

                m_fluidRenderer->SetDebugView(m_currentDebug);
                m_fluidRenderer->SetSSRQuality(static_cast<uint32_t>(m_ssrQuality));
                m_fluidRenderer->TogglePlanarReflection(m_usePlanarReflection);
                m_fluidRenderer->EnableGridDebug(m_gridDebug);
                m_fluidRenderer->SetDownsample(m_downsampleStep);
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
        commandList = g_Engine->CommandList();
        auto cmd = g_Engine->CommandList();
        auto camObj = g_Engine->GetObj<Camera>("Camera");
        if (!camObj) {
                return;
        }

        if (m_fluidRenderer) {
                m_fluidRenderer->BeginSceneRender(cmd);
        }

        for (auto& actor : m_objects) {
                if (actor->IsAlive) {
                        actor->Render(cmd);
                }
        }

        if (m_fluidRenderer) {
                m_fluidRenderer->EndSceneRender(cmd);
                m_fluidRenderer->RenderDepthThickness(cmd, *camObj, m_fluid);
                m_fluidRenderer->BlurAndNormal(cmd);
                if (m_ssrRenderer) {
                        m_ssrRenderer->RenderSSR(cmd, *m_fluidRenderer, *camObj);
                }
                m_fluidRenderer->Composite(cmd);
        }
}
