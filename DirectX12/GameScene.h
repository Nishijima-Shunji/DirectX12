#pragma once
#include "BaseScene.h"
#include <DirectXMath.h>
#include <vector>
#include <d3dx12.h>
#include <cstdint>
    uint64_t m_frameCounter = 0;          // t[ԍ̒~
    float    m_lastDeltaTime = 0.0f;      // ߃t[̌oߎ
#include "IActor.h"      // IActor の定義

#include "Camera.h"
#include "Particle.h"
#include "FluidSystem.h"

class GameScene : public BaseScene
{
private:
	ConstantBuffer* constantBuffer[Engine::FRAME_BUFFER_COUNT];
    ID3D12GraphicsCommandList* commandList = nullptr;

    // ===== オブジェクト =====
    FluidSystem  m_fluid;
    std::unique_ptr<Particle> particle;
    std::vector<std::unique_ptr<IActor>> m_objects;
    //Generator                          m_generator;

    // 現在のシーンを指すグローバルポインタ
    static GameScene* g_pCurrentScene;
public:
	GameScene(Game* game);
	~GameScene();
	bool Init() override;
	void Update(float deltaTime) override;
	void Draw() override;

	// シーン内に生成するための静的メソッド
    template<typename T, typename... Args>
    static void Spawn(Args&&... args) {
        if (g_pCurrentScene) {
            g_pCurrentScene->SpawnImpl<T>(std::forward<Args>(args)...);
        }
    }
    static void Destroy(IActor* object) {
        if (g_pCurrentScene) {
            g_pCurrentScene->DestroyImpl(object);
        }
    }

private:
	// シーン内での生成と破棄を実装するメソッド
    template<typename T, typename... Args>
    void SpawnImpl(Args&&... args) {
        m_objects.push_back(
            std::make_unique<T>(std::forward<Args>(args)...)
        );
    }
    void DestroyImpl(IActor* object) {
        object->IsAlive = false;
    }
};
