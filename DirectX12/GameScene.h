#pragma once
#include "BaseScene.h"
#include <DirectXMath.h>
#include <vector>
#include <d3dx12.h>
#include "IActor.h"      // IActor ̒`

#include "Camera.h"
#include "Particle.h"
#include "FluidSystem.h"

#include "FluidWaterRenderer.h"
#include "SSRRenderer.h"
class GameScene : public BaseScene
{
private:
	ConstantBuffer* constantBuffer[Engine::FRAME_BUFFER_COUNT];
    ID3D12GraphicsCommandList* commandList = nullptr;

    // ===== IuWFNg =====
    FluidSystem  m_fluid;
    std::unique_ptr<Particle> particle;
    std::unique_ptr<FluidWaterRenderer> m_fluidRenderer;
    std::unique_ptr<SSRRenderer> m_ssrRenderer;
    FluidDebugView m_currentDebug = FluidDebugView::Composite;
    int m_ssrQuality = 2;
    bool m_usePlanarReflection = false;
    bool m_gridDebug = false;
    uint32_t m_downsampleStep = 1;
    std::vector<std::unique_ptr<IActor>> m_objects;
    //Generator                          m_generator;

    // ݂̃V[wO[o|C^
    static GameScene* g_pCurrentScene;
public:
	GameScene(Game* game);
	~GameScene();
	bool Init() override;
	void Update(float deltaTime) override;
	void Draw() override;

	// V[ɐ邽߂̐ÓI\bh
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
	// V[ł̐Ɣj郁\bh
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
