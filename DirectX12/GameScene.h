#pragma once
#include "BaseScene.h"
#include <DirectXMath.h>
#include <vector>
#include <d3dx12.h>
#include "IActor.h"      // IActor �̒�`

#include "Camera.h"
#include "Particle.h"

class GameScene : public BaseScene
{
private:
	ConstantBuffer* constantBuffer[Engine::FRAME_BUFFER_COUNT];
    ID3D12GraphicsCommandList* commandList = nullptr;

    std::unique_ptr<Particle> particle;

    std::vector<std::unique_ptr<IActor>> m_objects;
    //Generator                          m_generator;

    // ���݂̃V�[�����w���O���[�o���|�C���^
    static GameScene* g_pCurrentScene;
public:
	GameScene(Game* game);
	~GameScene();
	bool Init() override;
	void Update() override;
	void Draw() override;

	// �V�[�����ɐ������邽�߂̐ÓI���\�b�h
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
	// �V�[�����ł̐����Ɣj�����������郁�\�b�h
    template<typename T, typename... Args>
    void SpawnImpl(Args&&... args) {
        m_actors.push_back(
            std::make_unique<T>(std::forward<Args>(args)...)
        );
    }
    void DestroyImpl(IActor* object) {
        object->IsAlive = false;
    }
};
