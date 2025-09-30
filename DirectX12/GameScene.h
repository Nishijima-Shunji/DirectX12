#pragma once
#include "BaseScene.h"
#include <DirectXMath.h>
#include <vector>
#include <d3dx12.h>
#include <wrl.h>
#include <memory>
#include "IActor.h"      // IActor の定義
#include "Engine.h"
#include "Camera.h"
#include "SharedStruct.h" // 粒子情報共有用構造体
#include "ConstantBuffer.h"
#include "DescriptorHeap.h"
#include "MetaBallPipelineState.h"

class GameScene : public BaseScene
{
private:
    // ===== オブジェクト =====
    std::vector<std::unique_ptr<IActor>> m_objects;

    // 現在のシーンを指すグローバルポインタ
    static GameScene* g_pCurrentScene;

    // ===== メタボール描画用のCPU保持データ =====
    std::vector<ParticleMeta> m_particles; // 粒子データ（CPU側）
    std::vector<DirectX::XMFLOAT3> m_velocities; // 粒子ごとの速度ベクトル

    // ===== メタボール描画用定数バッファのCPUミラー =====
    struct MetaCB_CPU {
        DirectX::XMFLOAT4X4 invViewProj;
        DirectX::XMFLOAT4X4 viewProj;
        DirectX::XMFLOAT4 camRadius;   // xyz: カメラ位置, w: 粒子半径
        DirectX::XMFLOAT4 isoCount;    // x: iso値, y: 粒子数, z: ステップ長, w: 未使用
        DirectX::XMFLOAT4 gridMinCell; // 未使用（w=0 で無効）
        UINT gridDimX, gridDimY, gridDimZ, totalCells; // 未使用
        DirectX::XMFLOAT4 waterDeep;    // 深い水の色＋吸収係数
        DirectX::XMFLOAT4 waterShallow; // 浅い水の色＋泡閾値
        DirectX::XMFLOAT4 shadingParams;// 泡・反射率・スペキュラ指数・時間
    } m_metaCB{};                       // CPU側の定数バッファミラー

    float m_radius = 0.12f; // 粒子の半径
    float m_iso = 0.18f;    // メタボール等値面の閾値
    float m_step = 2.5f;    // レイマーチのステップスケール
    float m_time = 0.0f;    // 経過時間

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_metaRootSignature; // メタボール用ルートシグネチャ
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_metaPipelineState; // メタボール用PSO
    Microsoft::WRL::ComPtr<ID3D12Resource>      m_particleBuffer;    // 粒子のStructuredBuffer
    ParticleMeta*                               m_particleMapped = nullptr; // GPUバッファへのマップ先
    DescriptorHandle*                           m_particleSRV = nullptr;    // 粒子バッファのSRV
    std::unique_ptr<ConstantBuffer>             m_metaConstantBuffer;       // 定数バッファ
    UINT                                        m_particleCapacity = 0;     // 確保済み粒子数

    float m_fpsTimer = 0.0f;  // FPS更新用のタイマー
    int   m_fpsFrameCount = 0; // 計測中のフレーム数

    bool CreateMetaPipeline();            // メタボール描画用PSOの生成
    bool CreateParticleBuffer(UINT count); // 粒子StructuredBufferの生成
    void UpdateParticleBufferGPU();        // 粒子情報をGPUへ書き戻す
    void WriteMetaCB();                    // 定数バッファの書き込み
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
