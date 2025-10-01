#pragma once
#include "BaseScene.h"
#include "FluidSystem.h"
#include "ConstantBuffer.h"
#include "VertexBuffer.h"
#include "RootSignature.h"
#include "PipelineState.h"
#include "DescriptorHeap.h"
#include "Camera.h"
#include <DirectXMath.h>
#include <Windows.h>
#include <array>
#include <memory>
#include <vector>

// ゲームメインシーン。流体シミュレーションの制御と描画を担当する。
class GameScene : public BaseScene
{
public:
    explicit GameScene(class Game* game);
    ~GameScene() override;

    bool Init() override;
    void Update(float deltaTime) override;
    void Draw() override;

private:
    struct GatherState
    {
        bool active = false;                 // 集水中かどうか
        float holdTime = 0.0f;               // ボタンを押し続けている時間
        DirectX::XMFLOAT3 hitPosition{};     // レイキャストでヒットした初期位置
        DirectX::XMFLOAT3 cameraOffset{};    // カメラからのオフセット（カメラを動かしたときに追従させる）
        float radius = 0.5f;                 // 集水半径
        float baseStrength = 12.0f;          // 集水強度の初期値
    };

    struct SplashParticle
    {
        DirectX::XMFLOAT3 position;          // 現在位置
        DirectX::XMFLOAT3 velocity;          // 移動速度
        float age = 0.0f;                    // 経過時間
        float lifetime = 0.6f;               // 寿命
        float size = 0.18f;                  // ビルボードサイズ
        float initialStrength = 1.0f;        // 生成時の強さ（色と速度の調整に使用）
    };

    struct ColorVertex
    {
        DirectX::XMFLOAT3 position;          // 頂点座標
        DirectX::XMFLOAT4 color;             // 頂点カラー
    };

    struct ColorPassCB
    {
        DirectX::XMFLOAT4X4 view;            // ビュー行列（転置済み）
        DirectX::XMFLOAT4X4 proj;            // プロジェクション行列（転置済み）
    };

    std::unique_ptr<FluidSystem> m_fluid;                     // 流体系本体
    GatherState                   m_gatherState;              // 集水操作の状態
    float                         m_deltaTime = 0.0f;         // 最新のデルタタイム
    std::vector<SplashParticle>   m_splashParticles;          // 水しぶき用パーティクル
    std::vector<ColorVertex>      m_stageVertices;            // 簡易ステージの頂点

    std::unique_ptr<VertexBuffer> m_stageVB;                  // ステージ描画用頂点バッファ
    UINT                          m_stageVertexCount = 0;     // ステージ頂点数

    std::unique_ptr<VertexBuffer> m_splashVB;                 // 水しぶき用頂点バッファ
    UINT                          m_splashVertexCount = 0;    // 現在アップロードされている水しぶき頂点数
    size_t                        m_splashVertexCapacity = 0; // バッファが保持できる頂点数

    std::unique_ptr<RootSignature> m_colorRootSignature;      // カラー表示用ルートシグネチャ
    std::unique_ptr<PipelineState> m_stagePipeline;           // ステージ描画用PSO
    std::unique_ptr<PipelineState> m_splashPipeline;          // 水しぶき描画用PSO
    std::array<std::unique_ptr<ConstantBuffer>, Engine::FRAME_BUFFER_COUNT> m_colorCB; // ビュー射影行列

    bool  m_leftButtonDown = false;                           // 左クリック押下状態

    void InitializeStageGeometry();
    void UpdateCameraConstantBuffer(const Camera* camera);
    void UpdateGatherOperation(Camera* camera);
    void SpawnSplashFromCollisions();
    void UpdateSplashParticles(float deltaTime, const Camera* camera);
    void UploadSplashVertices(const Camera* camera);
    void RenderStage(ID3D12GraphicsCommandList* cmd);
    void RenderSplash(ID3D12GraphicsCommandList* cmd);
    bool ScreenPointToRay(int mouseX, int mouseY, DirectX::XMFLOAT3& outOrigin, DirectX::XMFLOAT3& outDirection) const;
    bool BeginGather(Camera* camera);
    void ReleaseGather(Camera* camera);
};

