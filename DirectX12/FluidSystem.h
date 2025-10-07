#pragma once
#include "ConstantBuffer.h"
#include "Engine.h"
#include "IndexBuffer.h"
#include "PipelineState.h"
#include "RootSignature.h"
#include "SharedStruct.h"
#include "SpatialGrid.h"
#include "VertexBuffer.h"
#include <DirectXMath.h>
#include <array>
#include <memory>
#include <vector>

class Camera;

// 流体システム。粒子の管理と壁との当たり判定、描画を担当する。
class FluidSystem
{
public:
    struct Bounds
    {
        DirectX::XMFLOAT3 min; // AABB最小点
        DirectX::XMFLOAT3 max; // AABB最大点
    };

    FluidSystem();
    ~FluidSystem() = default;

    bool Init(ID3D12Device* device, const Bounds& initialBounds, UINT particleCount);
    void Update(float deltaTime);
    void Draw(ID3D12GraphicsCommandList* cmd, const Camera& camera);

    void SetBounds(const Bounds& bounds);
    void AdjustWall(const DirectX::XMFLOAT3& direction, float amount); // 左クリック操作に合わせて壁を押し引きする
    const Bounds& GetBounds() const { return m_bounds; }

private:
    struct Particle
    {
        DirectX::XMFLOAT3 position; // 現在位置
        DirectX::XMFLOAT3 velocity; // 速度ベクトル
    };

    struct FluidConstant
    {
        DirectX::XMMATRIX World; // ワールド行列
        DirectX::XMMATRIX View;  // ビュー行列
        DirectX::XMMATRIX Proj;  // プロジェクション行列
    };

    void InitializeParticles(UINT particleCount);
    void ResolveCollisions(Particle& particle) const;
    void UpdateInstanceBuffer();
    void UpdateGridLines();

    struct SphereVertex
    {
        DirectX::XMFLOAT3 position; // 球メッシュの頂点位置
        DirectX::XMFLOAT3 normal;   // 球メッシュの法線
    };

    struct ParticleInstance
    {
        DirectX::XMFLOAT3 position; // 粒子中心位置
        float radius;               // 描画半径
    };

    struct GridLineVertex
    {
        DirectX::XMFLOAT3 position; // グリッド境界線の頂点位置
    };

    std::vector<Particle> m_particles;               // 流体粒子
    std::vector<ParticleInstance> m_instances;       // インスタンシング用データ
    std::unique_ptr<VertexBuffer> m_instanceBuffer;  // 粒子描画用インスタンスVB
    std::unique_ptr<VertexBuffer> m_sphereVertexBuffer; // 球メッシュ頂点VB
    std::unique_ptr<IndexBuffer> m_sphereIndexBuffer;   // 球メッシュIB
    UINT m_indexCount = 0;                              // 球メッシュのインデックス数
    std::unique_ptr<RootSignature> m_rootSignature;  // 粒子描画用ルートシグネチャ
    std::unique_ptr<PipelineState> m_pipelineState;  // 粒子描画用PSO
    std::unique_ptr<PipelineState> m_gridPipelineState; // グリッド描画用PSO
    std::array<std::unique_ptr<ConstantBuffer>, Engine::FRAME_BUFFER_COUNT> m_constantBuffers; // 定数バッファ

    std::vector<GridLineVertex> m_gridLineVertices;  // グリッド線分頂点
    std::unique_ptr<VertexBuffer> m_gridLineBuffer;   // グリッド線分用VB
    size_t m_gridLineCapacity = 0;                    // グリッド線分VBの収容数
    UINT m_gridLineVertexCount = 0;                   // 描画に使用する頂点数

    Bounds m_bounds{};             // 現在の境界
    DirectX::XMMATRIX m_world;     // 粒子全体のワールド行列
    float m_particleRadius = 0.05f; // 粒子半径
    float m_restitution = 0.4f;     // 壁衝突時の反発係数
    float m_drag = 0.1f;            // 簡易減衰係数
    float m_supportRadius = 0.18f;  // 粒子同士の相互作用半径
    float m_interactionStrength = 8.0f; // 粒子押し戻し力の係数
    float m_maxVelocity = 6.0f;     // 速度クランプ値

    SpatialGrid m_grid;                                     // 近傍探索用グリッド
    std::vector<size_t> m_neighborIndices;                 // 近傍粒子インデックス一時バッファ
    std::vector<DirectX::XMFLOAT3> m_neighborForces;       // 粒子間相互作用の蓄積結果
};
