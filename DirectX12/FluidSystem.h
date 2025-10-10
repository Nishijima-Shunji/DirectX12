#pragma once
#include "ComputePipelineState.h"
#include "ConstantBuffer.h"
#include "Engine.h"
#include "IndexBuffer.h"
#include "PipelineState.h"
#include "RootSignature.h"
#include "SharedStruct.h"
#include "VertexBuffer.h"
#include <DirectXMath.h>
#include <cstdint>
#include <array>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include <wrl.h>

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

    enum class RenderMode
    {
        SSFR,            // スクリーンスペース流体描画
        MarchingCubes,   // マーチングキューブによるサーフェス生成
    };

    bool Init(ID3D12Device* device, const Bounds& initialBounds, UINT particleCount, RenderMode renderMode = RenderMode::SSFR);
    void Update(float deltaTime);
    void Draw(ID3D12GraphicsCommandList* cmd, const Camera& camera);

    void SetBounds(const Bounds& bounds);
    void AdjustWall(const DirectX::XMFLOAT3& direction, float amount); // 左クリック操作に合わせて壁を押し引きする
    const Bounds& GetBounds() const { return m_bounds; }

private:
    struct Particle
    {
        DirectX::XMFLOAT3 position;  // 現在位置
        DirectX::XMFLOAT3 velocity;  // 速度ベクトル
        DirectX::XMFLOAT3 predicted; // 位置予測（PBF の拘束解決に使用）
        float lambda = 0.0f;         // ラグランジュ乗数
        float density = 0.0f;        // 粒子密度
    };

    struct FluidConstant
    {
        DirectX::XMMATRIX World; // ワールド行列
        DirectX::XMMATRIX View;  // ビュー行列
        DirectX::XMMATRIX Proj;  // プロジェクション行列
        DirectX::XMFLOAT3 CameraPos; // ピクセルシェーダーで視線方向を求めるためのカメラ位置
        float _padding = 0.0f;       // 16バイト境界を維持するためのパディング
    };

    void InitializeParticles(UINT particleCount);
    void ResolveCollisions(Particle& particle) const;
    void UpdateInstanceBuffer();
    void UpdateGridLines();
    void InitializeGrid();
    void ClearGridNodes();
    size_t GridIndex(int x, int y, int z) const;
    void EnsureGridReady();
    void ConstrainToBounds(DirectX::XMFLOAT3& position) const;
    void UpdateKernelConstants();
    float EvaluatePoly6(float distance) const;
    DirectX::XMVECTOR EvaluateSpikyGradient(const DirectX::XMVECTOR& offset, float distance) const;
    void RebuildCellTable();
    template<typename Fn>
    void ForEachNeighbor(size_t index, const Fn& fn) const
    {
        // 予測位置の周辺 3x3x3 セルを走査して近傍粒子に処理を適用する
        if (index >= m_particles.size() || m_gridSpacing <= 0.0f)
        {
            return;
        }

        const float invSpacing = 1.0f / std::max(m_gridSpacing, 1e-4f);
        const int maxX = std::max(m_gridResolution.x - 1, 0);
        const int maxY = std::max(m_gridResolution.y - 1, 0);
        const int maxZ = std::max(m_gridResolution.z - 1, 0);

        const DirectX::XMFLOAT3& pos = m_particles[index].predicted;
        const float fx = std::clamp((pos.x - m_gridOrigin.x) * invSpacing, 0.0f, static_cast<float>(maxX));
        const float fy = std::clamp((pos.y - m_gridOrigin.y) * invSpacing, 0.0f, static_cast<float>(maxY));
        const float fz = std::clamp((pos.z - m_gridOrigin.z) * invSpacing, 0.0f, static_cast<float>(maxZ));

        const int baseX = std::clamp(static_cast<int>(std::floor(fx)), 0, maxX);
        const int baseY = std::clamp(static_cast<int>(std::floor(fy)), 0, maxY);
        const int baseZ = std::clamp(static_cast<int>(std::floor(fz)), 0, maxZ);

        for (int dz = -1; dz <= 1; ++dz)
        {
            const int cellZ = std::clamp(baseZ + dz, 0, maxZ);
            for (int dy = -1; dy <= 1; ++dy)
            {
                const int cellY = std::clamp(baseY + dy, 0, maxY);
                for (int dx = -1; dx <= 1; ++dx)
                {
                    const int cellX = std::clamp(baseX + dx, 0, maxX);
                    const size_t cellIndex = GridIndex(cellX, cellY, cellZ);
                    if (cellIndex >= m_cellParticleIndices.size())
                    {
                        continue;
                    }

                    const auto& cell = m_cellParticleIndices[cellIndex];
                    for (uint32_t neighbor : cell)
                    {
                        fn(neighbor);
                    }
                }
            }
        }
    }
    void ScatterDensityToGrid();

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
    float m_supportRadius = 0.18f;  // 粒子同士の相互作用半径
    float m_maxVelocity = 6.0f;     // 速度クランプ値

    RenderMode m_renderMode = RenderMode::SSFR; // 描画モード
    bool m_useSSFR = true;          // SSFR を使用するかどうか（既定で ON にして球メッシュ描画を置き換える）

    struct MarchingVertex
    {
        DirectX::XMFLOAT3 position; // サーフェス頂点位置
        DirectX::XMFLOAT3 normal;   // サーフェス法線
    };

    std::vector<MarchingVertex> m_marchingVertices;      // マーチングキューブ頂点
    std::vector<uint32_t> m_marchingIndices;             // マーチングキューブインデックス
    std::unique_ptr<VertexBuffer> m_marchingVertexBuffer; // マーチングキューブVB
    std::unique_ptr<IndexBuffer> m_marchingIndexBuffer;   // マーチングキューブIB
    size_t m_marchingVertexCapacity = 0;                  // VB確保済みサイズ
    size_t m_marchingIndexCapacity = 0;                   // IB確保済みサイズ
    UINT m_marchingIndexCount = 0;                        // 描画インデックス数
    std::unique_ptr<PipelineState> m_marchingPipelineState; // マーチングキューブ描画PSO

    struct alignas(256) SSFRConstant
    {
        DirectX::XMMATRIX view;             // ビュー行列（スクリーンスペース再構成用）
        DirectX::XMMATRIX proj;             // プロジェクション行列
        DirectX::XMFLOAT2 screenSize;       // 流体バッファの解像度
        float nearZ;                        // カメラの近クリップ
        float farZ;                         // カメラの遠クリップ
        DirectX::XMFLOAT3 iorF0;            // Fresnel 計算に使う F0（屈折率 1.33 相当）
        float absorb;                       // Beer-Lambert の吸収係数
        DirectX::XMFLOAT2 framebufferSize;     // 合成時に使用するフル解像度（UV計算のずれ防止）
        DirectX::XMFLOAT2 bilateralSigma;      // 空間シグマと深度シグマ
        DirectX::XMFLOAT2 bilateralNormalKernel; // 法線シグマとカーネル半径
        DirectX::XMFLOAT2 _pad;                // 256byte境界を維持するための余白
    };

    struct SSFRTarget
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource; // GPU リソース
        DescriptorHandle* srvHandle = nullptr;            // SRV ディスクリプタ
        DescriptorHandle* uavHandle = nullptr;            // UAV ディスクリプタ
        D3D12_CPU_DESCRIPTOR_HANDLE uavCpuHandle{};       // Clear 用 CPU ディスクリプタ
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle{};          // RTV 用 CPU ディスクリプタ（粒子スプラットのMRT用）
        D3D12_RESOURCE_STATES currentState = D3D12_RESOURCE_STATE_COMMON; // 現在のリソース状態
    };

    std::array<std::unique_ptr<ConstantBuffer>, Engine::FRAME_BUFFER_COUNT> m_ssfrConstantBuffers; // SSFR 用定数バッファ

    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_ssfrParticlePSO; // ビルボード粒子描画 PSO
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_ssfrCompositePSO; // 合成 PSO
    std::unique_ptr<ComputePipelineState> m_ssfrBilateralPSO; // 深度平滑化 CS
    std::unique_ptr<ComputePipelineState> m_ssfrNormalPSO;    // 法線再構成 CS
    std::unique_ptr<RootSignature> m_ssfrParticleRootSig;     // 粒子描画用ルートシグネチャ
    std::unique_ptr<RootSignature> m_ssfrCompositeRootSig;    // 合成用ルートシグネチャ
    std::unique_ptr<RootSignature> m_ssfrComputeRootSig;      // コンピュート用ルートシグネチャ

    SSFRTarget m_rawDepth;        // 粒子スプラット結果（深度）
    SSFRTarget m_smoothedDepth;   // バイラテラル平滑化後の深度
    SSFRTarget m_thickness;       // 粒子厚み積算
    SSFRTarget m_normal;          // 法線バッファ

    SSFRTarget m_sceneColorCopy;  // 背景カラー退避用（SRV のみ使用）
    DescriptorHandle* m_sceneDepthSrv = nullptr; // シーン深度 SRV
    ID3D12Resource* m_cachedSceneDepth = nullptr; // 深度バッファのリソースキャッシュ（リサイズ検出用）

    DescriptorHandle* m_particleBufferSrv = nullptr; // 粒子インスタンス SRV

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_ssfrCpuDescriptorHeap; // UAV クリア用 CPU ヒープ
    UINT m_ssfrCpuDescriptorSize = 0; // CPU ヒープのインクリメント
    UINT m_ssfrCpuDescriptorCursor = 0; // 次に割り当てる CPU ディスクリプタ位置
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_ssfrRtvDescriptorHeap; // RTV 用 CPU ヒープ
    UINT m_ssfrRtvDescriptorSize = 0;  // RTV ヒープのインクリメント
    UINT m_ssfrRtvDescriptorCursor = 0; // 次に割り当てる RTV ディスクリプタ位置

    UINT m_ssfrWidth = 0;  // 半解像度幅
    UINT m_ssfrHeight = 0; // 半解像度高さ

    UINT m_descriptorIncrement = 0; // CBV/SRV/UAV ヒープのインクリメント

    void RenderSSFR(ID3D12GraphicsCommandList* cmd, const Camera& camera); // SSFR 描画処理本体
    bool EnsureSSFRResources(); // SSFR リソース初期化
    bool CreateSSFRRootSignatures(); // SSFR 用ルートシグネチャ生成
    bool CreateSSFROnce(); // 初回初期化をまとめる
    bool ResizeSSFRTargets(UINT width, UINT height); // 解像度変更対応
    void UpdateSSFRConstants(const Camera& camera); // 定数バッファ更新
    void PrepareSSFRTargets(ID3D12GraphicsCommandList* cmd); // RTV/UAV を毎フレーム準備
    void TransitionSSFRTarget(ID3D12GraphicsCommandList* cmd, SSFRTarget& target, D3D12_RESOURCE_STATES newState); // 状態遷移
    D3D12_CPU_DESCRIPTOR_HANDLE AllocateCpuDescriptor(); // CPU ヒープから UAV 用ディスクリプタを確保
    D3D12_CPU_DESCRIPTOR_HANDLE AllocateRtvDescriptor(); // RTV 用ヒープからディスクリプタを確保
    bool CreateParticlePSO(); // 粒子 PSO 生成
    bool CreateCompositePSO(); // 合成 PSO 生成
    bool CreateComputePSO(); // コンピュート PSO 生成
    void GenerateMarchingCubesMesh(); // マーチングキューブメッシュ生成
    void UpdateMarchingBuffers(); // マーチングキューブ用VB/IB更新
    float GetSmoothedNodeMass(int x, int y, int z) const; // グリッド節点の密度を平滑化して取得
    float SampleGridDensity(const DirectX::XMFLOAT3& position) const; // 格子密度サンプル
    DirectX::XMFLOAT3 SampleGridGradient(const DirectX::XMFLOAT3& position) const; // 密度勾配サンプル

    struct GridNode
    {
        DirectX::XMFLOAT3 velocity; // 節点速度（粒子の重心速度を蓄積）
        float mass;                 // 節点質量（MLS-MPM での重み合計）
        float pressure;             // 節点圧力（密度超過に応じた押し返し力）
        DirectX::XMFLOAT3 pressureGradient; // 圧力勾配（粒子へ伝搬する反力ベクトル）
    };

    DirectX::XMFLOAT3 m_gridOrigin{ 0.0f, 0.0f, 0.0f }; // グリッド原点
    DirectX::XMINT3 m_gridResolution{ 1, 1, 1 };        // グリッド解像度
    float m_gridSpacing = 0.1f;                         // グリッドセル間隔
    std::vector<GridNode> m_gridNodes;                  // MLS-MPM 用の節点データ
    std::vector<std::vector<uint32_t>> m_cellParticleIndices; // 近傍探索用セルリスト
    float m_ssfrResolutionScale = 0.5f;                 // SSFR を半解像度で回すスケール
    float m_restDensity = 1.0f;                         // 静止密度（圧力計算の基準値）
    float m_bilateralSpatialSigma = 2.0f;               // バイラテラルフィルタの空間シグマ
    float m_bilateralDepthSigma = 0.05f;                // バイラテラルフィルタの深度シグマ
    float m_bilateralNormalSigma = 16.0f;               // 法線一致性の鋭さ
    float m_bilateralKernelRadius = 2.0f;               // サンプル半径（ピクセル単位）
    float m_particleMass = 1.0f;                        // 粒子質量
    int m_solverIterations = 4;                         // PBF の反復回数
    float m_pbfEpsilon = 1e-4f;                         // 拘束安定化用イプシロン
    float m_xsphC = 0.03f;                              // XSPH 係数
    float m_sCorrK = -0.01f;                            // 圧力補正項の係数
    float m_sCorrN = 4.0f;                              // 圧力補正項の指数
    float m_deltaQFactor = 0.3f;                        // s_corr 計算用の基準距離割合
    DirectX::XMFLOAT3 m_gravity{ 0.0f, -9.8f, 0.0f };     // 重力ベクトル
    float m_poly6Coefficient = 0.0f;                    // Poly6 カーネル係数
    float m_spikyCoefficient = 0.0f;                    // Spiky カーネル係数
    float m_deltaQ = 0.0f;                              // s_corr 用距離
    float m_poly6DeltaQ = 0.0f;                         // s_corr 用カーネル値
};
