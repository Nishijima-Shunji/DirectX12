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
    void InitializeGrid();
    void ClearGridNodes();
    size_t GridIndex(int x, int y, int z) const;
    void EnsureGridReady();

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
    std::unique_ptr<PipelineState> m_pointPipelineState; // 粒子位置確認用の点描画PSO
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
    float m_maxVelocity = 6.0f;     // 速度クランプ値

    RenderMode m_renderMode = RenderMode::SSFR; // 描画モード
    bool m_useSSFR = true;          // SSFR を使用するかどうか（既定で ON にして球メッシュ描画を置き換える）
    bool m_drawParticlePoints = true; // SSFR と並行して粒子位置を点で可視化するかどうか

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
        DirectX::XMMATRIX proj;             // プロジェクション行列
        DirectX::XMMATRIX view;             // ビュー行列
        DirectX::XMMATRIX world;            // ワールド行列（粒子とスクリーンスペース結果の整合用）
        DirectX::XMFLOAT2 screenSize;       // 流体用レンダーターゲットの解像度
        float nearZ;                        // 近クリップ
        float farZ;                         // 遠クリップ
        DirectX::XMFLOAT3 iorF0;            // Fresnel 計算に利用する F0
        float absorb;                       // Beer-Lambert 減衰係数
        DirectX::XMFLOAT2 framebufferSize;  // フル解像度のバックバッファサイズ
        float refractionScale;              // 屈折オフセット量
        float thicknessScale;               // 厚み減衰スケール
        DirectX::XMFLOAT2 invScreenSize;    // 流体ターゲットの逆解像度
        DirectX::XMFLOAT2 padding;          // 16byte 境界を維持する余白
    };

    struct SSFRTarget
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource; // GPU リソース
        DescriptorHandle* srvHandle = nullptr;            // SRV ディスクリプタ
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle{};          // RTV 用 CPU ディスクリプタ
        D3D12_RESOURCE_STATES currentState = D3D12_RESOURCE_STATE_COMMON; // 現在のリソース状態
    };

    std::array<std::unique_ptr<ConstantBuffer>, Engine::FRAME_BUFFER_COUNT> m_ssfrConstantBuffers; // SSFR 用定数バッファ

    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_ssfrParticlePSO; // ビルボード粒子描画 PSO
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_ssfrCompositePSO; // 合成 PSO
    std::unique_ptr<RootSignature> m_ssfrParticleRootSig;     // 粒子描画用ルートシグネチャ
    std::unique_ptr<RootSignature> m_ssfrCompositeRootSig;    // 合成用ルートシグネチャ

    SSFRTarget m_rawDepth;        // 粒子スプラット結果（深度）
    SSFRTarget m_thickness;       // 粒子厚み積算

    SSFRTarget m_sceneColorCopy;  // 背景カラー退避用（SRV のみ使用）
    DescriptorHandle* m_sceneDepthSrv = nullptr; // シーン深度 SRV
    ID3D12Resource* m_cachedSceneDepth = nullptr; // 深度バッファのリソースキャッシュ（リサイズ検出用）

    DescriptorHandle* m_particleBufferSrv = nullptr; // 粒子インスタンス SRV

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_ssfrRtvDescriptorHeap; // RTV 用 CPU ヒープ
    UINT m_ssfrRtvDescriptorSize = 0;  // RTV ヒープのインクリメント
    UINT m_ssfrRtvDescriptorCursor = 0; // 次に割り当てる RTV ディスクリプタ位置

    UINT m_ssfrWidth = 0;  // 半解像度幅
    UINT m_ssfrHeight = 0; // 半解像度高さ

    void RenderSSFR(ID3D12GraphicsCommandList* cmd, const Camera& camera); // SSFR 描画処理本体
    bool EnsureSSFRResources(); // SSFR リソース初期化
    bool CreateSSFRRootSignatures(); // SSFR 用ルートシグネチャ生成
    bool CreateSSFROnce(); // 初回初期化をまとめる
    bool ResizeSSFRTargets(UINT width, UINT height); // 解像度変更対応
    void UpdateSSFRConstants(const Camera& camera); // 定数バッファ更新
    void PrepareSSFRTargets(ID3D12GraphicsCommandList* cmd); // RTV/UAV を毎フレーム準備
    void TransitionSSFRTarget(ID3D12GraphicsCommandList* cmd, SSFRTarget& target, D3D12_RESOURCE_STATES newState); // 状態遷移
    D3D12_CPU_DESCRIPTOR_HANDLE AllocateRtvDescriptor(); // RTV 用ヒープからディスクリプタを確保
    bool CreateParticlePSO(); // 粒子 PSO 生成
    bool CreateCompositePSO(); // 合成 PSO 生成
    void GenerateMarchingCubesMesh(); // マーチングキューブメッシュ生成
    void UpdateMarchingBuffers(); // マーチングキューブ用VB/IB更新
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
    float m_ssfrResolutionScale = 0.5f;                 // SSFR を半解像度で回すスケール
    float m_restDensity = 1.0f;                         // 静止密度（圧力計算の基準値）
    float m_pressureStiffness = 6.0f;                   // 圧力係数（反発力の強さを制御）
    float m_refractionStrength = 0.05f;                 // 屈折オフセットの強さ
    float m_thicknessAttenuation = 1.2f;                // 厚みを使った減衰係数
};
