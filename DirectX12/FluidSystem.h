#pragma once
#include "ComPtr.h"
#include "ConstantBuffer.h"
#include "DescriptorHeap.h"
#include "SharedStruct.h"
#include "SpatialGrid.h"
#include "ComputePipelineState.h"
#include "Texture2D.h"
#include <d3d12.h>
#include <wrl.h>
#include <DirectXMath.h>
#include <array>
#include <memory>
#include <vector>
#include <Windows.h>

// ================================
//  流体マテリアルの設定パラメータ
// ================================
struct FluidMaterial
{
    float restDensity = 1000.0f;      // 静止密度
    float particleMass = 1.0f;        // 粒子質量
    float smoothingRadius = 0.12f;    // カーネル半径
    float viscosity = 0.02f;          // 粘性係数
    float stiffness = 200.0f;         // 圧力係数（GPU用）
    float renderRadius = 0.10f;       // メタボール描画半径
    float lambdaEpsilon = 100.0f;     // PBF安定化係数
    float xsphC = 0.05f;              // XSPH粘性（CPU）
    int   solverIterations = 1;       // PBF反復回数（安定動作用に軽量化）
};

// プリセットマテリアル
enum class FluidMaterialPreset
{
    Water,
    Magma,
};

// シミュレーションモード
enum class FluidSimulationMode
{
    CPU,
    GPU,
};

// CPU シミュレーション用の粒子情報
struct FluidParticle
{
    DirectX::XMFLOAT3 position;   // 現在位置
    DirectX::XMFLOAT3 velocity;   // 速度
    DirectX::XMFLOAT3 predicted;  // 予測位置
    float lambda = 0.0f;          // PBFラグランジュ乗数
    float density = 0.0f;         // 計算密度
    UINT  collisionMask = 0;      // 衝突が発生した境界を保持するビットマスク
};

// 流体が境界へ衝突した際に通知するイベント
struct FluidCollisionEvent
{
    DirectX::XMFLOAT3 position;   // 衝突位置
    DirectX::XMFLOAT3 normal;     // 衝突面の法線
    float               strength; // 衝突の強さ（速度から算出）
};

class FluidSystem
{
public:
    FluidSystem();
    ~FluidSystem();

    void Init(ID3D12Device* device, DXGI_FORMAT rtvFormat, UINT maxParticles, UINT threadGroupCount);

    void UseGPU(bool enable);
    FluidSimulationMode Mode() const;

    void SetMaterialPreset(FluidMaterialPreset preset);
    void SetMaterial(const FluidMaterial& material);
    void SetSimulationBounds(const DirectX::XMFLOAT3& minBound, const DirectX::XMFLOAT3& maxBound);
    const FluidMaterial& Material() const { return m_material; }

    void SpawnParticlesSphere(const DirectX::XMFLOAT3& center, float radius, UINT count);
    void RemoveParticlesSphere(const DirectX::XMFLOAT3& center, float radius);

    void QueueGather(const DirectX::XMFLOAT3& target, float radius, float strength);
    void QueueSplash(const DirectX::XMFLOAT3& position, float radius, float impulse);
    void QueueDirectionalImpulse(const DirectX::XMFLOAT3& center, float radius, const DirectX::XMFLOAT3& direction, float strength);
    void ClearDynamicOperations();

    bool Raycast(const DirectX::XMFLOAT3& origin, const DirectX::XMFLOAT3& direction, float maxDistance, float radius, DirectX::XMFLOAT3& hitPosition) const;
    void PopCollisionEvents(std::vector<FluidCollisionEvent>& outEvents);

    void Simulate(ID3D12GraphicsCommandList* cmd, float dt);
    void Render(ID3D12GraphicsCommandList* cmd,
        const DirectX::XMFLOAT4X4& view,
        const DirectX::XMFLOAT4X4& proj,
        const DirectX::XMFLOAT4X4& viewProj,
        const DirectX::XMFLOAT3& camPos,
        float isoLevel);

    void Composite(ID3D12GraphicsCommandList* cmd,
        ID3D12Resource* sceneColor,
        ID3D12Resource* sceneDepth,
        D3D12_CPU_DESCRIPTOR_HANDLE sceneRTV);

    // 水面のカラーや泡の強さを外部から調整できるようにする
    void SetWaterAppearance(const DirectX::XMFLOAT3& shallowColor,
        const DirectX::XMFLOAT3& deepColor,
        float absorption,
        float foamThreshold,
        float foamStrength,
        float reflectionStrength,
        float specularPower);

    void StartDrag(int, int, class Camera*) {}
    void Drag(int, int, class Camera*) {}
    void EndDrag() {}

private:
    struct SSFRCameraConstants
    {
        DirectX::XMFLOAT4X4 View;        // ビュー行列（転置済み）
        DirectX::XMFLOAT4X4 Proj;        // 射影行列（転置済み）
        DirectX::XMFLOAT4X4 ViewProj;    // ビュー射影行列（転置済み）
        DirectX::XMFLOAT2   ScreenSize;  // 画面解像度
        DirectX::XMFLOAT2   InvScreen;   // 逆解像度
        float               NearZ;       // ニア平面
        float               FarZ;        // ファー平面
        DirectX::XMFLOAT3   IorF0;       // フレネル用F0
        float               Absorption;  // Beer-Lambert係数
    };

    struct SSFRBlurParams
    {
        float Sigma;  // 空間ガウス係数
        float DepthK; // 深度差係数
        float NormalK;// 法線差係数
        float Padding;// 16byte揃え
    };

    struct GPUParams
    {
        float restDensity;
        float particleMass;
        float viscosity;
        float stiffness;
        float radius;
        float timeStep;
        UINT  particleCount;
        UINT  pad0;
        DirectX::XMFLOAT3 gridMin;
        float pad1;
        DirectX::XMUINT3  gridDim;
        UINT  pad2;
    };

    struct GatherOperation
    {
        DirectX::XMFLOAT3 target;
        float radius;
        float strength;
    };

    struct SplashOperation
    {
        DirectX::XMFLOAT3 origin;
        float radius;
        float impulse;
    };

    struct DirectionalImpulseOperation
    {
        DirectX::XMFLOAT3 center;
        float radius;
        DirectX::XMFLOAT3 direction;
        float strength;
    };

    struct GPUBufferHandle
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        DescriptorHandle* srv = nullptr;
        DescriptorHandle* uav = nullptr;
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    };

    static constexpr UINT kMaxParticlesPerCell = 64; // グリッド1セルが保持できる粒子数の上限
    static constexpr UINT kCollisionMinX = 1u << 0;
    static constexpr UINT kCollisionMaxX = 1u << 1;
    static constexpr UINT kCollisionMinY = 1u << 2;
    static constexpr UINT kCollisionMaxY = 1u << 3;
    static constexpr UINT kCollisionMinZ = 1u << 4;
    static constexpr UINT kCollisionMaxZ = 1u << 5;

    void UpdateGridSettings();
    void ApplyExternalOperationsCPU(float dt);
    void StepCPU(float dt);
    void StepGPU(ID3D12GraphicsCommandList* cmd, float dt);
    void UploadCPUToGPU(ID3D12GraphicsCommandList* cmd);
    void ReadbackGPUToCPU();
    void UpdateParticleBuffer();
    bool CreateSSFRResources(ID3D12Device* device, DXGI_FORMAT rtvFormat);
    void DestroySSFRResources();
    void CreateGPUResources(ID3D12Device* device);
    void UpdateComputeParams(float dt);
    UINT ResolveBounds(FluidParticle& p) const;
    ID3D12GraphicsCommandList* BeginComputeCommandList();
    void SubmitComputeCommandList();
    bool HasValidGPUResources() const;

    float EffectiveTimeStep(float dt) const;

    // CPU管理
    std::vector<FluidParticle> m_cpuParticles;
    SpatialGrid                m_spatialGrid;

    // 描画関連
    static constexpr UINT kFrameCount = 2;
    std::array<std::unique_ptr<ConstantBuffer>, kFrameCount> m_cameraCB{}; // カメラ定数バッファ
    std::unique_ptr<ConstantBuffer> m_blurParamsCB;                        // ブラー用定数
    Microsoft::WRL::ComPtr<ID3D12Resource> m_cpuMetaBuffer;                // 粒子インスタンスデータ（CPU側）
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuMetaBuffer;                // 粒子インスタンスデータ（GPU側）
    DescriptorHandle* m_cpuMetaSRV = nullptr;                              // CPU粒子のSRV
    DescriptorHandle* m_gpuMetaSRV = nullptr;                              // GPU粒子のSRV
    DescriptorHandle* m_gpuMetaUAV = nullptr;                              // GPU粒子のUAV
    DescriptorHandle* m_activeMetaSRV = nullptr;                           // 現在描画に利用する粒子SRV

    // SSFR用テクスチャ
    std::unique_ptr<Texture2D> m_particleDepthTexture;                     // 粒子の深度
    std::unique_ptr<Texture2D> m_smoothedDepthTexture;                     // 平滑化後の深度
    std::unique_ptr<Texture2D> m_normalTexture;                            // 法線
    std::unique_ptr<Texture2D> m_thicknessTexture;                         // 厚み

    DescriptorHandle* m_particleDepthSRV = nullptr;
    DescriptorHandle* m_particleDepthUAV = nullptr;
    DescriptorHandle* m_smoothedDepthSRV = nullptr;
    DescriptorHandle* m_smoothedDepthUAV = nullptr;
    DescriptorHandle* m_normalSRV = nullptr;
    DescriptorHandle* m_normalUAV = nullptr;
    DescriptorHandle* m_thicknessSRV = nullptr;
    DescriptorHandle* m_thicknessUAV = nullptr;

    DescriptorHandle* m_sceneColorSRV = nullptr;
    DescriptorHandle* m_sceneDepthSRV = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_sceneColorCopy;
    D3D12_RESOURCE_STATES m_sceneColorCopyState = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES m_sceneDepthState = D3D12_RESOURCE_STATE_DEPTH_WRITE;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_ssfrRtvHeap;            // RTV専用ヒープ
    UINT m_ssfrRtvDescriptorSize = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE m_particleDepthRTV = {};
    D3D12_CPU_DESCRIPTOR_HANDLE m_smoothedDepthRTV = {};
    D3D12_CPU_DESCRIPTOR_HANDLE m_thicknessRTV = {};

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_ssfrCpuUavHeap;         // UAVクリア用CPUヒープ
    UINT m_ssfrCpuUavDescriptorSize = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE m_particleDepthUavCpuHandle = {};
    D3D12_CPU_DESCRIPTOR_HANDLE m_smoothedDepthUavCpuHandle = {};

    D3D12_RESOURCE_STATES m_particleDepthState = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES m_smoothedDepthState = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES m_normalState = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES m_thicknessState = D3D12_RESOURCE_STATE_COMMON;

    // SSFR用ルートシグネチャとPSO
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_particleRootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_particlePipelineState;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_blurRootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_blurPipelineState;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_normalRootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_normalPipelineState;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_compositeRootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_compositePipelineState;

    // GPUシミュレーション関連
    bool m_useGPU = false;
    bool m_gpuAvailable = false;
    bool m_cpuDirty = true;
    bool m_gpuDirty = true;
    bool m_pendingReadback = false;
    UINT m_gpuReadIndex = 0;
    GPUBufferHandle m_gpuParticleBuffers[2];
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuGridCount;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuGridTable;
    DescriptorHandle* m_gpuGridCountSRV = nullptr;
    DescriptorHandle* m_gpuGridTableSRV = nullptr;
    DescriptorHandle* m_gpuGridCountUAV = nullptr;
    DescriptorHandle* m_gpuGridTableUAV = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuUpload;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuReadback;
    std::unique_ptr<ConstantBuffer> m_computeParamsCB;
    std::unique_ptr<ConstantBuffer> m_dummyViewCB;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_computeRootSignature;
    std::unique_ptr<ComputePipelineState> m_buildGridPipeline;
    std::unique_ptr<ComputePipelineState> m_particlePipeline;
    std::unique_ptr<ComputePipelineState> m_clearGridPipeline;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_computeAllocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_computeCommandList;
    Microsoft::WRL::ComPtr<ID3D12Fence> m_computeFence;
    HANDLE m_computeFenceEvent = nullptr;
    UINT64 m_computeFenceValue = 0;
    UINT64 m_lastSubmittedComputeFence = 0;

    // 共通設定
    FluidMaterial m_material;
    DirectX::XMFLOAT3 m_boundsMin;
    DirectX::XMFLOAT3 m_boundsMax;
    DirectX::XMUINT3  m_gridDim;
    UINT              m_particleCount = 0;
    UINT              m_maxParticles = 0;
    bool              m_initialized = false;

    std::vector<GatherOperation> m_gatherOps;
    std::vector<SplashOperation> m_splashOps;
    std::vector<DirectionalImpulseOperation> m_directionalOps;
    std::vector<FluidCollisionEvent> m_collisionEvents;

    ID3D12Device* m_device = nullptr;
    DXGI_FORMAT   m_rtvFormat = DXGI_FORMAT_R8G8B8A8_UNORM;

    DirectX::XMFLOAT3 m_waterColorDeep;
    DirectX::XMFLOAT3 m_waterColorShallow;
    float m_waterAbsorption = 0.35f;
    float m_foamCurvatureThreshold = 0.45f;
    float m_foamStrength = 0.35f;
    float m_reflectionStrength = 0.6f;
    float m_specularPower = 64.0f;
    float m_totalSimulatedTime = 0.0f;
};

// プリセットマテリアルの生成ヘルパー
FluidMaterial CreateFluidMaterial(FluidMaterialPreset preset);
