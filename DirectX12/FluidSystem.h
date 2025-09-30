#pragma once
#include "ComPtr.h"
#include "ConstantBuffer.h"
#include "DescriptorHeap.h"
#include "SharedStruct.h"
#include "SpatialGrid.h"
#include "ComputePipelineState.h"
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
    const FluidMaterial& Material() const { return m_material; }

    void SpawnParticlesSphere(const DirectX::XMFLOAT3& center, float radius, UINT count);
    void RemoveParticlesSphere(const DirectX::XMFLOAT3& center, float radius);

    void QueueGather(const DirectX::XMFLOAT3& target, float radius, float strength);
    void QueueSplash(const DirectX::XMFLOAT3& position, float radius, float impulse);
    void ClearDynamicOperations();

    void Simulate(ID3D12GraphicsCommandList* cmd, float dt);
    void Render(ID3D12GraphicsCommandList* cmd,
        const DirectX::XMFLOAT4X4& invViewProj,
        const DirectX::XMFLOAT4X4& viewProj,
        const DirectX::XMFLOAT3& camPos,
        float isoLevel);

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
    struct MetaConstants
    {
        DirectX::XMFLOAT4X4 InvViewProj; // ビュー射影逆行列（転置済み）
        DirectX::XMFLOAT4X4 ViewProj;    // ビュー射影行列（転置済み）
        DirectX::XMFLOAT4   CamRadius;   // カメラ座標と粒子半径
        DirectX::XMFLOAT4   IsoCount;    // 等値面しきい値 / 粒子数 / レイマーチ係数 / 未使用
        DirectX::XMFLOAT4   WaterDeep;   // 深い水の色 / w=吸収係数
        DirectX::XMFLOAT4   WaterShallow;// 浅い水の色 / w=泡検出のしきい値
        DirectX::XMFLOAT4   ShadingParams;// x=泡強度 y=反射割合 z=スペキュラ強度 w=時間
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

    struct GPUBufferHandle
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        DescriptorHandle* srv = nullptr;
        DescriptorHandle* uav = nullptr;
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    };

    static constexpr UINT kMaxParticlesPerCell = 64; // グリッド1セルが保持できる粒子数の上限

    void UpdateGridSettings();
    void ApplyExternalOperationsCPU(float dt);
    void StepCPU(float dt);
    void StepGPU(ID3D12GraphicsCommandList* cmd, float dt);
    void UploadCPUToGPU(ID3D12GraphicsCommandList* cmd);
    void ReadbackGPUToCPU();
    void UpdateParticleBuffer();
    bool CreateMetaPipeline(ID3D12Device* device, DXGI_FORMAT rtvFormat);
    void CreateGPUResources(ID3D12Device* device);
    void UpdateComputeParams(float dt);
    void ResolveBounds(FluidParticle& p) const;
    ID3D12GraphicsCommandList* BeginComputeCommandList();
    void SubmitComputeCommandList();
    bool HasValidGPUResources() const;

    float EffectiveTimeStep(float dt) const;

    // CPU管理
    std::vector<FluidParticle> m_cpuParticles;
    SpatialGrid                m_spatialGrid;

    // 描画関連
    static constexpr UINT kFrameCount = 2;
    std::array<std::unique_ptr<ConstantBuffer>, kFrameCount> m_metaCB{};
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_metaRootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_metaPipelineState;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_cpuMetaBuffer;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuMetaBuffer;
    DescriptorHandle* m_cpuMetaSRV = nullptr;
    DescriptorHandle* m_gpuMetaSRV = nullptr;
    DescriptorHandle* m_gpuMetaUAV = nullptr;
    DescriptorHandle* m_activeMetaSRV = nullptr;

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
