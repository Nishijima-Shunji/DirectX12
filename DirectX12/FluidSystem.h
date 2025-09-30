#pragma once
#include "ComPtr.h"
#include "ConstantBuffer.h"
#include "DescriptorHeap.h"
#include "MetaBallPipelineState.h"
#include <d3d12.h>
#include <wrl.h>
#include <DirectXMath.h>
#include <memory>
#include <vector>

// ================================
//  シンプルな流体（メタボール）描画用の構造体
// ================================
struct FluidMaterial
{
    float renderRadius = 0.10f;   // メタボール半径
    float restDensity = 0.0f;     // 旧機能との互換用（未使用）
    float particleMass = 0.0f;    // 旧機能との互換用（未使用）
    float smoothingRadius = 0.0f; // 旧機能との互換用（未使用）
    float viscosity = 0.0f;       // 旧機能との互換用（未使用）
    float stiffness = 0.0f;       // 旧機能との互換用（未使用）
    float lambdaEpsilon = 0.0f;   // 旧機能との互換用（未使用）
    float xsphC = 0.0f;           // 旧機能との互換用（未使用）
    int   solverIterations = 1;   // 旧機能との互換用（未使用）
};

enum class FluidMaterialPreset
{
    Water,
    Magma,
};

// ================================
//  シンプルなメタボール描画のための流体システム
// ================================
class FluidSystem
{
public:
    FluidSystem();
    ~FluidSystem();

    void Init(ID3D12Device* device, DXGI_FORMAT rtvFormat, UINT maxParticles, UINT threadGroupCount);

    void UseGPU(bool /*enable*/) {} // GPU シミュレーションは無効化

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
        const DirectX::XMFLOAT4X4& view,
        const DirectX::XMFLOAT4X4& proj,
        const DirectX::XMFLOAT4X4& viewProj,
        const DirectX::XMFLOAT3& camPos,
        float isoLevel);

    void Composite(ID3D12GraphicsCommandList* cmd,
        ID3D12Resource* sceneColor,
        ID3D12Resource* sceneDepth,
        D3D12_CPU_DESCRIPTOR_HANDLE sceneRTV);

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
    struct Particle
    {
        DirectX::XMFLOAT3 position;
        DirectX::XMFLOAT3 velocity;
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

    struct MetaBallInstance
    {
        DirectX::XMFLOAT3 position;
        float radius;
    };

    struct MetaConstants
    {
        DirectX::XMFLOAT4X4 InvViewProj;
        DirectX::XMFLOAT4X4 ViewProj;
        DirectX::XMFLOAT4   CamRadius;
        DirectX::XMFLOAT4   IsoCount;
        DirectX::XMFLOAT4   GridMinCell;
        DirectX::XMUINT4    GridDimInfo;
        DirectX::XMFLOAT4   WaterDeep;
        DirectX::XMFLOAT4   WaterShallow;
        DirectX::XMFLOAT4   ShadingParams;
    };

    bool CreateMetaResources(ID3D12Device* device, DXGI_FORMAT rtvFormat);
    void UpdateParticleBuffer();

    // シミュレーション設定
    FluidMaterial m_material{};
    std::vector<Particle> m_particles;
    std::vector<GatherOperation> m_gathers;
    std::vector<SplashOperation> m_splashes;
    DirectX::XMFLOAT3 m_boundsMin;
    DirectX::XMFLOAT3 m_boundsMax;
    float m_elapsedTime = 0.0f;

    // GPU リソース
    UINT m_maxParticles = 0;
    UINT m_particleCount = 0;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_particleBuffer;
    void* m_particleMapped = nullptr;
    DescriptorHandle* m_particleSRV = nullptr;

    Microsoft::WRL::ComPtr<ID3D12Resource> m_dummyGridTable;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_dummyGridCount;
    DescriptorHandle* m_dummyGridTableSRV = nullptr;
    DescriptorHandle* m_dummyGridCountSRV = nullptr;

    std::unique_ptr<ConstantBuffer> m_constantBuffer;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pipelineState;

    // 描画パラメータ
    DirectX::XMFLOAT3 m_waterColorShallow;
    DirectX::XMFLOAT3 m_waterColorDeep;
    float m_waterAbsorption = 0.3f;
    float m_foamThreshold = 0.4f;
    float m_foamStrength = 0.5f;
    float m_reflectionStrength = 0.6f;
    float m_specularPower = 64.0f;
    float m_isoLevel = 1.0f;
};

FluidMaterial CreateFluidMaterial(FluidMaterialPreset preset);
