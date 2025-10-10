#pragma once
#include <DirectXMath.h>
#include <d3d12.h>
#include <memory>
#include <vector>
#include "ConstantBuffer.h"
#include "RootSignature.h"
#include "PipelineState.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "Camera.h"
#include "Engine.h"
#include "SpatialGrid.h"

// ※コメントは分かりやすい日本語(Shift-JIS)で記入すること。
// ※MLS-MPM流体とSSFR描画をDirectX12で統合管理するクラス。

class FluidSystem
{
public:
    enum class RenderMode
    {
        SSFR,
        InstancedSpheres,
        MarchingCubes
    };

    struct Bounds
    {
        DirectX::XMFLOAT3 min;
        DirectX::XMFLOAT3 max;
    };

    FluidSystem();
    ~FluidSystem();

    bool Init(ID3D12Device* device, const Bounds& bounds, size_t particleCount, RenderMode mode);
    void Update(float deltaTime);
    void Draw(ID3D12GraphicsCommandList* cmd, const Camera& camera);

    void AdjustWall(const DirectX::XMFLOAT3& direction, float amount);
    void SetCameraLiftRequest(const DirectX::XMFLOAT3& origin, const DirectX::XMFLOAT3& direction, float deltaTime);
    void ClearCameraLiftRequest();

    const Bounds& GetBounds() const { return m_bounds; }
    RenderMode CurrentRenderMode() const { return m_renderMode; }

private:
    struct Particle
    {
        DirectX::XMFLOAT3 position;
        DirectX::XMFLOAT3 velocity;
        DirectX::XMFLOAT3 previous;
        DirectX::XMFLOAT3X3 affine;  // MLS-MPM用のC行列
        float mass;
    };

    struct GridNode
    {
        DirectX::XMFLOAT3 velocity;
        float mass;
    };

    struct InstanceData
    {
        DirectX::XMFLOAT3 position;
        float radius;
    };

    struct CameraConstants
    {
        DirectX::XMFLOAT4X4 world;
        DirectX::XMFLOAT4X4 view;
        DirectX::XMFLOAT4X4 proj;
    };

    bool BuildParticles(size_t particleCount);
    bool BuildGrid();
    bool BuildRenderResources();
    bool BuildInstanceBuffer();

    void StepMLSMPM(float deltaTime);
    void TransferParticlesToGrid();
    void UpdateGrid(float deltaTime);
    void TransferGridToParticles(float deltaTime);

    void ApplyBounds(Particle& particle);
    void RebuildSpatialGrid();
    void ApplyCameraLift(float deltaTime);
    size_t FindRayHitParticle(const DirectX::XMFLOAT3& origin, const DirectX::XMFLOAT3& direction) const;

    void UpdateInstanceBuffer();
    void UpdateCameraCB(const Camera& camera);

    D3D12_INPUT_LAYOUT_DESC CreateInputLayout();

    GridNode& GridAt(int x, int y, int z);
    const GridNode& GridAt(int x, int y, int z) const;

    size_t GridIndex(int x, int y, int z) const;
    DirectX::XMFLOAT3 GridCellCenter(int x, int y, int z) const;

private:
    ID3D12Device* m_device = nullptr;

    Bounds m_bounds{};
    RenderMode m_renderMode = RenderMode::SSFR;

    float m_particleRadius = 0.03f;
    float m_cellSize = 0.06f;

    DirectX::XMINT3 m_gridDim{};
    DirectX::XMFLOAT3 m_gridMin{};

    std::vector<Particle> m_particles;
    std::vector<GridNode> m_grid;

    SpatialGrid m_neighborGrid;

    std::unique_ptr<RootSignature> m_rootSignature;
    std::unique_ptr<PipelineState> m_instancedPso;
    std::unique_ptr<PipelineState> m_ssfrPso;

    std::unique_ptr<VertexBuffer> m_meshVB;
    std::unique_ptr<VertexBuffer> m_instanceVB;
    std::unique_ptr<IndexBuffer> m_meshIB;
    UINT m_indexCount = 0;

    std::unique_ptr<ConstantBuffer> m_cameraCB[Engine::FRAME_BUFFER_COUNT];

    std::vector<InstanceData> m_instanceData;

    bool m_liftRequested = false;
    DirectX::XMFLOAT3 m_liftRayOrigin{};
    DirectX::XMFLOAT3 m_liftRayDirection{};
    size_t m_liftTargetIndex = static_cast<size_t>(-1);
    float m_liftAccumulated = 0.0f;

    float m_restDensity = 900.0f;
    DirectX::XMFLOAT3 m_gravity{ 0.0f, -9.8f, 0.0f };
};

