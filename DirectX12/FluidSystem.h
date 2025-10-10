#pragma once
#include <DirectXMath.h>
#include <d3d12.h>
#include <memory>
#include <vector>
#include <array>
#include "ConstantBuffer.h"
#include "RootSignature.h"
#include "PipelineState.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "Camera.h"
#include "Engine.h"

// ※コメントは分かりやすい日本語(Shift-JIS)で記入すること。
// ※WebGPU Oceanに合わせて格子波シミュレーションをDirectX12へ移植した管理クラス。

class FluidSystem
{
public:
    struct Bounds
    {
        DirectX::XMFLOAT3 min;
        DirectX::XMFLOAT3 max;
    };

    FluidSystem();
    ~FluidSystem();

    bool Init(ID3D12Device* device, const Bounds& bounds, size_t particleCount);
    void Update(float deltaTime);
    void Draw(ID3D12GraphicsCommandList* cmd, const Camera& camera);

    void AdjustWall(const DirectX::XMFLOAT3& direction, float amount);
    void AdjustWall(const DirectX::XMFLOAT3& dir, float amount, float deltaTime);
    void SetCameraLiftRequest(const DirectX::XMFLOAT3& origin, const DirectX::XMFLOAT3& direction, float deltaTime);
    void ClearCameraLiftRequest();

    const Bounds& GetBounds() const { return m_bounds; }

private:
    struct OceanVertex
    {
        DirectX::XMFLOAT3 position;
        DirectX::XMFLOAT3 normal;
        DirectX::XMFLOAT2 uv;
        DirectX::XMFLOAT4 color;
    };

    struct OceanConstant
    {
        DirectX::XMFLOAT4X4 world;
        DirectX::XMFLOAT4X4 view;
        DirectX::XMFLOAT4X4 proj;
        DirectX::XMFLOAT4 color;
    };

    struct DropRequest
    {
        DirectX::XMFLOAT2 uv;
        float strength;
        float radius;
    };

    bool BuildSimulationResources();
    bool BuildRenderResources();
    void StepSimulation(float deltaTime);
    void ApplyPendingDrops();
    void ApplyDrop(const DropRequest& drop);
    void UpdateVertexBuffer();
    void UpdateCameraCB(const Camera& camera);
    void ResetWaveState();
    bool RayIntersectBounds(const DirectX::XMFLOAT3& origin, const DirectX::XMFLOAT3& direction, DirectX::XMFLOAT3& hitPoint) const;

    size_t Index(size_t x, size_t z) const { return z * m_resolution + x; }

private:
    ID3D12Device* m_device = nullptr;
    Bounds m_bounds{};

    int   m_resolution = 128;
    float m_waveSpeed = 9.0f;       // 波速↑（水の“走り”を速く）
    float m_damping = 0.9975f;      // 減衰を弱く（粘度感↓）
    float m_waveTimeScale = 1.35f;  // VSの波アニメ用の時刻スケール
    float m_timeSeconds = 0.0f;     // 経過秒
    float m_waterLevel = 0.0f;

    DirectX::XMFLOAT3 m_simMin{ 0,0,0 }; // 初期グリッドの原点（XZ用）
    float m_cellDx = 0.0f;                // X のセル幅（固定）
    float m_cellDz = 0.0f;                // Z のセル幅（固定）

    std::vector<float> m_height;
    std::vector<float> m_velocity;
    std::vector<OceanVertex> m_vertices;
    std::vector<uint32_t> m_indices;
    std::vector<DropRequest> m_pendingDrops;

    std::unique_ptr<RootSignature> m_rootSignature;
    std::unique_ptr<PipelineState> m_pipelineState;
    std::array<std::unique_ptr<ConstantBuffer>, Engine::FRAME_BUFFER_COUNT> m_constantBuffers;

    std::unique_ptr<VertexBuffer> m_vertexBuffer;
    std::unique_ptr<IndexBuffer> m_indexBuffer;

    bool m_liftRequested = false;

    void ApplyWallImpulse(const Bounds& prev, const Bounds& curr, float dt);

    float m_minWallExtent = 0.5f;
};

