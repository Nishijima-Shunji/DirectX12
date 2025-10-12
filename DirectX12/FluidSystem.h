#pragma once
#include <DirectXMath.h>
#include <d3d12.h>
#include <memory>
#include <vector>
#include <array>
#include <unordered_map>
#include <functional>
#include <cmath>
#include <algorithm>

#include "ComPtr.h"
#include "ConstantBuffer.h"
#include "RootSignature.h"
#include "PipelineState.h"
#include "FullscreenPSO.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "Camera.h"
#include "Engine.h"
#include "DescriptorHeap.h"

class FluidSystem
{
public:
    struct Bounds { DirectX::XMFLOAT3 min; DirectX::XMFLOAT3 max; };

    enum class SimMode { Heightfield, Particles };

    FluidSystem();
    ~FluidSystem();

    // particleCount>0 なら 粒子モード、0 なら 高さ場モード
    bool Init(ID3D12Device* device, const Bounds& bounds, size_t particleCount);

    void Update(float deltaTime);
    void Draw(ID3D12GraphicsCommandList* cmd, const Camera& camera);

    void AdjustWall(const DirectX::XMFLOAT3& direction, float amount);
    void AdjustWall(const DirectX::XMFLOAT3& dir, float amount, float deltaTime);

    void SetCameraLiftRequest(const DirectX::XMFLOAT3& origin, const DirectX::XMFLOAT3& direction, float deltaTime);
    void ClearCameraLiftRequest();

    const Bounds& GetBounds() const { return m_bounds; }
    float GetWaterLevel() const { return m_waterLevel; }
    SimMode GetMode() const { return m_mode; }

    // マウス等から呼ぶ（GameScene側でRay→XZ座標にして渡す）
    void BeginGrab(const DirectX::XMFLOAT2& xz, float radius);
    void UpdateGrab(const DirectX::XMFLOAT2& xz, float liftPerSec, float deltaTime);
    void EndGrab(const DirectX::XMFLOAT2& throwDirXZ, float throwSpeed);

    // 水を切る：線上に連続で押し下げ
    void CutWater(const DirectX::XMFLOAT2& xzFrom, const DirectX::XMFLOAT2& xzTo, float radius, float depth);

    // （オプション）粒子に直接インタラクション
    void ParticleAttractDisc(const DirectX::XMFLOAT2& centerXZ, float radius, float strength, float dt);
    void ParticleLaunch(const DirectX::XMFLOAT2& dirXZ, float speed);

    // --- デバッグ粒子の見える化（GameScene から呼ぶ） ---
    struct DebugParticleView { DirectX::XMFLOAT3 pos, vel; float radius; };

    template<class Fn>
    void ForEachDebugParticle(Fn&& fn) const {
        // しぶき & 水玉（高さ場モード由来）
        const float sprayR = 0.02f;
        for (const auto& s : m_spray) fn(DebugParticleView{ s.pos, s.vel, sprayR });
        if (m_blob.active)            fn(DebugParticleView{ m_blob.pos, m_blob.vel, m_blob.radius });

        // 粒子モードの粒子（多すぎると重いので可視サンプリング）
        if (m_mode == SimMode::Particles) {
            const int stride = std::max(1, (int)(m_particles.size() / 2000)); // 最大~2000個だけ描く
            for (size_t i = 0; i < m_particles.size(); i += stride)
                fn(DebugParticleView{ m_particles[i].pos, m_particles[i].vel, m_particleRadius });
        }
    }

private:
    // ===== 高さ場（従来） =====
    struct OceanVertex {
        DirectX::XMFLOAT3 position;
        DirectX::XMFLOAT3 normal;
        DirectX::XMFLOAT2 uv;
        DirectX::XMFLOAT4 color;
    };
    struct OceanConstant {
        DirectX::XMFLOAT4X4 world, view, proj;
        DirectX::XMFLOAT4 color;
    };
    struct DropRequest { DirectX::XMFLOAT2 uv; float strength; float radius; };

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

    void ApplyWallImpulse(const Bounds& prev, const Bounds& curr, float dt);
    void ApplyDiscImpulse(const DirectX::XMFLOAT2& centerXZ, float radius, float addHeight, float addVel);
    void UpdateInteractions(float dt);

    struct WavePacket {
        DirectX::XMFLOAT2 center, vel;
        float radius, amp, life;
    };
    std::vector<WavePacket> m_packets;

    struct SprayParticle { DirectX::XMFLOAT3 pos, vel; float life; };
    std::vector<SprayParticle> m_spray;

    struct WaterBlob {
        bool active = false;
        DirectX::XMFLOAT3 pos{ 0,0,0 }, vel{ 0,0,0 };
        float radius = 0, volume = 0;
    } m_blob;

    // Gather（体積保存・分離）
    void TransferAnnulusToCenter(const DirectX::XMFLOAT2& centerXZ, float innerR, float ringW, float wantVolume, float& outTakenVolume);
    void DepositVolumeGaussian(const DirectX::XMFLOAT2& centerXZ, float volume, float sigma);
    void UpdateBlob(float dt);

    bool  m_gatherActive = false;
    DirectX::XMFLOAT2 m_gatherCenter{ 0,0 };
    float m_gatherRadius = 0.18f, m_gatherRingW = 0.20f, m_gatheredVolume = 0.0f, m_gatherRate = 0.015f;

    // ===== 粒子（SPHライト） =====
    struct Particle { DirectX::XMFLOAT3 pos{ 0,0,0 }, vel{ 0,0,0 }; float density = 0.0f, pressure = 0.0f; };
    std::vector<Particle> m_particles;
    struct ParticleInstanceGPU { DirectX::XMFLOAT3 pos; float radius; };

    // 空間ハッシュ
    std::unordered_map<long long, std::vector<int>> m_grid;
    long long HashCell(int ix, int iy, int iz) const {
        return ((long long)ix * 73856093LL) ^ ((long long)iy * 19349663LL) ^ ((long long)iz * 83492791LL);
    }
    void BuildGrid();
    void ForNeighbors(int i, const std::function<void(int j, const DirectX::XMFLOAT3& rij, float rlen)>& fn);

    void InitParticles(int count);
    void UpdateParticles(float dt);
    void SPH_DensityPressure();
    void SPH_ForcesIntegrate(float dt);
    void ResolveWall(DirectX::XMFLOAT3& p, DirectX::XMFLOAT3& v, float restitution = 0.2f, float friction = 0.2f);

    void DrawParticles(ID3D12GraphicsCommandList* cmd, const Camera& cam);

    bool BuildParticleRenderResources();

private:
    // 共有
    ID3D12Device* m_device = nullptr;
    Bounds m_bounds{};
    SimMode m_mode = SimMode::Heightfield;

    // 高さ場パラメタ
    int   m_resolution = 128;
    float m_waveSpeed = 9.0f;
    float m_damping = 0.9975f;
    float m_waveTimeScale = 1.35f;
    float m_timeSeconds = 0.0f;
    float m_waterLevel = 0.0f;

    DirectX::XMFLOAT3 m_simMin{ 0,0,0 };
    float m_cellDx = 0.0f, m_cellDz = 0.0f;

    std::vector<float> m_height, m_velocity;
    std::vector<OceanVertex> m_vertices;
    std::vector<uint32_t> m_indices;
    std::vector<DropRequest> m_pendingDrops;

    std::unique_ptr<RootSignature> m_rootSignature;
    std::unique_ptr<PipelineState> m_pipelineState;
    std::array<std::unique_ptr<ConstantBuffer>, Engine::FRAME_BUFFER_COUNT> m_constantBuffers;
    std::unique_ptr<VertexBuffer> m_vertexBuffer;
    std::unique_ptr<IndexBuffer>  m_indexBuffer;

    bool  m_liftRequested = false;
    float m_minWallExtent = 0.5f;

    // パケット/しぶき用ノブ
    float m_packetDefaultRadius = 0.18f, m_packetDefaultAmp = 0.08f;
    float m_packetFriction = 0.25f, m_packetDecay = 0.9f;
    float m_sprayRate = 60.0f, m_sprayGravity = -9.8f, m_sprayDrag = 2.0f, m_sprayLife = 0.6f;

    // 粒子パラメタ
    int   m_particleCap = 0;
    float m_kernelH = 0.07f;      // 近傍半径h
    float m_particleRadius = 0.03f;
    float m_mass = 0.10f;         // 粒子質量
    float m_rho0 = 1000.0f;       // 目標密度
    float m_stiffness = 3000.0f;  // 圧力係数
    float m_viscosity = 0.08f;    // 粘性
    float m_xsph = 0.05f;         // XSPH係数

    // 粒子描画用リソース
    std::unique_ptr<VertexBuffer> m_particleQuadVB;
    std::unique_ptr<IndexBuffer>  m_particleQuadIB;
    std::unique_ptr<VertexBuffer> m_particleInstanceVB;
    std::unique_ptr<RootSignature> m_particleDepthRoot;
    std::unique_ptr<PipelineState> m_particleDepthPSO;
    std::array<std::unique_ptr<ConstantBuffer>, Engine::FRAME_BUFFER_COUNT> m_particleDepthCB;
    std::unique_ptr<RootSignature> m_particleCompositeRoot;
    std::unique_ptr<FullscreenPSO> m_particleCompositePSO;
    std::array<std::unique_ptr<ConstantBuffer>, Engine::FRAME_BUFFER_COUNT> m_particleCompositeCB;
    ComPtr<ID3D12DescriptorHeap> m_particleRtvHeap;
    D3D12_CPU_DESCRIPTOR_HANDLE m_particleDepthRtv{};
    ComPtr<ID3D12Resource> m_particleDepthTexture;
    DescriptorHandle* m_particleDepthSrv = nullptr;
    D3D12_RESOURCE_STATES m_particleDepthState = D3D12_RESOURCE_STATE_COMMON;
};
