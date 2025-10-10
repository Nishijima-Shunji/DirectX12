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

	// マウス等から呼ぶ（GameScene側でRay→XZ座標にして渡す）
	void BeginGrab(const DirectX::XMFLOAT2& xz, float radius);
	void UpdateGrab(const DirectX::XMFLOAT2& xz, float liftPerSec, float deltaTime);
	void EndGrab(const DirectX::XMFLOAT2& throwDirXZ, float throwSpeed); // 投げる

	// 割る：線上に連続で押し下げインパルス
	void CutWater(const DirectX::XMFLOAT2& xzFrom, const DirectX::XMFLOAT2& xzTo, float radius, float depth);
	float GetWaterLevel() const { return m_waterLevel; }

	// 集める操作
	void BeginGather(const DirectX::XMFLOAT2& xz, float gatherRadius);
	void UpdateGather(const DirectX::XMFLOAT2& xz, float gatherRate, float dt); // 周囲→中心へ体積移送
	void EndGather(const DirectX::XMFLOAT2& aimDirXZ, float launchSpeed);       // 分離→発射

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

	// --- インタラクションの中核 ---
	void ApplyDiscImpulse(const DirectX::XMFLOAT2& centerXZ,
		float radius,
		float addHeight,     // 高さを直接足す（持ち上げ/割る）
		float addVel);       // 速度を足す（波発生/加速）
	void UpdateInteractions(float dt); // Update()内で呼ぶ

	// --- 投げる“パケット” ---
	struct WavePacket {
		DirectX::XMFLOAT2 center; // XZ
		DirectX::XMFLOAT2 vel;    // m/s
		float radius;             // 見かけの塊半径
		float amp;                // 山の振幅（高さ）
		float life;               // 秒

	};
	std::vector<WavePacket> m_packets;

	// --- 簡易しぶき（論理のみ。描画は次手で）---
	struct SprayParticle {
		DirectX::XMFLOAT3 pos;
		DirectX::XMFLOAT3 vel;
		float life;

	};
	std::vector<SprayParticle> m_spray;

	// 体積保存で集めて分離→飛ばすための構造体
	struct WaterBlob {
		bool  active = false;
		DirectX::XMFLOAT3 pos{ 0,0,0 };   // 中心
		DirectX::XMFLOAT3 vel{ 0,0,0 };   // 速度
		float radius = 0.0f;            // 見かけ半径（描画＆衝突）
		float volume = 0.0f;            // 集めた体積（m^3）
	};

	// ブラシ状態
	bool  m_grabActive = false;
	DirectX::XMFLOAT2 m_grabCenterXZ{ 0,0 };
	float m_grabRadius = 0.15f; // m

	// 調整ノブ
	float m_packetDefaultRadius = 0.18f;
	float m_packetDefaultAmp = 0.08f;
	float m_packetFriction = 0.25f; // 減速
	float m_packetDecay = 0.9f;  // 1秒当たりの減衰（exp）
	float m_sprayRate = 60.0f; // 1m移動あたりの発生目安（粒）
	float m_sprayGravity = -9.8f;
	float m_sprayDrag = 2.0f;  // 空気抵抗
	float m_sprayLife = 0.6f;

	//

	// 高さ場の体積操作
	void TransferAnnulusToCenter(const DirectX::XMFLOAT2& centerXZ,
		float innerR, float ringWidth,
		float wantVolume, float& outTakenVolume);

	void DepositVolumeGaussian(const DirectX::XMFLOAT2& centerXZ,
		float volume, float sigma);

	void UpdateBlob(float dt); // 球の飛行・着水処理

	// Gatherの状態
	bool  m_gatherActive = false;
	DirectX::XMFLOAT2 m_gatherCenter{ 0,0 };
	float m_gatherRadius = 0.18f; // 中心の半径
	float m_gatherRingW = 0.20f; // 吸い上げる輪の厚み
	float m_gatheredVolume = 0.0f; // 累積体積（m^3）
	float m_gatherRate = 0.015f; // 毎秒吸い上げ体積（m^3/s）の基準

	// 球
	WaterBlob m_blob;
};

