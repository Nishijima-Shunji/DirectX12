#pragma once
#include "Object.h"
#include "Camera.h"
#include <directxMath.h>
#include <SimpleMath.h>
#include "Engine.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "RootSignature.h"
#include "ParticlePipelineState.h"
#include "ConstantBuffer.h"
#include "ComputeRootSignature.h" 
#include "ComputePipelineState.h"
#include <wrl.h> 

using namespace DirectX::SimpleMath;
// GPUへ渡すパーティクル情報
using ParticleSB = DirectX::XMFLOAT4;


// フルスクリーントライアングル頂点
struct FullscreenVertex {
	DirectX::XMFLOAT2 pos;
};

// パーティクルの位置と速度
struct Point {
	Vector3 position = {};
	Vector3 velocity = {};
	float radius = 0.1f; // パーティクルの半径
};

struct ParticleMeta {
	float x, y;    // スクリーン空間(0～1) or NDC(-1～1) 座標
	float r;       // 半径（スクリーン空間 or NDC いずれかに合わせる）
	float pad;     // 16 バイト境界合わせ
};

// パーティクルの物理パラメータ
struct SPHParams {
	float restDensity = 1000.0f;
	float particleMass = 1.0f;
	float viscosity = 0.1f;
	float stiffness = 200.0f;
	float radius = 0.1f;
	float timeStep = 0.016f;
	UINT particleCount = 80;
};

struct InstanceData {
	DirectX::XMFLOAT4 row0;
	DirectX::XMFLOAT4 row1;
	DirectX::XMFLOAT4 row2;
	DirectX::XMFLOAT4 row3;
};

class Particle : public Object
{
private:
	// パーティクルの情報
	std::vector<Point>		m_Particles;
	VertexBuffer* m_VertexBuffer = nullptr;
	RootSignature* m_RootSignature = nullptr;
	ParticlePipelineState* m_PipelineState = nullptr;
	ConstantBuffer* m_ConstantBuffer[Engine::FRAME_BUFFER_COUNT] = {};

	// 球体表示用
	VertexBuffer* m_MeshVertexBuffer = nullptr;  // 球メッシュ頂点バッファ
	IndexBuffer* m_MeshIndexBuffer = nullptr;    // 球メッシュインデックスバッファ
	UINT m_IndexCount = 0;                       // 球メッシュのインデックス数
	VertexBuffer* m_InstanceBuffer = nullptr;    // インスタンス行列バッファ

	// Metaball 用
	VertexBuffer* m_QuadVB = nullptr;
	RootSignature* m_MetaRootSig = nullptr;
	ParticlePipelineState* m_MetaPSO = nullptr;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_ParticleSBGPU = nullptr;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_ParticleSBUpload = nullptr;
	D3D12_GPU_DESCRIPTOR_HANDLE m_ParticleSB_SRV; // SRVハンドル
	// ComputeShader の outMeta UAV 用リソース
	Microsoft::WRL::ComPtr<ID3D12Resource> m_gpuMetaBuffer;
	// ComputeDescriptorHeap に登録する outMeta UAV の GPU ハンドル
	D3D12_GPU_DESCRIPTOR_HANDLE m_metaUAVHandle;
	// 描画用に描画ヒープへ登録する SRV の GPU ハンドル
	D3D12_GPU_DESCRIPTOR_HANDLE m_metaSRVHandle;

	// ComputeShader用
	// Compute 用ルートシグネチャ／PSO
	ComputeRootSignature            m_computeRS;
	ComputePipelineState            m_computePSO;

	// SRV/UAV 用 DescriptorHeap
	ComPtr<ID3D12DescriptorHeap>    m_srvUavHeap;
	ComPtr<ID3D12DescriptorHeap>	m_computeDescHeap;

	// SPHParams 用定数バッファ
	ConstantBuffer* m_paramCB = nullptr;

	// 粒子データの GPU バッファ (ping-pong)
	ComPtr<ID3D12Resource>			m_gpuInBuffer;
	ComPtr<ID3D12Resource>			m_gpuOutBuffer;
	D3D12_GPU_DESCRIPTOR_HANDLE		m_srvHandle{};	// t0
	D3D12_GPU_DESCRIPTOR_HANDLE		m_uavHandle{};	// u0


	// GPU待機用のフェンス
	ComPtr<ID3D12Fence> m_fence;
	static constexpr UINT FrameCount = 3;
	UINT64 m_fenceValue[FrameCount] = {};
	HANDLE m_fenceEvent = nullptr;
	UINT64 m_fenceCounter = 0;

	// Compute用のフェンス
	static constexpr UINT FrameCountCOM = 3;
	Microsoft::WRL::ComPtr<ID3D12CommandAllocator>	m_computeAllocators[FrameCount];
	Microsoft::WRL::ComPtr<ID3D12Fence>				m_computeFence;
	UINT64											m_computeFenceValues[FrameCount] = {};
	HANDLE											m_computeFenceEvent = nullptr;
	UINT64											m_computeFenceCounter = 0;
	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_computeCommandLists[FrameCount];

	Camera* camera;
	SPHParams m_SPHParams;

	int ParticleCount = 80;
public:
	Particle(Camera* cam);
	bool Init();
	void Update();
	void Draw();

	// 初期化関数
	bool InitParticle();
	bool InitMesh();
	bool InitMetaball();
	bool InitComputeShader();

	// 更新関数
	void UpdateParticles();

	// 描画関数
	void DrawMetaball();

	// SPH用の関数
	void ComputeDensityPressure(std::vector<float>& densities, std::vector<float>& pressures);
	void ComputeForces(const std::vector<float>& densities, const std::vector<float>& pressures, std::vector<Vector3>& forces);
	void Integrate(const std::vector<Vector3>& forces);
};


