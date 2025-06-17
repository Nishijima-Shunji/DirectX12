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
// float4(x_ndc, y_ndc, radius_ndc, unused)
using ParticleSB = DirectX::XMFLOAT4;


// フルスクリーントライアングル頂点
struct FullscreenVertex {
	DirectX::XMFLOAT2 pos;
};

struct Point {
	Vector3 position = {};
	Vector3 velocity = {};
};

struct SPHParams {
	float restDensity = 1000.0f;
	float particleMass = 1.0f;
	float viscosity = 0.1f;
	float stiffness = 200.0f;
	float radius = 0.1f;
	float timeStep = 0.016f;
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
	std::vector<Point>		m_Particles;
	VertexBuffer* m_VertexBuffer = nullptr;
	RootSignature* m_RootSignature = nullptr;
	ParticlePipelineState* m_PipelineState = nullptr;
	ConstantBuffer* m_ConstantBuffer[Engine::FRAME_BUFFER_COUNT] = {};

	// 球体表示用
	VertexBuffer* m_MeshVertexBuffer = nullptr;  // 球メッシュ頂点バッファ
	IndexBuffer* m_MeshIndexBuffer = nullptr;    // 球メッシュインデックスバッファ
	UINT m_IndexCount = 0;                        // 球メッシュのインデックス数
	VertexBuffer* m_InstanceBuffer = nullptr;    // インスタンス行列バッファ

	// Metaball 用
	VertexBuffer* m_QuadVB = nullptr;
	RootSignature* m_MetaRootSig = nullptr;
	ParticlePipelineState* m_MetaPSO = nullptr;
	ID3D12Resource* m_ParticleSBGPU = nullptr;
	ID3D12Resource* m_ParticleSBUpload = nullptr;
	D3D12_GPU_DESCRIPTOR_HANDLE m_ParticleSB_SRV; // SRV ハンドル

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
	ComPtr<ID3D12Resource>          m_gpuInBuffer;
	ComPtr<ID3D12Resource>          m_gpuOutBuffer;

	// GPU待機用のフェンス
	ComPtr<ID3D12Fence> m_fence;
	static constexpr UINT FrameCount = 3;
	UINT64 m_fenceValue[FrameCount] = {};
	HANDLE m_fenceEvent = nullptr;



	Camera* camera;
	SPHParams m_SPHParams;

	int ParticleCount = 200;
public:
	Particle(Camera* cam);
	bool Init();
	void Update();
	void Draw();


	bool InitParticle();
	bool InitMesh();
	bool InitMetaball();
	bool InitComputeShader();

	void UpdateParticles();
	void UpdateVertexBuffer();
	void UpdateInstanceBuffer();

	// SPH用の関数
	void ComputeDensityPressure(std::vector<float>& densities, std::vector<float>& pressures);
	void ComputeForces(const std::vector<float>& densities, const std::vector<float>& pressures, std::vector<Vector3>& forces);
	void Integrate(const std::vector<Vector3>& forces);
};


