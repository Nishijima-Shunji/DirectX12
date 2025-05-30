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

struct ParticleCS {
	DirectX::XMFLOAT3 position;
	DirectX::XMFLOAT3 velocity;
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
	ComPtr<ID3D12Resource>   m_pMetaballCB;

	// Compute用
	ComPtr<ID3D12Resource>          m_gpuInBuffer;			// in：StructuredBuffer<Particle>
	ComPtr<ID3D12Resource>          m_gpuOutBuffer;			// out：RWStructuredBuffer<Particle>
	ComPtr<ID3D12Resource>          m_uploadBuffer;			// 初期データ転送用 UploadHeap
	ComPtr<ID3D12DescriptorHeap>    m_srvUavHeap;			// SRV/UAVを登録するヒープ
	ConstantBuffer*					m_paramCB = nullptr;	// SPHParams用定数バッファ
	ComPtr<ID3D12Resource>			m_readbackBuffer;		// Readback用バッファ

	ComputeRootSignature            m_computeRS;			// Compute用ルートシグネチャ
	ComputePipelineState            m_computePSO;			// Compute用のパイプラインステート

	std::unique_ptr<VertexBuffer> m_meshVB;					// メッシュ頂点バッファ
	std::unique_ptr<VertexBuffer> m_instanceVB;				// インスタンス行列バッファ
	D3D12_INDEX_BUFFER_VIEW       m_indexBufferView;		// 既存のインデックスバッファビュー
	UINT                          m_vertexCount;
	UINT                          m_indexCount;



	Camera* camera;
	SPHParams m_SPHParams;

	int ParticleCount = 200;
public:
	Particle(Camera* cam);
	bool Init();
	void Update();
	void Draw();

	void UpdateParticles();
	void UpdateVertexBuffer();
	void UpdateInstanceBuffer();

	// SPH用の関数
	void ComputeDensityPressure(std::vector<float>& densities, std::vector<float>& pressures);
	void ComputeForces(const std::vector<float>& densities, const std::vector<float>& pressures, std::vector<Vector3>& forces);
	void Integrate(const std::vector<Vector3>& forces);
};


