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
// GPU�֓n���p�[�e�B�N�����
// float4(x_ndc, y_ndc, radius_ndc, unused)
using ParticleSB = DirectX::XMFLOAT4;


// �t���X�N���[���g���C�A���O�����_
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

	// ���̕\���p
	VertexBuffer* m_MeshVertexBuffer = nullptr;  // �����b�V�����_�o�b�t�@
	IndexBuffer* m_MeshIndexBuffer = nullptr;    // �����b�V���C���f�b�N�X�o�b�t�@
	UINT m_IndexCount = 0;                        // �����b�V���̃C���f�b�N�X��
	VertexBuffer* m_InstanceBuffer = nullptr;    // �C���X�^���X�s��o�b�t�@

	// Metaball �p
	VertexBuffer* m_QuadVB = nullptr;
	RootSignature* m_MetaRootSig = nullptr;
	ParticlePipelineState* m_MetaPSO = nullptr;
	ID3D12Resource* m_ParticleSBGPU = nullptr;
	ID3D12Resource* m_ParticleSBUpload = nullptr;
	D3D12_GPU_DESCRIPTOR_HANDLE m_ParticleSB_SRV; // SRV �n���h��
	ComPtr<ID3D12Resource>   m_pMetaballCB;

	// Compute�p
	ComPtr<ID3D12Resource>          m_gpuInBuffer;			// in�FStructuredBuffer<Particle>
	ComPtr<ID3D12Resource>          m_gpuOutBuffer;			// out�FRWStructuredBuffer<Particle>
	ComPtr<ID3D12Resource>          m_uploadBuffer;			// �����f�[�^�]���p UploadHeap
	ComPtr<ID3D12DescriptorHeap>    m_srvUavHeap;			// SRV/UAV��o�^����q�[�v
	ConstantBuffer*					m_paramCB = nullptr;	// SPHParams�p�萔�o�b�t�@
	ComPtr<ID3D12Resource>			m_readbackBuffer;		// Readback�p�o�b�t�@

	ComputeRootSignature            m_computeRS;			// Compute�p���[�g�V�O�l�`��
	ComputePipelineState            m_computePSO;			// Compute�p�̃p�C�v���C���X�e�[�g

	std::unique_ptr<VertexBuffer> m_meshVB;					// ���b�V�����_�o�b�t�@
	std::unique_ptr<VertexBuffer> m_instanceVB;				// �C���X�^���X�s��o�b�t�@
	D3D12_INDEX_BUFFER_VIEW       m_indexBufferView;		// �����̃C���f�b�N�X�o�b�t�@�r���[
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

	// SPH�p�̊֐�
	void ComputeDensityPressure(std::vector<float>& densities, std::vector<float>& pressures);
	void ComputeForces(const std::vector<float>& densities, const std::vector<float>& pressures, std::vector<Vector3>& forces);
	void Integrate(const std::vector<Vector3>& forces);
};


