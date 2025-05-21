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

using namespace DirectX::SimpleMath;

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


	Camera* camera;
	SPHParams m_SPHParams;
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


