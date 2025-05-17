#include "Particle.h"
#include "SharedStruct.h"

#include <random>
// �w�肵���͈� [min, max] �̃����_���ȕ���������Ԃ�
float RandFloat(float min, float max)
{
	static std::random_device rd;
	static std::mt19937 gen(rd()); // ����������i�����Z���k�E�c�C�X�^�j
	std::uniform_real_distribution<float> dist(min, max);
	return dist(gen);
}

Particle::Particle(Camera* cam) : camera (cam) {
	Init();
}

bool Particle::Init(){
	// ���q����
	for (int i = 0; i < 50000; ++i) {
		Point p;
		//p.position = { RandFloat(-1, 1), RandFloat(-1, 1), RandFloat(-1, 1) };
		p.position = { RandFloat(-0.5f, 0.5f), RandFloat(-0.5f, 0.5f), RandFloat(-0.5f, 0.5f) };
		p.velocity = { 0, 0, 0 };
		m_Particles.push_back(p);
	}

	// ���_�o�b�t�@����
	std::vector<ParticleVertex> vertices(m_Particles.size());
	for (size_t i = 0; i < m_Particles.size(); ++i) {
		vertices[i].position = m_Particles[i].position;
	}

	m_VertexBuffer = new VertexBuffer(sizeof(ParticleVertex) * vertices.size(), sizeof(ParticleVertex), vertices.data());
	if (!m_VertexBuffer) {
		printf("VertexBuffer�쐬�Ɏ��s\n");
		return false;
	}

	// �s��ϊ��p
	for (size_t i = 0; i < Engine::FRAME_BUFFER_COUNT; i++)
	{
		m_ConstantBuffer[i] = new ConstantBuffer(sizeof(Transform));
		if (!m_ConstantBuffer[i]->IsValid())
		{
			printf("�ϊ��s��p�萔�o�b�t�@�̐����Ɏ��s\n");
			return false;
		}

		// �ϊ��s��̓o�^
		auto ptr = m_ConstantBuffer[i]->GetPtr<Transform>();
		ptr->World = DirectX::XMMatrixIdentity();
		ptr->View = DirectX::XMMatrixLookAtRH(camera->GetEyePos() , camera->GetTargetPos(), camera->GetUpward());
		ptr->Proj = DirectX::XMMatrixPerspectiveFovRH(camera->GetFov(), camera->GetAspect(), 0.3f, 1000.0f);
	}

	// ���[�g�V�O�l�`��
	m_RootSignature = new RootSignature();
	if (!m_RootSignature->IsValid()) {
		printf("RootSignature�쐬�Ɏ��s\n");
		return false;
	}

	// �p�C�v���C���X�e�[�g
	m_PipelineState = new PipelineState();
	m_PipelineState->SetInputLayout(ParticleVertex::InputLayout);
	m_PipelineState->SetRootSignature(m_RootSignature->Get());
	m_PipelineState->SetVS(L"../x64/Debug/ParticleVS.cso");
	m_PipelineState->SetPS(L"../x64/Debug/ParticlePS.cso");


	m_PipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT);
	if (!m_PipelineState->IsValid()) {
		printf("PipelineState�쐬�Ɏ��s\n");
		return false;
	}
	return true;
}

void Particle::Update() {
	UpdateParticles();
	UpdateVertexBuffer();

	auto ptr = m_ConstantBuffer[0]->GetPtr<Transform>();
	ptr->World = DirectX::XMMatrixIdentity();
	ptr->View = DirectX::XMMatrixLookAtRH(camera->GetEyePos(), camera->GetTargetPos(), camera->GetUpward());
	ptr->Proj = DirectX::XMMatrixPerspectiveFovRH(camera->GetFov(), camera->GetAspect(), 0.3f, 1000.0f);

	for (int i = 0; i < 10; ++i) {
		const auto& p = m_Particles[i];
		printf("p%d: pos=(%.3f, %.3f, %.3f), vel=(%.6f, %.6f, %.6f)\n",
			i, p.position.x, p.position.y, p.position.z,
			p.velocity.x, p.velocity.y, p.velocity.z);
	}
}

//void Particle::Draw() {
//
//	auto commandList = g_Engine->CommandList();
//
//	int frameIndex = 0; // ���̓t���[�������������ČŒ�
//	commandList->SetGraphicsRootConstantBufferView(0, m_ConstantBuffer[frameIndex]->GetAddress());
//
//	commandList->SetGraphicsRootSignature(m_RootSignature->Get());
//	commandList->SetPipelineState(m_PipelineState->Get());
//
//	auto vbView = m_VertexBuffer->View();
//	commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
//
//	commandList->IASetVertexBuffers(0, 1, &vbView);
//
//	commandList->DrawInstanced(m_Particles.size(), 1, 0, 0);
//}
void Particle::Draw() {
	auto commandList = g_Engine->CommandList();

	int frameIndex = 0; // ���̓t���[�������������ČŒ�

	commandList->SetGraphicsRootSignature(m_RootSignature->Get());  // ��� RootSignature
	commandList->SetPipelineState(m_PipelineState->Get());          // ���� PSO

	commandList->SetGraphicsRootConstantBufferView(0, m_ConstantBuffer[frameIndex]->GetAddress()); // ���̌�� CBV

	auto vbView = m_VertexBuffer->View();
	commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
	commandList->IASetVertexBuffers(0, 1, &vbView);

	commandList->DrawInstanced(m_Particles.size(), 1, 0, 0);
}


void Particle::UpdateParticles() {
	float dt = 0.016f;
	DirectX::XMFLOAT3 gravity = { 0.0f, -0.8f, 0.0f };

	for (auto& p : m_Particles) {
		// �d�͓K�p
		p.velocity.x += gravity.x * dt;
		p.velocity.y += gravity.y * dt;
		p.velocity.z += gravity.z * dt;

		// �ʒu�X�V
		p.position.x += p.velocity.x * dt;
		p.position.y += p.velocity.y * dt;
		p.position.z += p.velocity.z * dt;

		// �� (Y����)
		if (p.position.y < -1.0f) {
			p.position.y = -1.0f;
			p.velocity.y *= -0.6f;
		}

		// �� (X����)
		if (p.position.x < -5.0f) {
			p.position.x = -5.0f;
			p.velocity.x *= -0.6f;
		}
		if (p.position.x > 5.0f) {
			p.position.x = 5.0f;
			p.velocity.x *= -0.6f;
		}

		// �� (Z����)
		if (p.position.z < -5.0f) {
			p.position.z = -5.0f;
			p.velocity.z *= -0.6f;
		}
		if (p.position.z > 5.0f) {
			p.position.z = 5.0f;
			p.velocity.z *= -0.6f;
		}

	}
}

// �`��O��GPU�o�b�t�@�ɔ��f������
void Particle::UpdateVertexBuffer() {
	std::vector<ParticleVertex> vertices(m_Particles.size());
	for (size_t i = 0; i < m_Particles.size(); ++i) {
		vertices[i].position = m_Particles[i].position;
	}

	void* ptr = nullptr;
	m_VertexBuffer->GetResource()->Map(0, nullptr, &ptr);
	memcpy(ptr, vertices.data(), sizeof(ParticleVertex) * vertices.size());
	m_VertexBuffer->GetResource()->Unmap(0, nullptr);
}