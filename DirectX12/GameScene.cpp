#include "Game.h"
#include "SceneManager.h"
#include "App.h"
#include "SharedStruct.h"
#include "ConstantBuffer.h"
#include "GameScene.h"

#include <windows.h>

#include <random>
// �w�肵���͈� [min, max] �̃����_���ȕ���������Ԃ�
float RandFloat(float min, float max)
{
	static std::random_device rd;
	static std::mt19937 gen(rd()); // ����������i�����Z���k�E�c�C�X�^�j
	std::uniform_real_distribution<float> dist(min, max);
	return dist(gen);
}

GameScene::GameScene(Game* game) : BaseScene(game) {
	printf("GameScene����\n");
	Init();
}

GameScene::~GameScene() {
	printf("GameScene�j��\n");
}

bool GameScene::Init() {
	// ���q����
	for (int i = 0; i < 1000; ++i) {
		Particle p;
		p.position = { RandFloat(-1, 1), RandFloat(-1, 1), RandFloat(-1, 1) };
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

	// �s��ϊ�
	eyePos = DirectX::XMVectorSet(0.0f, 0.0f, 5.0f, 0.0f);										// ���_�̈ʒu
	targetPos = DirectX::XMVectorZero();														// ���_����������W
	upward = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);										// �������\���x�N�g��
	fov = DirectX::XMConvertToRadians(37.5);													// ����p
	auto aspect = static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT);			// �A�X�y�N�g��

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
		ptr->View = DirectX::XMMatrixLookAtRH(eyePos, targetPos, upward);
		ptr->Proj = DirectX::XMMatrixPerspectiveFovRH(fov, aspect, 0.3f, 1000.0f);
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
	m_PipelineState->Create();
	if (!m_PipelineState->IsValid()) {
		printf("PipelineState�쐬�Ɏ��s\n");
		return false;
	}

	return true;
}

void GameScene::Update() {
	UpdateParticles();

	if (GetAsyncKeyState(VK_UP)) {
		m_game->ChangeScene("Title");
	}
}

void GameScene::Draw() {
	auto commandList = g_Engine->CommandList();

	commandList->SetGraphicsRootSignature(m_RootSignature->Get());
	commandList->SetPipelineState(m_PipelineState->Get());

	auto vbView = m_VertexBuffer->View();
	commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);

	commandList->IASetVertexBuffers(0, 1, &vbView);

	commandList->DrawInstanced(m_Particles.size(), 1, 0, 0);
}

void GameScene::UpdateParticles() {
	float dt = 0.016f;
	DirectX::XMFLOAT3 gravity = { 0.0f, -9.8f, 0.0f };

	for (auto& p : m_Particles) {
		// �d�͓K�p
		p.velocity.x += gravity.x * dt;
		p.velocity.y += gravity.y * dt;
		p.velocity.z += gravity.z * dt;

		// �ʒu�X�V
		p.position.x += p.velocity.x * dt;
		p.position.y += p.velocity.y * dt;
		p.position.z += p.velocity.z * dt;

		// ���Ŕ���
		if (p.position.y < -1.0f) {
			p.position.y = -1.0f;
			p.velocity.y *= -0.6f;
		}
	}

	// ���_�o�b�t�@���X�V�iDirectX12�ł̓A�b�v���[�h�o�b�t�@�����\�[�X�R�s�[�j
}