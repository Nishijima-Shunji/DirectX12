#include "Game.h"
#include "SceneManager.h"
#include "App.h"
#include "SharedStruct.h"
#include "ConstantBuffer.h"
#include "GameScene.h"

#include <windows.h>

#include <random>
// 指定した範囲 [min, max] のランダムな浮動小数を返す
float RandFloat(float min, float max)
{
	static std::random_device rd;
	static std::mt19937 gen(rd()); // 乱数生成器（メルセンヌ・ツイスタ）
	std::uniform_real_distribution<float> dist(min, max);
	return dist(gen);
}

GameScene::GameScene(Game* game) : BaseScene(game) {
	printf("GameScene生成\n");
	Init();
}

GameScene::~GameScene() {
	printf("GameScene破棄\n");
}

bool GameScene::Init() {
	// 粒子生成
	for (int i = 0; i < 1000; ++i) {
		Particle p;
		p.position = { RandFloat(-1, 1), RandFloat(-1, 1), RandFloat(-1, 1) };
		p.velocity = { 0, 0, 0 };
		m_Particles.push_back(p);
	}

	// 頂点バッファ生成
	std::vector<ParticleVertex> vertices(m_Particles.size());
	for (size_t i = 0; i < m_Particles.size(); ++i) {
		vertices[i].position = m_Particles[i].position;
	}

	m_VertexBuffer = new VertexBuffer(sizeof(ParticleVertex) * vertices.size(), sizeof(ParticleVertex), vertices.data());
	if (!m_VertexBuffer) {
		printf("VertexBuffer作成に失敗\n");
		return false;
	}

	// 行列変換
	eyePos = DirectX::XMVectorSet(0.0f, 0.0f, 5.0f, 0.0f);										// 視点の位置
	targetPos = DirectX::XMVectorZero();														// 視点を向ける座標
	upward = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);										// 上方向を表すベクトル
	fov = DirectX::XMConvertToRadians(37.5);													// 視野角
	auto aspect = static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT);			// アスペクト比

	for (size_t i = 0; i < Engine::FRAME_BUFFER_COUNT; i++)
	{
		m_ConstantBuffer[i] = new ConstantBuffer(sizeof(Transform));
		if (!m_ConstantBuffer[i]->IsValid())
		{
			printf("変換行列用定数バッファの生成に失敗\n");
			return false;
		}

		// 変換行列の登録
		auto ptr = m_ConstantBuffer[i]->GetPtr<Transform>();
		ptr->World = DirectX::XMMatrixIdentity();
		ptr->View = DirectX::XMMatrixLookAtRH(eyePos, targetPos, upward);
		ptr->Proj = DirectX::XMMatrixPerspectiveFovRH(fov, aspect, 0.3f, 1000.0f);
	}

	// ルートシグネチャ
	m_RootSignature = new RootSignature();
	if (!m_RootSignature->IsValid()) {
		printf("RootSignature作成に失敗\n");
		return false;
	}

	// パイプラインステート
	m_PipelineState = new PipelineState();
	m_PipelineState->SetInputLayout(ParticleVertex::InputLayout);
	m_PipelineState->SetRootSignature(m_RootSignature->Get());
	m_PipelineState->SetVS(L"../x64/Debug/ParticleVS.cso");
	m_PipelineState->SetPS(L"../x64/Debug/ParticlePS.cso");
	m_PipelineState->Create();
	if (!m_PipelineState->IsValid()) {
		printf("PipelineState作成に失敗\n");
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
		// 重力適用
		p.velocity.x += gravity.x * dt;
		p.velocity.y += gravity.y * dt;
		p.velocity.z += gravity.z * dt;

		// 位置更新
		p.position.x += p.velocity.x * dt;
		p.position.y += p.velocity.y * dt;
		p.position.z += p.velocity.z * dt;

		// 床で反発
		if (p.position.y < -1.0f) {
			p.position.y = -1.0f;
			p.velocity.y *= -0.6f;
		}
	}

	// 頂点バッファを更新（DirectX12ではアップロードバッファかリソースコピー）
}