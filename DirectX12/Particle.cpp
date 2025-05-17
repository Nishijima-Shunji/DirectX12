#include "Particle.h"
#include "SharedStruct.h"

#include <random>
// 指定した範囲 [min, max] のランダムな浮動小数を返す
float RandFloat(float min, float max)
{
	static std::random_device rd;
	static std::mt19937 gen(rd()); // 乱数生成器（メルセンヌ・ツイスタ）
	std::uniform_real_distribution<float> dist(min, max);
	return dist(gen);
}

Particle::Particle(Camera* cam) : camera (cam) {
	Init();
}

bool Particle::Init(){
	// 粒子生成
	for (int i = 0; i < 50000; ++i) {
		Point p;
		//p.position = { RandFloat(-1, 1), RandFloat(-1, 1), RandFloat(-1, 1) };
		p.position = { RandFloat(-0.5f, 0.5f), RandFloat(-0.5f, 0.5f), RandFloat(-0.5f, 0.5f) };
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

	// 行列変換用
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
		ptr->View = DirectX::XMMatrixLookAtRH(camera->GetEyePos() , camera->GetTargetPos(), camera->GetUpward());
		ptr->Proj = DirectX::XMMatrixPerspectiveFovRH(camera->GetFov(), camera->GetAspect(), 0.3f, 1000.0f);
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


	m_PipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT);
	if (!m_PipelineState->IsValid()) {
		printf("PipelineState作成に失敗\n");
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
//	int frameIndex = 0; // 今はフレーム同期無視して固定
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

	int frameIndex = 0; // 今はフレーム同期無視して固定

	commandList->SetGraphicsRootSignature(m_RootSignature->Get());  // 先に RootSignature
	commandList->SetPipelineState(m_PipelineState->Get());          // 次に PSO

	commandList->SetGraphicsRootConstantBufferView(0, m_ConstantBuffer[frameIndex]->GetAddress()); // その後に CBV

	auto vbView = m_VertexBuffer->View();
	commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
	commandList->IASetVertexBuffers(0, 1, &vbView);

	commandList->DrawInstanced(m_Particles.size(), 1, 0, 0);
}


void Particle::UpdateParticles() {
	float dt = 0.016f;
	DirectX::XMFLOAT3 gravity = { 0.0f, -0.8f, 0.0f };

	for (auto& p : m_Particles) {
		// 重力適用
		p.velocity.x += gravity.x * dt;
		p.velocity.y += gravity.y * dt;
		p.velocity.z += gravity.z * dt;

		// 位置更新
		p.position.x += p.velocity.x * dt;
		p.position.y += p.velocity.y * dt;
		p.position.z += p.velocity.z * dt;

		// 床 (Y方向)
		if (p.position.y < -1.0f) {
			p.position.y = -1.0f;
			p.velocity.y *= -0.6f;
		}

		// 壁 (X方向)
		if (p.position.x < -5.0f) {
			p.position.x = -5.0f;
			p.velocity.x *= -0.6f;
		}
		if (p.position.x > 5.0f) {
			p.position.x = 5.0f;
			p.velocity.x *= -0.6f;
		}

		// 壁 (Z方向)
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

// 描画前にGPUバッファに反映させる
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