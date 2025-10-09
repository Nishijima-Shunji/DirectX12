#include "Particle.h"
#include "SharedStruct.h"
#include "RandomUtil.h"


Particle::Particle(Camera* cam) : camera(cam) {
	Init();
}

bool Particle::Init() {
	// 
	for (int i = 0; i < 500; ++i) {
		Point p;
		p.position = { RandFloat(-0.5f, 0.5f), RandFloat(-0.5f, 0.5f), RandFloat(-0.5f, 0.5f) };
		p.velocity = { 0, 0, 0 };
		m_Particles.push_back(p);
	}

	// 
	std::vector<ParticleVertex> vertices(m_Particles.size());
	for (size_t i = 0; i < m_Particles.size(); ++i) {
		vertices[i].position = m_Particles[i].position;
	}

        m_VertexBuffer = new VertexBuffer(sizeof(ParticleVertex) * vertices.size(), sizeof(ParticleVertex), vertices.data());
        if (!m_VertexBuffer) {
                printf("VertexBuffer失敗\n");
                return false;
        }

        // ※クアッドを三角形2枚で共有するためのインデックス（描画が「線」になる不具合対策）
        static const uint32_t indices[6] = { 0, 1, 2, 2, 1, 3 };
        m_IndexBuffer = new IndexBuffer(sizeof(indices), indices);
        if (!m_IndexBuffer || !m_IndexBuffer->IsValid()) {
                printf("IndexBuffer失敗\n");
                return false;
        }

	// 
	for (size_t i = 0; i < Engine::FRAME_BUFFER_COUNT; i++)
	{
		m_ConstantBuffer[i] = new ConstantBuffer(sizeof(Transform));
		if (!m_ConstantBuffer[i]->IsValid())
		{
			
			return false;
		}

		// 
		auto ptr = m_ConstantBuffer[i]->GetPtr<Transform>();
		ptr->World = DirectX::XMMatrixIdentity();
		ptr->View = DirectX::XMMatrixLookAtRH(camera->GetEyePos(), camera->GetTargetPos(), camera->GetUpward());
		ptr->Proj = DirectX::XMMatrixPerspectiveFovRH(camera->GetFov(), camera->GetAspect(), 0.3f, 1000.0f);
	}

	// 
	m_RootSignature = new RootSignature();
	if (!m_RootSignature->IsValid()) {
		printf("RootSignature失敗\n");
		return false;
	}

	// 
	m_PipelineState = new PipelineState();
	m_PipelineState->SetInputLayout(ParticleVertex::InputLayout);
	m_PipelineState->SetRootSignature(m_RootSignature->Get());
	m_PipelineState->SetVS(L"ParticleVS.cso");
	m_PipelineState->SetPS(L"ParticlePS.cso");


        // ※ビルボードを三角形リストで描くためトポロジを TRIANGLE に変更
        m_PipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
	if (!m_PipelineState->IsValid()) {
		printf("PipelineState失敗\n");
		return false;
	}
	return true;
}

void Particle::Update(float deltaTime) {
	UpdateParticles();
	UpdateVertexBuffer();

	auto ptr = m_ConstantBuffer[0]->GetPtr<Transform>();
	ptr->World = DirectX::XMMatrixIdentity();
	ptr->View = DirectX::XMMatrixLookAtRH(camera->GetEyePos(), camera->GetTargetPos(), camera->GetUpward());
	ptr->Proj = DirectX::XMMatrixPerspectiveFovRH(camera->GetFov(), camera->GetAspect(), 0.3f, 1000.0f);

	for (int i = 0; i < 10; ++i) {
		const auto& p = m_Particles[i];
	}
}

void Particle::Draw() {
	auto commandList = g_Engine->CommandList();

	int frameIndex = 0;

	commandList->SetGraphicsRootSignature(m_RootSignature->Get());  // RootSignature
	commandList->SetPipelineState(m_PipelineState->Get());          // PSO

	commandList->SetGraphicsRootConstantBufferView(0, m_ConstantBuffer[frameIndex]->GetAddress()); // CBV

        auto vbView = m_VertexBuffer->View();
        auto ibView = m_IndexBuffer->View();
        commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commandList->IASetVertexBuffers(0, 1, &vbView);
        commandList->IASetIndexBuffer(&ibView);

        // ※四隅を VS で生成するため、インデックス6個×粒子数で描画する
        commandList->DrawIndexedInstanced(6, static_cast<UINT>(m_Particles.size()), 0, 0, 0);
}


void Particle::UpdateParticles() {
	float dt = 0.016f;
	DirectX::XMFLOAT3 gravity = { 0.0f, -0.8f, 0.0f };

	for (auto& p : m_Particles) {
		// 速度更新
		p.velocity.x += gravity.x * dt;
		p.velocity.y += gravity.y * dt;
		p.velocity.z += gravity.z * dt;

		// 位置更新
		p.position.x += p.velocity.x * dt;
		p.position.y += p.velocity.y * dt;
		p.position.z += p.velocity.z * dt;

		// Y境界
		if (p.position.y < -1.0f) {
			p.position.y = -1.0f;
			p.velocity.y *= -0.6f;
		}

		// x境界
		if (p.position.x < -5.0f) {
			p.position.x = -5.0f;
			p.velocity.x *= -0.6f;
		}
		if (p.position.x > 5.0f) {
			p.position.x = 5.0f;
			p.velocity.x *= -0.6f;
		}

		// X境界
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

// vertex buffer 更新
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