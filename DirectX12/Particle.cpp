#include "Particle.h"
#include "SharedStruct.h"
#include "RandomUtil.h"
#include "SphereMeshGenerator.h"


Particle::Particle(Camera* cam) : camera(cam) {
	Init();
}

bool Particle::Init() {
	// 粒子の初期位置と速度を用意（球描画へ切り替えても挙動を維持するため）
	for (int i = 0; i < 500; ++i) {
		Point p;
		p.position = { RandFloat(-0.5f, 0.5f), RandFloat(-0.5f, 0.5f), RandFloat(-0.5f, 0.5f) };
		p.velocity = { 0, 0, 0 };
		m_Particles.push_back(p);
	}

	// 粒子を球メッシュで描画するための頂点データを生成
	MeshData sphereMesh = CreateLowPolySphere(1.0f, 0);
	std::vector<ParticleVertex> sphereVertices(sphereMesh.vertices.size());
	for (size_t i = 0; i < sphereMesh.vertices.size(); ++i) {
		sphereVertices[i].position = sphereMesh.vertices[i].Position;
		sphereVertices[i].normal = sphereMesh.vertices[i].Normal;
	}
	m_SphereVertexBuffer = new VertexBuffer(sizeof(ParticleVertex) * sphereVertices.size(), sizeof(ParticleVertex), sphereVertices.data());
	if (!m_SphereVertexBuffer || !m_SphereVertexBuffer->IsValid()) {
		printf("SphereVertexBuffer失敗\n");
		return false;
	}

	// 球メッシュ用のインデックスバッファを準備（球描画へ切り替える理由を明示）
	m_SphereIndexBuffer = new IndexBuffer(sizeof(uint32_t) * sphereMesh.indices.size(), sphereMesh.indices.data());
	if (!m_SphereIndexBuffer || !m_SphereIndexBuffer->IsValid()) {
		printf("SphereIndexBuffer失敗\n");
		return false;
	}
	m_SphereIndexCount = static_cast<UINT>(sphereMesh.indices.size());

	// 粒子インスタンス用バッファ（位置と半径）を作成して球描画へ対応
	std::vector<ParticleInstance> instances(m_Particles.size());
	for (size_t i = 0; i < m_Particles.size(); ++i) {
		instances[i].position = m_Particles[i].position;
		instances[i].radius = m_ParticleRadius;
	}
	m_InstanceBuffer = new VertexBuffer(sizeof(ParticleInstance) * instances.size(), sizeof(ParticleInstance), instances.data());
	if (!m_InstanceBuffer || !m_InstanceBuffer->IsValid()) {
		printf("InstanceBuffer失敗\n");
		return false;
	}

	// ビュー行列などの更新頻度が高い定数バッファを用意
	for (size_t i = 0; i < Engine::FRAME_BUFFER_COUNT; i++)
	{
		m_ConstantBuffer[i] = new ConstantBuffer(sizeof(Transform));
		if (!m_ConstantBuffer[i]->IsValid())
		{
			return false;
		}

		// HLSL 側では行列が列優先で格納されるため、CPU 側の行列を転置して合わせる
		auto ptr = m_ConstantBuffer[i]->GetPtr<Transform>();
		ptr->World = DirectX::XMMatrixTranspose(DirectX::XMMatrixIdentity());
		ptr->View = DirectX::XMMatrixTranspose(DirectX::XMMatrixLookAtRH(camera->GetEyePos(), camera->GetTargetPos(), camera->GetUpward()));
		ptr->Proj = DirectX::XMMatrixTranspose(DirectX::XMMatrixPerspectiveFovRH(camera->GetFov(), camera->GetAspect(), 0.3f, 1000.0f));
	}

	// 球を描画するためのルートシグネチャを用意
	m_RootSignature = new RootSignature();
	if (!m_RootSignature->IsValid()) {
		printf("RootSignature失敗\n");
		return false;
	}

	// 球描画に合わせたパイプラインステートを設定（入力レイアウトは球+インスタンス）
	m_PipelineState = new PipelineState();
	m_PipelineState->SetInputLayout(ParticleVertex::InputLayout);
	m_PipelineState->SetRootSignature(m_RootSignature->Get());
	m_PipelineState->SetVS(L"ParticleVS.cso");
	m_PipelineState->SetPS(L"ParticlePS.cso");

	m_PipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
	if (!m_PipelineState->IsValid()) {
		printf("PipelineState失敗\n");
		return false;
	}
	return true;
}



void Particle::Update(float deltaTime) {
	UpdateParticles();
	UpdateInstanceBuffer();

	auto ptr = m_ConstantBuffer[0]->GetPtr<Transform>();
	// HLSL 側では行列が列優先で格納されるため、CPU 側の行列を転置して合わせる
	ptr->World = DirectX::XMMatrixTranspose(DirectX::XMMatrixIdentity());
	ptr->View = DirectX::XMMatrixTranspose(DirectX::XMMatrixLookAtRH(camera->GetEyePos(), camera->GetTargetPos(), camera->GetUpward()));
	ptr->Proj = DirectX::XMMatrixTranspose(DirectX::XMMatrixPerspectiveFovRH(camera->GetFov(), camera->GetAspect(), 0.3f, 1000.0f));

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

	D3D12_VERTEX_BUFFER_VIEW vbViews[2] = {
		m_SphereVertexBuffer->View(),
		m_InstanceBuffer->View()
	};
	D3D12_INDEX_BUFFER_VIEW ibView = m_SphereIndexBuffer->View();
	commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	commandList->IASetVertexBuffers(0, 2, vbViews);
	commandList->IASetIndexBuffer(&ibView);

	commandList->DrawIndexedInstanced(m_SphereIndexCount, static_cast<UINT>(m_Particles.size()), 0, 0, 0);
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

// インスタンスバッファ更新（球描画の半径を毎フレーム反映するため）
void Particle::UpdateInstanceBuffer() {
	std::vector<ParticleInstance> instances(m_Particles.size());
	for (size_t i = 0; i < m_Particles.size(); ++i) {
		instances[i].position = m_Particles[i].position;
		instances[i].radius = m_ParticleRadius;
	}

	void* ptr = nullptr;
	m_InstanceBuffer->GetResource()->Map(0, nullptr, &ptr);
	memcpy(ptr, instances.data(), sizeof(ParticleInstance) * instances.size());
	m_InstanceBuffer->GetResource()->Unmap(0, nullptr);
}
