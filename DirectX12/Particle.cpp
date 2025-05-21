#include "Particle.h"
#include "SharedStruct.h"
#include "SphereMeshGenerator.h"

#include <random>
// 指定した範囲 [min, max] のランダムな浮動小数を返す
float RandFloat(float min, float max)
{
	static std::random_device rd;
	static std::mt19937 gen(rd()); // 乱数生成器（メルセンヌ・ツイスタ）
	std::uniform_real_distribution<float> dist(min, max);
	return dist(gen);
}

// =============================================================================
// 物理の公式系
// =============================================================================

// Poly6カーネル
float Poly6Kernel(float r, float h) {
	if (r >= 0 && r <= h) {
		float x = (h * h - r * r);
		return (315.0f / (64.0f * DirectX::XM_PI * powf(h, 9))) * (x * x * x);
	}
	return 0.0f;
}

DirectX::XMFLOAT3 SpikyGradient(DirectX::XMFLOAT3 rij, float r, float h) {
	if (r > 0 && r <= h) {
		float coeff = -45.0f / (DirectX::XM_PI * powf(h, 6)) * (h - r) * (h - r);
		return { coeff * (rij.x / r), coeff * (rij.y / r), coeff * (rij.z / r) };
	}
	return { 0, 0, 0 };
}

float ViscosityLaplacian(float r, float h) {
	if (r >= 0 && r <= h) {
		return 45.0f / (DirectX::XM_PI * powf(h, 6)) * (h - r);
	}
	return 0.0f;
}

// 演算子オーバーロード
DirectX::XMFLOAT3 operator+(const DirectX::XMFLOAT3& a, const DirectX::XMFLOAT3& b) {
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}
DirectX::XMFLOAT3 operator-(const DirectX::XMFLOAT3& a, const DirectX::XMFLOAT3& b) {
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}
DirectX::XMFLOAT3 operator*(const DirectX::XMFLOAT3& a, float s) {
	return { a.x * s, a.y * s, a.z * s };
}
DirectX::XMFLOAT3& operator+=(DirectX::XMFLOAT3& a, const DirectX::XMFLOAT3& b) {
	a.x += b.x; a.y += b.y; a.z += b.z;
	return a;
}

// =============================================================================
// main
// =============================================================================
Particle::Particle(Camera* cam) : camera(cam) {
	Init();
}

bool Particle::Init() {
	// 粒子生成
	for (int i = 0; i < 100; ++i) {
		Point p;

		p.position = { RandFloat(-0.5f, 0.5f), RandFloat(-0.5f, 0.5f), RandFloat(-0.5f, 0.5f) };
		p.velocity = { 0, 0, 0 };
		m_Particles.push_back(p);
	}
	// 粒子のパラメーターの初期化
	m_SPHParams.restDensity		= 1000.0f;	//
	m_SPHParams.particleMass	= 1.0f;		// 重さ
	m_SPHParams.viscosity		= 5.0f;		// 粘性
	m_SPHParams.stiffness		= 1.0f;		// 剛性
	m_SPHParams.radius			= 0.1f;		//
	m_SPHParams.timeStep		= 0.016f;	//

	// 低ポリ球メッシュ生成
    auto mesh = CreateLowPolySphere(m_SPHParams.radius, 0);
    m_IndexCount = (UINT)mesh.indices.size();

    m_MeshVertexBuffer = new VertexBuffer(sizeof(Vertex) * mesh.vertices.size(), sizeof(Vertex), mesh.vertices.data());
	m_MeshIndexBuffer = new IndexBuffer(sizeof(uint32_t) * mesh.indices.size(), mesh.indices.data());

    if (!m_MeshVertexBuffer || !m_MeshIndexBuffer) {
        printf("Meshバッファ作成失敗\n");
        return false;
    }

	// 定数バッファをフレーム数分生成
	for (int i = 0; i < Engine::FRAME_BUFFER_COUNT; ++i)
	{
		m_ConstantBuffer[i] = new ConstantBuffer(sizeof(SPHParams));
		if (!m_ConstantBuffer[i] || !m_ConstantBuffer[i]->IsValid()) {
			printf("定数バッファ[%d]作成に失敗\n", i);
			return false;
		}

		// 初期SPHパラメータを書き込む
		memcpy(m_ConstantBuffer[i]->GetPtr(), &m_SPHParams, sizeof(SPHParams));
	}

    // インスタンスバッファ初期化（位置＋スケール行列）
    std::vector<DirectX::XMMATRIX> instanceMatrices(m_Particles.size(), DirectX::XMMatrixIdentity());
    m_InstanceBuffer = new VertexBuffer(sizeof(DirectX::XMMATRIX) * instanceMatrices.size(), sizeof(DirectX::XMMATRIX), instanceMatrices.data());

    if (!m_InstanceBuffer) {
        printf("インスタンスバッファ作成失敗\n");
        return false;
    }

	// ルートシグネチャ
	m_RootSignature = new RootSignature();
	if (!m_RootSignature->IsValid()) {
		printf("RootSignature作成に失敗\n");
		return false;
	}

	// パイプラインステート
	m_PipelineState = new ParticlePipelineState();
	m_PipelineState->SetInputLayout(ParticleVertex::ParticleInputLayout);
	m_PipelineState->SetRootSignature(m_RootSignature->Get());
	m_PipelineState->SetVS(L"../x64/Debug/ParticleVS.cso");
	m_PipelineState->SetPS(L"../x64/Debug/ParticlePS.cso");

	m_PipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
	if (!m_PipelineState->IsValid()) {
		printf("PipelineState作成に失敗\n");
		return false;
	}

	return true;
}

void Particle::Update() {
	UpdateParticles();
	//UpdateVertexBuffer();
	UpdateInstanceBuffer();

	auto ptr = m_ConstantBuffer[0]->GetPtr<Transform>();
	ptr->World = DirectX::XMMatrixIdentity();
	ptr->View = DirectX::XMMatrixLookAtRH(camera->GetEyePos(), camera->GetTargetPos(), camera->GetUpward());
	ptr->Proj = DirectX::XMMatrixPerspectiveFovRH(camera->GetFov(), camera->GetAspect(), 0.3f, 1000.0f);
}

void Particle::Draw() {
	auto commandList = g_Engine->CommandList();

	int frameIndex = 0;

	commandList->SetGraphicsRootSignature(m_RootSignature->Get());
	commandList->SetPipelineState(m_PipelineState->Get());
	commandList->SetGraphicsRootConstantBufferView(0, m_ConstantBuffer[frameIndex]->GetAddress());

	// 球メッシュ頂点・インデックスバッファセット
	auto vbView = m_MeshVertexBuffer->View();
	auto ibView = m_MeshIndexBuffer->View();
	commandList->IASetVertexBuffers(0, 1, &vbView);

	// インスタンスバッファはスロット1にセット（InputLayoutで指定）
	auto instView = m_InstanceBuffer->View();
	commandList->IASetVertexBuffers(1, 1, &instView);

	commandList->IASetIndexBuffer(&ibView);
	commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	// インスタンス描画
	commandList->DrawIndexedInstanced(m_IndexCount, (UINT)m_Particles.size(), 0, 0, 0);
}


void Particle::UpdateParticles() {
	int n = (int)m_Particles.size();

	std::vector<float> densities(n);
	std::vector<float> pressures(n);
	std::vector<Vector3> forces(n, { 0,0,0 });

	// 密度と圧力計算
	ComputeDensityPressure(densities, pressures);

	// 力計算
	ComputeForces(densities, pressures, forces);

	// 速度・位置更新
	Integrate(forces);
}

void Particle::ComputeDensityPressure(std::vector<float>& densities, std::vector<float>& pressures) {
	int n = (int)m_Particles.size();
	for (int i = 0; i < n; ++i) {
		float density = 0.0f;
		for (int j = 0; j < n; ++j) {
			Vector3 rij = m_Particles[i].position - m_Particles[j].position;
			float r = sqrtf(rij.x * rij.x + rij.y * rij.y + rij.z * rij.z);
			density += m_SPHParams.particleMass * Poly6Kernel(r, m_SPHParams.radius);
		}
		densities[i] = density;
		pressures[i] = m_SPHParams.stiffness * (densities[i] - m_SPHParams.restDensity);
	}
}

void Particle::ComputeForces(const std::vector<float>& densities, const std::vector<float>& pressures, std::vector<Vector3>& forces) {
	int n = (int)m_Particles.size();
	Vector3 gravity = { 0.0f, -9.8f, 0.0f };

	for (int i = 0; i < n; ++i) {
		Vector3 pressureForce = { 0, 0, 0 };
		Vector3 viscosityForce = { 0, 0, 0 };

		for (int j = 0; j < n; ++j) {
			if (i == j) continue;

			Vector3 rij = m_Particles[i].position - m_Particles[j].position;
			float r = sqrtf(rij.x * rij.x + rij.y * rij.y + rij.z * rij.z);

			if (r < m_SPHParams.radius && r > 0.0001f) {
				// 圧力力
				Vector3 grad = SpikyGradient(rij, r, m_SPHParams.radius);
				float pressureTerm = (pressures[i] + pressures[j]) / (2.0f * densities[j]);
				pressureForce += grad * (-m_SPHParams.particleMass * pressureTerm);

				// 粘性力
				Vector3 vij = m_Particles[j].velocity - m_Particles[i].velocity;
				float lap = ViscosityLaplacian(r, m_SPHParams.radius);
				viscosityForce += vij * (m_SPHParams.viscosity * m_SPHParams.particleMass * lap / densities[j]);
			}
		}
		// 合力 = 圧力 + 粘性 + 重力
		forces[i] = pressureForce + viscosityForce + gravity * densities[i];
	}
}

void Particle::Integrate(const std::vector<Vector3>& forces) {
	int n = (int)m_Particles.size();

	// 箱の境界サイズ（例）
	const float xmin = -1.0f, xmax = 1.0f;
	const float ymin = -1.0f, ymax = 5.0f;
	const float zmin = -1.0f, zmax = 1.0f;

	for (int i = 0; i < n; ++i) {
		Vector3 accel = forces[i] * (1.0f / (std::max)(m_SPHParams.restDensity, 0.0001f)); // 加速度

		m_Particles[i].velocity += accel * m_SPHParams.timeStep;
		m_Particles[i].position += m_Particles[i].velocity * m_SPHParams.timeStep;

		// X軸壁
		if (m_Particles[i].position.x < xmin) {
			m_Particles[i].position.x = xmin;
			m_Particles[i].velocity.x *= -0.1f;
		}
		if (m_Particles[i].position.x > xmax) {
			m_Particles[i].position.x = xmax;
			m_Particles[i].velocity.x *= -0.1f;
		}

		// Y軸壁（床と天井）
		if (m_Particles[i].position.y < ymin) {
			m_Particles[i].position.y = ymin;
			m_Particles[i].velocity.y *= -0.1f;
		}
		if (m_Particles[i].position.y > ymax) {
			m_Particles[i].position.y = ymax;
			m_Particles[i].velocity.y *= -0.1f;
		}

		// Z軸壁
		if (m_Particles[i].position.z < zmin) {
			m_Particles[i].position.z = zmin;
			m_Particles[i].velocity.z *= -0.1f;
		}
		if (m_Particles[i].position.z > zmax) {
			m_Particles[i].position.z = zmax;
			m_Particles[i].velocity.z *= -0.1f;
		}

		float maxSpeed = 3.0f;
		if (m_Particles[i].velocity.Length() > maxSpeed) {
			m_Particles[i].velocity.Normalize();
			m_Particles[i].velocity *= maxSpeed;
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

void Particle::UpdateInstanceBuffer()	 {
	std::vector<DirectX::XMMATRIX> matrices(m_Particles.size());

	for (size_t i = 0; i < m_Particles.size(); ++i) {
		auto pos = m_Particles[i].position;
		DirectX::XMMATRIX scale = DirectX::XMMatrixScaling(m_SPHParams.radius, m_SPHParams.radius, m_SPHParams.radius);
		DirectX::XMMATRIX trans = DirectX::XMMatrixTranslation(pos.x, pos.y, pos.z);
		//matrices[i] = trans * scale;
		DirectX::XMMATRIX mat = scale * trans;
		matrices[i] = mat;

		if (i == 0) {
			printf("matrix[0] = {%f, %f, %f, %f}\n", mat.r[3].m128_f32[0], mat.r[3].m128_f32[1], mat.r[3].m128_f32[2], mat.r[3].m128_f32[3]);
		}
	}

	//printf("pos = %f, %f, %f\n", m_Particles[0].position.x, m_Particles[0].position.y, m_Particles[0].position.z);
	
	void* ptr = nullptr;
	m_InstanceBuffer->GetResource()->Map(0, nullptr, &ptr);
	memcpy(ptr, matrices.data(), sizeof(DirectX::XMMATRIX) * matrices.size());
	m_InstanceBuffer->GetResource()->Unmap(0, nullptr);
}