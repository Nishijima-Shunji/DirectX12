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
	// --- CPU側のm_Particlesを生成する ---
	m_Particles.reserve(ParticleCount);
	for (int i = 0; i < ParticleCount; ++i) {
		Point p;
		p.position = { RandFloat(-0.5f,0.5f), RandFloat(-0.5f,0.5f), RandFloat(-0.5f,0.5f) };
		p.velocity = { 0,0,0 };
		m_Particles.push_back(p);
	}

	// --- GPUバッファを２つ（ping-pong）作成 ---
	UINT64 sz = sizeof(Point) * ParticleCount;

	// DefaultHeap 用のヒーププロパティ／リソース記述子をローカル変数に
	CD3DX12_HEAP_PROPERTIES heapDefault(D3D12_HEAP_TYPE_DEFAULT);
	CD3DX12_RESOURCE_DESC   descDefault =
		CD3DX12_RESOURCE_DESC::Buffer(sz, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

	// inBuffer
	HRESULT hr = g_Engine->Device()->CreateCommittedResource(
		&heapDefault,                     // ここはローカル変数のアドレス
		D3D12_HEAP_FLAG_NONE,
		&descDefault,                     // これもローカル変数
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&m_gpuInBuffer)
	);
	if (FAILED(hr)) return false;

	// outBuffer
	hr = g_Engine->Device()->CreateCommittedResource(
		&heapDefault,
		D3D12_HEAP_FLAG_NONE,
		&descDefault,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		nullptr,
		IID_PPV_ARGS(&m_gpuOutBuffer)
	);
	if (FAILED(hr)) return false;

	// UploadHeap 用も同様にローカル変数で用意
	CD3DX12_HEAP_PROPERTIES heapUpload(D3D12_HEAP_TYPE_UPLOAD);
	CD3DX12_RESOURCE_DESC   descUpload = CD3DX12_RESOURCE_DESC::Buffer(sz);

	hr = g_Engine->Device()->CreateCommittedResource(
		&heapUpload,
		D3D12_HEAP_FLAG_NONE,
		&descUpload,
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&m_uploadBuffer)
	);
	if (FAILED(hr)) return false;

	// ローカル変数としてヒーププロパティを用意
	CD3DX12_HEAP_PROPERTIES heapReadback(D3D12_HEAP_TYPE_READBACK);

	// リソース記述子もローカル変数
	CD3DX12_RESOURCE_DESC readbackDesc = CD3DX12_RESOURCE_DESC::Buffer(sz);

	// 生成呼び出し
	hr = g_Engine->Device()->CreateCommittedResource(
		&heapReadback,              // ← 一時ではなくローカル変数
		D3D12_HEAP_FLAG_NONE,
		&readbackDesc,              // ← こちらもローカル変数
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&m_readbackBuffer)
	);
	if (FAILED(hr)) {
		// エラーハンドリング
		return false;
	}

	// map → memcpy → Unmap
	void* ptr = nullptr;
	m_uploadBuffer->Map(0, nullptr, &ptr);
	memcpy(ptr, m_Particles.data(), (size_t)sz);
	m_uploadBuffer->Unmap(0, nullptr);

	// CopyResourceでDefaultHeap(in)に転送
	auto cmd = g_Engine->CommandList();
	cmd->CopyResource(m_gpuInBuffer.Get(), m_uploadBuffer.Get());
	// バリアを貼って状態遷移
	D3D12_RESOURCE_BARRIER barrierTransition =
		CD3DX12_RESOURCE_BARRIER::Transition(
			m_gpuInBuffer.Get(),
			D3D12_RESOURCE_STATE_COPY_DEST,
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
		);
	cmd->ResourceBarrier(1, &barrierTransition);

	// --- 定数バッファ(SPHParams)の生成・書き込み ---
	{
		m_paramCB = new ConstantBuffer(sizeof(SPHParams));
		// 初期値を書き込む
		memcpy(m_paramCB->GetPtr(), &m_SPHParams, sizeof(SPHParams));
	}

	// --- SRV/UAV用DescriptorHeapの作成・登録 ---
	{
		D3D12_DESCRIPTOR_HEAP_DESC hdesc = {};
		hdesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		hdesc.NodeMask = 0;
		hdesc.NumDescriptors = 2;  // SRV, UAV
		hdesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		g_Engine->Device()->CreateDescriptorHeap(&hdesc, IID_PPV_ARGS(&m_srvUavHeap));

		auto cpu = m_srvUavHeap->GetCPUDescriptorHandleForHeapStart();
		UINT stride = g_Engine->Device()->GetDescriptorHandleIncrementSize(
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		// SRV(inBuffer)
		D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
		srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srv.Format = DXGI_FORMAT_UNKNOWN;
		srv.Buffer.NumElements = ParticleCount;
		srv.Buffer.StructureByteStride = sizeof(Point);
		srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		g_Engine->Device()->CreateShaderResourceView(
			m_gpuInBuffer.Get(), &srv, cpu);

		// UAV(outBuffer)
		cpu.ptr += stride;
		D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
		uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		uav.Buffer.NumElements = ParticleCount;
		uav.Buffer.StructureByteStride = sizeof(Point);
		g_Engine->Device()->CreateUnorderedAccessView(
			m_gpuOutBuffer.Get(), nullptr, &uav, cpu);
	}

	// --- InstanceBufferの生成 ---

	// インスタンス行列を Identity で初期化
	std::vector<DirectX::XMMATRIX> instanceMatrices(
		m_Particles.size(),
		DirectX::XMMatrixIdentity()
	);

	// インスタンスバッファ生成
	m_InstanceBuffer = new VertexBuffer(
		sizeof(DirectX::XMMATRIX) * instanceMatrices.size(),
		sizeof(DirectX::XMMATRIX),
		instanceMatrices.data()
	);

	if (!m_InstanceBuffer) {
		printf("インスタンスバッファ作成失敗\n");
		return false;
	}


	// --- Compute用ルートシグネチャ/PSOの初期化 ---
	m_computeRS.InitForSPH();                                            // ComputeRootSignature.cpp
	m_computePSO.SetRootSignature(m_computeRS.Get());                    // ComputePipelineState.cpp
	m_computePSO.SetCS(L"..\\x64\\Debug\\ParticleComputeShader.cso");						 // あらかじめビルドしたCSOファイル
	m_computePSO.Create();


	// --- ビューバッファの生成 ---
	// 頂点バッファのビュー
	m_vertexBufferView.BufferLocation = m_vertexBuffer->GetGPUVirtualAddress();
	m_vertexBufferView.SizeInBytes = static_cast<UINT>(sizeof(Vertex) * m_vertexCount);
	m_vertexBufferView.StrideInBytes = sizeof(Vertex);

	// インデックスバッファのビュー
	m_indexBufferView.BufferLocation = m_indexBuffer->GetGPUVirtualAddress();
	m_indexBufferView.SizeInBytes = static_cast<UINT>(sizeof(uint16_t) * m_indexCount);
	m_indexBufferView.Format = DXGI_FORMAT_R16_UINT;  // または R32_UINT



	// --- グラフィックス描画用のルートシグネチャ生成 ---
	m_RootSignature = new RootSignature();
	if (!m_RootSignature->IsValid()) {
		printf("RootSignature 作成に失敗\n");
		return false;
	}

	// --- グラフィックス用 PSO の設定・生成 ---
	m_PipelineState = new ParticlePipelineState();
	// 頂点入力レイアウトは既定のものを使うか、必要に応じて独自定義
	m_PipelineState->SetInputLayout(ParticleVertex::ParticleInputLayout);  // :contentReference[oaicite:0]{index=0}
	m_PipelineState->SetRootSignature(m_RootSignature->Get());
	m_PipelineState->SetVS(L"..\\x64\\Debug\\ParticleVS.cso");
	m_PipelineState->SetPS(L"..\\x64\\Debug\\ParticlePS.cso");
	m_PipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);        // :contentReference[oaicite:1]{index=1}

	if (!m_PipelineState->IsValid()) {
		printf("Graphics PipelineState 作成に失敗\n");
		return false;
	}

	return true;
}

void Particle::Update() {
	// パラメータが変わっていれば定数バッファを更新
	memcpy(m_paramCB->GetPtr(), &m_SPHParams, sizeof(SPHParams));

	// Compute Dispatch
	auto cmd = g_Engine->CommandList();
	cmd->SetComputeRootSignature(m_computeRS.Get());
	cmd->SetPipelineState(m_computePSO.Get());
	ID3D12DescriptorHeap* heaps[] = { m_srvUavHeap.Get() };
	cmd->SetDescriptorHeaps(1, heaps);

	// b0: SPHParams, t0/u0: in/out Buffers
	cmd->SetComputeRootConstantBufferView(0, m_paramCB->GetAddress());
	cmd->SetComputeRootDescriptorTable(1, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());

	UINT groups = (ParticleCount + 255) / 256;
	cmd->Dispatch(groups, 1, 1);

	// UAV バリア → ping-pong
	D3D12_RESOURCE_BARRIER barrierUAV =
		CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
	cmd->ResourceBarrier(1, &barrierUAV);
	std::swap(m_gpuInBuffer, m_gpuOutBuffer);

	// Readback 用にコピーするときのサイズ
	UINT64 sz = sizeof(Point) * ParticleCount;
	// （あるいは m_Particles.size() を使う）

	// GPU→Readback バッファへコピー
	cmd->CopyResource(m_readbackBuffer.Get(), m_gpuInBuffer.Get());

	// コマンドを流して GPU 完了待ち
	g_Engine->ExecuteCommandList();
	g_Engine->WaitForGpu();

	// CPU 側へマップしてデータ戻し
	void* src = nullptr;
	m_readbackBuffer->Map(0, nullptr, &src);
	memcpy(m_Particles.data(), src, (size_t)sz);
	m_readbackBuffer->Unmap(0, nullptr);

	// 既存のInstanceBuffer更新へ
	UpdateInstanceBuffer();

	/*auto ptr = m_ConstantBuffer[0]->GetPtr<Transform>();
	ptr->World = DirectX::XMMatrixIdentity();
	ptr->View = DirectX::XMMatrixLookAtRH(camera->GetEyePos(), camera->GetTargetPos(), camera->GetUpward());
	ptr->Proj = DirectX::XMMatrixPerspectiveFovRH(camera->GetFov(), camera->GetAspect(), 0.3f, 1000.0f);*/
}

void Particle::Draw() {
	auto cmd = g_Engine->CommandList();

	// Barrier for instance buffer
	D3D12_RESOURCE_BARRIER barrier =
		CD3DX12_RESOURCE_BARRIER::Transition(
			m_InstanceBuffer->GetResource(),
			D3D12_RESOURCE_STATE_COPY_DEST,            // または UAV
			D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
		);
	cmd->ResourceBarrier(1, &barrier);

	cmd->SetGraphicsRootSignature(m_RootSignature->Get());
	cmd->SetPipelineState(m_PipelineState->Get());

	//cmd->SetGraphicsRootConstantBufferView(
	//	/*paramIndex=*/0,
	//	cameraCB->GetGPUVirtualAddress()
	//);

	// 0: メッシュ頂点
	cmd->IASetVertexBuffers(0, 1, &m_vertexBufferView);
	// 1: インスタンス行列
	cmd->IASetVertexBuffers(1, 1, &m_instanceBufferView);
	cmd->IASetIndexBuffer(&m_indexBufferView);
	cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	cmd->DrawIndexedInstanced(m_indexCount, ParticleCount, 0, 0, 0);
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


	// CPU→UploadBuffer へ書き込み
	ParticleSB* mapped = nullptr;
	CD3DX12_RANGE readRange(0, 0);
	m_ParticleSBUpload->Map(0, &readRange, reinterpret_cast<void**>(&mapped));

	// カメラ行列を用意
	DirectX::XMMATRIX view = DirectX::XMMatrixLookAtRH(
		camera->GetEyePos(),
		camera->GetTargetPos(),
		camera->GetUpward()
	);
	DirectX::XMMATRIX proj = DirectX::XMMatrixPerspectiveFovRH(
		camera->GetFov(),
		camera->GetAspect(),
		0.3f,
		1000.0f
	);
	DirectX::XMMATRIX vpT = DirectX::XMMatrixTranspose(view * proj);

	float projScaleX = DirectX::XMVectorGetX(proj.r[0]);

	for (int i = 0; i < ParticleCount; ++i) {
		// ワールド空間の粒子位置
		auto& P = m_Particles[i].position;
		DirectX::XMVECTOR worldPos = DirectX::XMVectorSet(P.x, P.y, P.z, 1.0f);

		// クリップ空間へ変換
		DirectX::XMVECTOR clip = DirectX::XMVector3Transform(worldPos, vpT);
		float w = DirectX::XMVectorGetW(clip);

		// NDC 空間に正規化
		DirectX::XMVECTOR ndc = DirectX::XMVectorScale(clip, 1.0f / w);

		// 画面座標 (0–1) にマップ
		float x = DirectX::XMVectorGetX(ndc) * 0.5f + 0.5f;
		float y = 1.0f - (DirectX::XMVectorGetY(ndc) * 0.5f + 0.5f);

		// Z は粒子半径を w で補正したもの
		float z = m_SPHParams.radius * (1.0f / w);

		// ワールド半径 × プロジェクションの X軸スケール ÷ w
		float r_ndc = m_SPHParams.radius * projScaleX / w;
		// UV(0–1)空間にマップするならさらに×0.5
		float r_uv = r_ndc * 0.5f;

		mapped[i] = { x, y, r_uv, 0 };
	}

	m_ParticleSBUpload->Unmap(0, nullptr);
	g_Engine->CommandList()->CopyResource(m_ParticleSBGPU, m_ParticleSBUpload);


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

void Particle::UpdateInstanceBuffer()
{
	std::vector<InstanceData> instances(m_Particles.size());
	for (size_t i = 0; i < m_Particles.size(); ++i) {
		// パーティクルの物理空間での位置と半径
		auto pos = m_Particles[i].position;
		float r = m_SPHParams.radius;

		// 半径1の球メッシュを、物理半径rでスケール
		DirectX::XMMATRIX scale = DirectX::XMMatrixScaling(r, r, r);
		// パーティクル位置に移動
		DirectX::XMMATRIX trans = DirectX::XMMatrixTranslation(pos.x, pos.y, pos.z);
		DirectX::XMMATRIX world = scale * trans;
		// 列優先に合わせて転置
		DirectX::XMMATRIX worldT = DirectX::XMMatrixTranspose(world);

		// 行データを InstanceData に書き込む
		InstanceData& data = instances[i];
		XMStoreFloat4(&data.row0, worldT.r[0]);
		XMStoreFloat4(&data.row1, worldT.r[1]);
		XMStoreFloat4(&data.row2, worldT.r[2]);
		XMStoreFloat4(&data.row3, worldT.r[3]);
	}

	// GPUバッファへアップロード
	void* ptr = nullptr;
	m_InstanceBuffer->GetResource()->Map(0, nullptr, &ptr);
	memcpy(ptr, instances.data(), sizeof(InstanceData) * instances.size());
	m_InstanceBuffer->GetResource()->Unmap(0, nullptr);
}
