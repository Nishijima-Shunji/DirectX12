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
	if (!InitParticle()) {			// Particle情報を初期化
		return false;
	}
	if (!InitMesh()) {				// メッシュ情報を初期化
		return false;
	}
	if (!InitMetaball()) {			// メタボール情報を初期化
		return false;
	}
	if (!InitComputeShader()) {		// コンピュートシェーダー情報を初期化
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
	commandList->DrawIndexedInstanced(
		m_IndexCount,               // 球メッシュのインデックス数
		(UINT)m_Particles.size(),   // インスタンス数
		0, 0, 0);


	// ====================================================
	// --- Metaball 描画 ---
	auto cmd = g_Engine->CommandList();

	// SRV/UAV 用ヒープをルートにバインド
	ID3D12DescriptorHeap* heaps[] = { g_Engine->CbvSrvUavHeap()->GetHeap() };
	cmd->SetDescriptorHeaps(_countof(heaps), heaps);

	cmd->SetPipelineState(m_MetaPSO->Get());
	cmd->SetGraphicsRootSignature(m_MetaRootSig->Get());

	// b0: screenSize + threshold
	struct {
		float w;
		float h;
		float thr;
		UINT count;
	} cbv = {
	(float)g_Engine->FrameBufferWidth(), (float)g_Engine->FrameBufferHeight(),
	/*threshold=*/0.5f,
	ParticleCount
	};
	cmd->SetGraphicsRoot32BitConstants(0, 4, &cbv, 0);

	// t0: Particle SB
	cmd->SetGraphicsRootDescriptorTable(1, m_ParticleSB_SRV);

	// VB/IA 設定
	auto vbv = m_QuadVB->View();
	cmd->IASetVertexBuffers(0, 1, &vbv);
	cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	// 三角１枚
	cmd->DrawInstanced(3, 1, 0, 0);
}


void Particle::UpdateParticles() {
	UINT frameIndex = g_Engine->CurrentBackBufferIndex();

	auto device		= g_Engine->Device();
	auto cmd		= g_Engine->CommandList();
	auto allocator	= g_Engine->CommandAllocator(frameIndex);
	auto queue		= g_Engine->CommandQueue();

	// FenceでGPUの完了を確認
	if (m_fence->GetCompletedValue() < m_fenceValue[frameIndex]) {
		m_fence->SetEventOnCompletion(m_fenceValue[frameIndex], m_fenceEvent);
		WaitForSingleObject(m_fenceEvent, INFINITE);
	}

	// 安全にReset
	allocator->Reset();
	cmd->Reset(allocator, nullptr);

	// 定数バッファ更新
	m_SPHParams.particleCount = static_cast<UINT>(m_Particles.size());
	memcpy(m_paramCB->GetPtr(), &m_SPHParams, sizeof(SPHParams));

	// ルートシグネチャとPSO
	cmd->SetComputeRootSignature(m_computeRS.Get());
	cmd->SetPipelineState(m_computePSO.Get());

	// ディスクリプタヒープ設定
	ID3D12DescriptorHeap* heaps[] = { m_computeDescHeap.Get() };
	cmd->SetDescriptorHeaps(1, heaps);

	cmd->SetComputeRootConstantBufferView(0, m_paramCB->GetAddress());
	cmd->SetComputeRootDescriptorTable(1, m_srvHandle);
	cmd->SetComputeRootDescriptorTable(2, m_uavHandle);

	// コンピュートシェーダー実行
	UINT groups = (ParticleCount + 255) / 256;
	cmd->Dispatch(groups, 1, 1);

	// UAVバリア（ping-pong用）
	D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
	cmd->ResourceBarrier(1, &barrier);

	// CPU→UploadBuffer へ書き込み
	ParticleSB* mapped = nullptr;
	CD3DX12_RANGE readRange(0, 0);
	m_ParticleSBUpload->Map(0, &readRange, reinterpret_cast<void**>(&mapped));
	if (!mapped) {
		printf("UploadBufferのマッピングに失敗\n");
		return;
	}

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
		auto& P = m_Particles[i].position;
		DirectX::XMVECTOR worldPos = DirectX::XMVectorSet(P.x, P.y, P.z, 1.0f);
		DirectX::XMVECTOR clip = DirectX::XMVector3Transform(worldPos, vpT);
		float w = DirectX::XMVectorGetW(clip);
		DirectX::XMVECTOR ndc = DirectX::XMVectorScale(clip, 1.0f / w);
		float x = DirectX::XMVectorGetX(ndc) * 0.5f + 0.5f;
		float y = 1.0f - (DirectX::XMVectorGetY(ndc) * 0.5f + 0.5f);
		float r_uv = m_SPHParams.radius * projScaleX / w * 0.5f;

		mapped[i] = { x, y, r_uv, 0 };
	}

	m_ParticleSBUpload->Unmap(0, nullptr);

	// CopyResource（CommandList内に配置）
	cmd->CopyResource(m_ParticleSBGPU, m_ParticleSBUpload);

	// コマンドリストを閉じる
	cmd->Close();

	// 実行＆フェンス
	ID3D12CommandList* cmdLists[] = { cmd };
	queue->ExecuteCommandLists(1, cmdLists);
	queue->Signal(m_fence.Get(), ++m_fenceValue[frameIndex]);

	// ping-pong バッファ入れ替え
	std::swap(m_gpuInBuffer, m_gpuOutBuffer);
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
				// 圧力
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
	UINT frameIndex = g_Engine->CurrentBackBufferIndex();
	if (m_fence->GetCompletedValue() < m_fenceValue[frameIndex]) {
		m_fence->SetEventOnCompletion(m_fenceValue[frameIndex], m_fenceEvent);
		WaitForSingleObject(m_fenceEvent, INFINITE);
	}

	std::vector<InstanceData> instances(m_Particles.size());
	for (size_t i = 0; i < m_Particles.size(); ++i) {
		auto pos = m_Particles[i].position;
		float r = m_SPHParams.radius;
		DirectX::XMMATRIX scale = DirectX::XMMatrixScaling(r, r, r);
		DirectX::XMMATRIX trans = DirectX::XMMatrixTranslation(pos.x, pos.y, pos.z);
		DirectX::XMMATRIX worldT = XMMatrixTranspose(scale * trans);

		InstanceData& data = instances[i];
		XMStoreFloat4(&data.row0, worldT.r[0]);
		XMStoreFloat4(&data.row1, worldT.r[1]);
		XMStoreFloat4(&data.row2, worldT.r[2]);
		XMStoreFloat4(&data.row3, worldT.r[3]);
	}

	void* ptr = nullptr;
	HRESULT hr = m_InstanceBuffer->GetResource()->Map(0, nullptr, &ptr);
	if (FAILED(hr) || !ptr) {
		printf("インスタンスバッファ Map に失敗\n");
		return;
	}
	memcpy(ptr, instances.data(), sizeof(InstanceData) * instances.size());
	m_InstanceBuffer->GetResource()->Unmap(0, nullptr);
}



// ==========================================================================
// 初期化処理関係
// ==========================================================================
bool Particle::InitParticle() {
	// 粒子生成
	for (int i = 0; i < ParticleCount; ++i) {
		Point p;

		p.position = { RandFloat(-0.5f, 0.5f), RandFloat(-0.5f, 0.5f), RandFloat(-0.5f, 0.5f) };
		p.velocity = { 0, 0, 0 };
		m_Particles.push_back(p);
	}
	// 粒子のパラメーターの初期化
	m_SPHParams.restDensity = 1000.0f;	// 
	m_SPHParams.particleMass = 1.0f;	// 重さ
	m_SPHParams.viscosity = 5.0f;		// 粘性
	m_SPHParams.stiffness = 1.0f;		// 剛性
	m_SPHParams.radius = 0.1f;			// 
	m_SPHParams.timeStep = 0.016f;		// 
	m_SPHParams.particleCount = ParticleCount;

	// ディスクリプタヒープの生成
	D3D12_DESCRIPTOR_HEAP_DESC hdesc = {};
	hdesc.NumDescriptors = 2; // SRV + UAV
	hdesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	hdesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	g_Engine->Device()->CreateDescriptorHeap(&hdesc, IID_PPV_ARGS(&m_srvUavHeap));

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

bool Particle::InitMesh() {
	// 半径 1 の低ポリ球を生成
	auto mesh = CreateLowPolySphere(1.0f, 0);
	m_IndexCount = (UINT)mesh.indices.size();

	// 頂点バッファ
	m_MeshVertexBuffer = new VertexBuffer(
		sizeof(Vertex) * mesh.vertices.size(),
		sizeof(Vertex),
		mesh.vertices.data());

	// インデックスバッファ
	m_MeshIndexBuffer = new IndexBuffer(
		sizeof(uint32_t) * mesh.indices.size(),
		mesh.indices.data());

	if (!m_MeshVertexBuffer || !m_MeshIndexBuffer) {
		printf("Meshバッファ作成失敗\n");
		return false;
	}

	return true;
}

bool Particle::InitMetaball() {
	// フルスクリーントライアングル頂点バッファ生成
	FullscreenVertex quad[3] = {
		{{-1,  1}},
		{{ 3,  1}},
		{{-1, -3}}
	};
	m_QuadVB = new VertexBuffer(
		sizeof(FullscreenVertex) * 3,
		sizeof(FullscreenVertex),
		quad);

	// パーティクル SB(StructuredBuffer) 用 GPU & Upload バッファ
	UINT64 sbSize = sizeof(ParticleSB) * ParticleCount;

	// --- デフォルトヒープ用リソース ---
	// ローカル変数に束縛
	CD3DX12_HEAP_PROPERTIES defaultHeapProps(D3D12_HEAP_TYPE_DEFAULT);
	CD3DX12_RESOURCE_DESC   defaultResDesc = CD3DX12_RESOURCE_DESC::Buffer(sbSize);

	HRESULT hr = g_Engine->Device()->CreateCommittedResource(
		&defaultHeapProps,                   // ← 変数のアドレスを渡す
		D3D12_HEAP_FLAG_NONE,
		&defaultResDesc,                     // ← 変数のアドレスを渡す
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&m_ParticleSBGPU)
	);
	if (FAILED(hr)) {
		printf("デフォルトヒープ作成失敗\n");
		return false;
	}

	// --- アップロードヒープ用リソース ---
	CD3DX12_HEAP_PROPERTIES uploadHeapProps(D3D12_HEAP_TYPE_UPLOAD);
	CD3DX12_RESOURCE_DESC   uploadResDesc = CD3DX12_RESOURCE_DESC::Buffer(sbSize);

	hr = g_Engine->Device()->CreateCommittedResource(
		&uploadHeapProps,                    // ← 変数のアドレスを渡す
		D3D12_HEAP_FLAG_NONE,
		&uploadResDesc,                      // ← 変数のアドレスを渡す
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&m_ParticleSBUpload)
	);
	if (FAILED(hr)) {
		printf("アップロードヒープ作成失敗\n");
		return false;
	}

	// SRV ディスクリプタ準備 
	auto handle = g_Engine->CbvSrvUavHeap()->RegisterBuffer(
		m_ParticleSBGPU,       // ID3D12Resource*
		ParticleCount,         // 要素数
		sizeof(ParticleSB)     // １要素のバイト幅
	);
	if (!handle) {
		// エラー処理
		printf("Particle SB 用 SRV の登録に失敗\n");
		return false;
	}
	// ピクセルシェーダーで SetGraphicsRootDescriptorTable に使う GPU ハンドル
	m_ParticleSB_SRV = handle->HandleGPU;

	// RootSignature 作成
	{
		CD3DX12_DESCRIPTOR_RANGE ranges[1];
		ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0); // t0

		CD3DX12_ROOT_PARAMETER params[2];
		// b0: screenSize + threshold
		params[0].InitAsConstants(4, 0);
		// t0: パーティクル SB
		params[1].InitAsDescriptorTable(1, &ranges[0], D3D12_SHADER_VISIBILITY_PIXEL);

		CD3DX12_STATIC_SAMPLER_DESC sampler(0);
		sampler.Init(0, D3D12_FILTER_MIN_MAG_MIP_LINEAR);

		CD3DX12_ROOT_SIGNATURE_DESC rsigDesc;
		rsigDesc.Init(
			_countof(params), params,
			1, &sampler,
			D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		m_MetaRootSig = new RootSignature();
		m_MetaRootSig->Init(rsigDesc);
	}

	// PSO 作成
	{
		m_MetaPSO = new ParticlePipelineState();
		m_MetaPSO->SetRootSignature(m_MetaRootSig->Get());
		// フルスクリーン用入力レイアウト
		D3D12_INPUT_ELEMENT_DESC elems[] = {
			{ "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
			  D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
		};
		D3D12_INPUT_LAYOUT_DESC layout{ elems, 1 };
		m_MetaPSO->SetInputLayout(layout);
		m_MetaPSO->SetVS(L"../x64/Debug/MetaballVS.cso");
		m_MetaPSO->SetPS(L"../x64/Debug/MetaballPS.cso");
		m_MetaPSO->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
	}
	return true;
}

bool Particle::InitComputeShader()
{
	// コンピュートシェーダー用のリソースを初期化
	ID3D12Device* device = g_Engine->Device();

	// フェンスの初期化
	device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence));
	m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

	// GPUバッファの作成（入力・出力）
	UINT elementCount = static_cast<UINT>(m_Particles.size());
	UINT elementSize = sizeof(Point);
	UINT bufferSize = elementCount * elementSize;

	CD3DX12_HEAP_PROPERTIES heapProp(D3D12_HEAP_TYPE_DEFAULT);
	CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

	// 入力用バッファ
	HRESULT hr = device->CreateCommittedResource( &heapProp, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,	nullptr, IID_PPV_ARGS(&m_gpuInBuffer));
	if (FAILED(hr)) {
		printf("m_gpuInBuffer 作成失敗\n");
		return false;
	}

	// 出力用バッファ
	hr = device->CreateCommittedResource(&heapProp, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_gpuOutBuffer));
	if (FAILED(hr)) {
		printf("m_gpuOutBuffer 作成失敗\n");
		return false;
	}

	// Compute用ルートシグネチャの初期化
	if (!m_computeRS.InitForSPH()) {
		return false;
	}

	// Compute PSO 設定
	m_computePSO.SetRootSignature(m_computeRS.Get());
	m_computePSO.SetCS(L"../x64/Debug/ParticleCS.cso"); // ※コンパイル済みCSOを前提にしている
	m_computePSO.Create();

	// 定数バッファ（SPHParams）を作成
	m_paramCB = new ConstantBuffer(sizeof(SPHParams));
	if (!m_paramCB || !m_paramCB->IsValid()) {
		printf("Compute用定数バッファの作成に失敗\n");
		return false;
	}

	// 初期値書き込み
	memcpy(m_paramCB->GetPtr(), &m_SPHParams, sizeof(SPHParams));

	// ディスクリプタヒープ（CBV, SRV, UAV）作成
	D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
	heapDesc.NumDescriptors = 3;
	heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	if (FAILED(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_computeDescHeap)))) {
		return false;
	}

	// 定数バッファ（m_paramCB）→ CBV作成
	D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
	cbvDesc.BufferLocation = m_paramCB->GetAddress();
	cbvDesc.SizeInBytes = (sizeof(SPHParams) + 255) & ~255;

	auto handle = m_computeDescHeap->GetCPUDescriptorHandleForHeapStart();
	device->CreateConstantBufferView(&cbvDesc, handle);

	UINT handleSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	// SRV 作成（m_gpuInBuffer）
	handle.ptr += handleSize;
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srvDesc.Buffer.FirstElement = 0;
	srvDesc.Buffer.NumElements = static_cast<UINT>(m_Particles.size());
	srvDesc.Buffer.StructureByteStride = sizeof(Point);
	srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
	srvDesc.Format = DXGI_FORMAT_UNKNOWN;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

	device->CreateShaderResourceView(m_gpuInBuffer.Get(), &srvDesc, handle);

	// UAV 作成（m_gpuOutBuffer）
	handle.ptr += handleSize;
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavDesc.Buffer.FirstElement = 0;
	uavDesc.Buffer.NumElements = static_cast<UINT>(m_Particles.size());
	uavDesc.Buffer.StructureByteStride = sizeof(Point);
	uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
	uavDesc.Format = DXGI_FORMAT_UNKNOWN;

	device->CreateUnorderedAccessView(m_gpuOutBuffer.Get(), nullptr, &uavDesc, handle);

	// ヒープ先頭を取得
	auto gpuHeapStart = m_computeDescHeap->GetGPUDescriptorHandleForHeapStart();

	m_srvHandle.ptr = gpuHeapStart.ptr + handleSize * 1;   // t0
	m_uavHandle.ptr = gpuHeapStart.ptr + handleSize * 2;   // u0

	return true;
}


