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
}

void Particle::Draw() {
	auto commandList = g_Engine->CommandList();

	int frameIndex = g_Engine->CurrentBackBufferIndex();

	auto ptr = m_ConstantBuffer[0]->GetPtr<Transform>();
	ptr->World = DirectX::XMMatrixIdentity();
	ptr->View = camera->GetViewMatrix();
	ptr->Proj = camera->GetProjMatrix();

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

	//// インスタンス描画
	//commandList->DrawIndexedInstanced(
	//	m_IndexCount,               // 球メッシュのインデックス数
	//	(UINT)m_Particles.size(),   // インスタンス数
	//	0, 0, 0);


	// メタボール描画
	DrawMetaball();
}


void Particle::UpdateParticles() {
	UINT frameIndex = g_Engine->CurrentBackBufferIndex();
	auto device = g_Engine->Device();
	auto queue = g_Engine->ComputeCommandQueue();
	auto cmd = m_computeCommandLists[frameIndex].Get();

	// 前フレームが終わるまで待つ
	UINT64 waitVal = m_computeFenceValues[frameIndex];
	if (m_computeFence->GetCompletedValue() < waitVal) {
		m_computeFence->SetEventOnCompletion(waitVal, m_computeFenceEvent);
		WaitForSingleObject(m_computeFenceEvent, INFINITE);
	}

	// Compute専用アロケーターとコマンドリストをリセット
	auto compAlloc = m_computeAllocators[frameIndex].Get();
	compAlloc->Reset();

	cmd->Reset(m_computeAllocators[frameIndex].Get(), nullptr);

	// 定数バッファ更新
	m_SPHParams.particleCount = static_cast<UINT>(m_Particles.size());
	memcpy(m_paramCB->GetPtr(), &m_SPHParams, sizeof(SPHParams));

	// Computeシェーダー実行設定
	cmd->SetPipelineState(m_computePSO.Get());
	cmd->SetComputeRootSignature(m_computeRS.Get());
	ID3D12DescriptorHeap* heaps[] = { m_computeDescHeap.Get() };
	cmd->SetDescriptorHeaps(1, heaps);
	cmd->SetComputeRootConstantBufferView(0, m_paramCB->GetAddress());
	cmd->SetComputeRootDescriptorTable(1, m_srvHandle);
	cmd->SetComputeRootDescriptorTable(2, m_uavHandle);
	cmd->SetComputeRootDescriptorTable(3, m_metaUAVHandle); // outMeta (u1)

	UINT groups = (ParticleCount + 255) / 256;
	cmd->Dispatch(groups, 1, 1);

	// UAV バリア：UAV 書き込み完了を待つ
	D3D12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
	cmd->ResourceBarrier(1, &uavBarrier);

	// Ping-Pong バッファ＆ディスクリプタ更新
	std::swap(m_gpuInBuffer, m_gpuOutBuffer);
	std::swap(m_srvHandle, m_uavHandle);

	// CPU ディスクリプタヒープ先頭／インクリメント取得
	auto cpuStart = m_computeDescHeap->GetCPUDescriptorHandleForHeapStart();
	UINT inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	// SRV (t0) を更新
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srvDesc.Buffer.NumElements = ParticleCount;
	srvDesc.Buffer.StructureByteStride = sizeof(Point);
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	device->CreateShaderResourceView(m_gpuInBuffer.Get(), &srvDesc, CD3DX12_CPU_DESCRIPTOR_HANDLE(cpuStart, 1, inc));

	// UAV (u0) を更新
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavDesc.Buffer.NumElements = ParticleCount;
	uavDesc.Buffer.StructureByteStride = sizeof(Point);
	device->CreateUnorderedAccessView(m_gpuOutBuffer.Get(), nullptr, &uavDesc, CD3DX12_CPU_DESCRIPTOR_HANDLE(cpuStart, 2, inc));


	// コマンド送信
	cmd->Close();
	ID3D12CommandList* lists[]{ cmd };
	queue->ExecuteCommandLists(1, lists);

	UINT64 nextVal = ++m_computeFenceCounter;
	queue->Signal(m_computeFence.Get(), nextVal);
	m_computeFenceValues[frameIndex] = nextVal;

	// グラフィックス側でもこのフェンス値を待つ
	auto graphicsQ = g_Engine->CommandQueue();
	graphicsQ->Wait(m_computeFence.Get(), nextVal);
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
	m_PipelineState->SetVS(L"ParticleVS.cso");
	m_PipelineState->SetPS(L"ParticlePS.cso");

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
	auto device = g_Engine->Device();

	// フルスクリーントライアングル頂点バッファ
	FullscreenVertex quad[3] = { {{-1,1}}, {{3,1}}, {{-1,-3}} };
	m_QuadVB = new VertexBuffer(
		sizeof(FullscreenVertex) * 3,
		sizeof(FullscreenVertex),
		quad
	);

	// ParticleMeta 用 GPU バッファを作成
	UINT elementCount = (UINT)m_Particles.size();
	UINT elementSize = sizeof(ParticleMeta);      // =16 バイト
	UINT bufferSize = elementCount * elementSize;
	CD3DX12_HEAP_PROPERTIES heapDef(D3D12_HEAP_TYPE_DEFAULT);
	CD3DX12_RESOURCE_DESC   descDef = CD3DX12_RESOURCE_DESC::Buffer(
		bufferSize,
		D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS  // UAV 用にフラグ
	);
	HRESULT hr = device->CreateCommittedResource(
		&heapDef, D3D12_HEAP_FLAG_NONE,
		&descDef,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		nullptr,
		IID_PPV_ARGS(&m_gpuMetaBuffer)
	);
	if (FAILED(hr)) {
		printf("m_gpuMetaBuffer 作成失敗: 0x%08X\n", hr);
		return false;
	}

	// 描画用ディスクリプタヒープに SRV を登録
	{
		auto handle = g_Engine->CbvSrvUavHeap()->RegisterBuffer(
			m_gpuMetaBuffer.Get(),
			elementCount,
			elementSize
		);
		if (!handle) {
			printf("描画用 SRV 登録失敗\n");
			return false;
		}
		m_metaSRVHandle = handle->HandleGPU;  // DrawMetaball で使う
	}


	// ルートシグネチャ／PSO の作成
	{
		CD3DX12_DESCRIPTOR_RANGE ranges[1] = {};
		ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0); // t0

		CD3DX12_ROOT_PARAMETER params[2];
		params[0].InitAsConstants(4, 0);                       // b0
		params[1].InitAsDescriptorTable(1, &ranges[0],
			D3D12_SHADER_VISIBILITY_PIXEL);

		CD3DX12_STATIC_SAMPLER_DESC sampler(0);
		sampler.Init(0, D3D12_FILTER_MIN_MAG_MIP_LINEAR);

		CD3DX12_ROOT_SIGNATURE_DESC rsDesc;
		rsDesc.Init(_countof(params), params,
			1, &sampler,
			D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		m_MetaRootSig = new RootSignature();
		m_MetaRootSig->Init(rsDesc);

		m_MetaPSO = new ParticlePipelineState();
		m_MetaPSO->SetRootSignature(m_MetaRootSig->Get());
		// フルスクリーン三角用
		D3D12_INPUT_ELEMENT_DESC elems[] = {
			{ "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
			  D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
		};
		D3D12_INPUT_LAYOUT_DESC layout{ elems, 1 };
		m_MetaPSO->SetInputLayout(layout);
		m_MetaPSO->SetVS(L"MetaBallVS.cso");
		m_MetaPSO->SetPS(L"MetaBallPS.cso");
		m_MetaPSO->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
	}

	return true;
}

bool Particle::InitComputeShader()
{
	// コンピュートシェーダー用のリソースを初期化
	ID3D12Device* device = g_Engine->Device();

	// Compute専用アロケーターをフレーム数分作成
	for (UINT i = 0; i < FrameCount; ++i) {
		HRESULT hr = device->CreateCommandAllocator(
			D3D12_COMMAND_LIST_TYPE_COMPUTE,
			IID_PPV_ARGS(&m_computeAllocators[i]));
		if (FAILED(hr)) {
			printf("ComputeAllocator[%u] の作成に失敗 HRESULT=0x%08X\n", i, hr);
			return false;
		}

		device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_computeAllocators[i].Get(), nullptr, IID_PPV_ARGS(&m_computeCommandLists[i]));
		// 生成後は一度 Close() しておく
		m_computeCommandLists[i]->Close();
	}

	// Compute専用フェンス
	HRESULT hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_computeFence));
	if (FAILED(hr)) {
		printf("ComputeFence の作成に失敗 HRESULT=0x%08X\n", hr);
		return false;
	}
	m_computeFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);


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
	hr = device->CreateCommittedResource(&heapProp, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&m_gpuInBuffer));
	m_gpuInBuffer->SetName(L"Particle_In");
	if (FAILED(hr)) {
		printf("m_gpuInBuffer 作成失敗\n");
		return false;
	}

	// 出力用バッファ
	hr = device->CreateCommittedResource(&heapProp, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_gpuOutBuffer));
	m_gpuOutBuffer->SetName(L"Particle_Out");
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
	m_computePSO.SetCS(L"ParticleCS.cso");
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
	heapDesc.NumDescriptors = 4;
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


	// CPU 側ハンドルを slot3 にオフセット
	D3D12_CPU_DESCRIPTOR_HANDLE cpuUAV = m_computeDescHeap->GetCPUDescriptorHandleForHeapStart();
	cpuUAV.ptr += handleSize * 3;

	// outMeta 用 UAV の記述
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDescMeta = {};
	uavDescMeta.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavDescMeta.Buffer.FirstElement = 0;
	uavDescMeta.Buffer.NumElements = static_cast<UINT>(m_Particles.size());
	uavDescMeta.Buffer.StructureByteStride = sizeof(ParticleMeta);
	uavDescMeta.Format = DXGI_FORMAT_UNKNOWN;

	// UAV 作成
	device->CreateUnorderedAccessView(m_gpuMetaBuffer.Get(), nullptr, &uavDescMeta, cpuUAV);

	// GPU 側ハンドルを同じ slot3 で計算して保持
	m_metaUAVHandle = m_computeDescHeap->GetGPUDescriptorHandleForHeapStart();
	m_metaUAVHandle.ptr += handleSize * 3;


	// 初期データを GPU 入力バッファにコピー
	ComPtr<ID3D12Resource> uploadBuf;
	CD3DX12_HEAP_PROPERTIES upProps(D3D12_HEAP_TYPE_UPLOAD);
	CD3DX12_RESOURCE_DESC   resDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
	device->CreateCommittedResource(&upProps, D3D12_HEAP_FLAG_NONE, &resDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadBuf));
	if (FAILED(hr)) {
		printf("アップロードバッファの作成に失敗\n");
		return false;
	}


	// メモリ書き込み
	void* p;  uploadBuf->Map(0, nullptr, &p);
	memcpy(p, m_Particles.data(), bufferSize);
	uploadBuf->Unmap(0, nullptr);

	// コマンドで inBuffer にコピー
	auto cmd = g_Engine->CommandList();
	auto alloc = g_Engine->CommandAllocator(0);
	alloc->Reset();
	cmd->Reset(alloc, nullptr);
	cmd->CopyResource(m_gpuInBuffer.Get(), uploadBuf.Get());
	CD3DX12_RESOURCE_BARRIER br = CD3DX12_RESOURCE_BARRIER::Transition(m_gpuInBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	cmd->ResourceBarrier(1, &br);
	cmd->Close();
	ID3D12CommandList* lists[]{ cmd };
	g_Engine->CommandQueue()->ExecuteCommandLists(1, lists);
	g_Engine->Flush();

	Sleep(10);

	return true;
}


// ==========================================================================
// 描画処理関係
// ==========================================================================
// Metaball 描画
void Particle::DrawMetaball() {
	auto cmd = g_Engine->CommandList();

	// UAVバリア
	D3D12_RESOURCE_BARRIER uavBar = CD3DX12_RESOURCE_BARRIER::UAV(m_gpuMetaBuffer.Get());
	cmd->ResourceBarrier(1, &uavBar);

	// ステート遷移
	D3D12_RESOURCE_BARRIER trans = CD3DX12_RESOURCE_BARRIER::Transition(
		m_gpuMetaBuffer.Get(),
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
	);
	cmd->ResourceBarrier(1, &trans);

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
	(float)g_Engine->FrameBufferWidth(), (float)g_Engine->FrameBufferHeight(), 0.5f, ParticleCount };
	cmd->SetGraphicsRoot32BitConstants(0, 4, &cbv, 0);

	// t0: metaball
	cmd->SetGraphicsRootDescriptorTable(1, m_metaSRVHandle);

	// VB/IA 設定
	auto vbv = m_QuadVB->View();
	cmd->IASetVertexBuffers(0, 1, &vbv);
	cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	cmd->DrawInstanced(3, 1, 0, 0);

	D3D12_RESOURCE_BARRIER back = CD3DX12_RESOURCE_BARRIER::Transition(
		m_gpuMetaBuffer.Get(),
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS
	);
	cmd->ResourceBarrier(1, &back);
}


