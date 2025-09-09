#include "FluidSystem.h"
#include "d3dx12.h"
#include <algorithm>
#include <d3dcompiler.h>
#include <vector>
#include <cmath>
#include "Engine.h"
#include "Camera.h"
#include "RandomUtil.h"
#include <d3dcompiler.h>
#pragma comment(lib, "d3dcompiler.lib")

// 粒子のパラメーター
struct SPHParamsCB {
	float restDensity;
	float particleMass;
	float viscosity;
	float stiffness;
	float radius;
	float timeStep;
	uint32_t particleCount;
	uint32_t pad0;
	DirectX::XMFLOAT3 gridMin;
	uint32_t pad1;
	DirectX::XMUINT3  gridDim;
	uint32_t pad2;
};

// View-Projection行列
struct ViewProjCB {
	DirectX::XMFLOAT4X4 viewProj;
};

// 流体システム初期化
void FluidSystem::Init(ID3D12Device* device, DXGI_FORMAT rtvFormat, UINT maxParticles, UINT threadGroupCount) {
	m_maxParticles = maxParticles;
	m_threadGroupCount = (maxParticles + 255) / 256;
	m_cpuParticles.resize(maxParticles);
	m_density.resize(maxParticles);
	m_neighborBuffer.reserve(MAX_PARTICLES_PER_CELL * 27);
	m_mainRTFormat = rtvFormat;


	for (auto& p : m_cpuParticles) {
		p.position = { RandFloat(-1.0f, 1.0f), RandFloat(-1.0f, 5.0f), RandFloat(-1.0f, 1.0f) };
		p.velocity = { 0.0f, 0.0f, 0.0f };
	}

	// ---------------------------------------------------------------------
	// Computeルートシグネチャー
	// 0 : CBV (b0)
	// 1 : CBV (b1)
	// 2 : SRV (t0)
	// 3 : UAV (u0)
	// 4 : UAV (u1)
	CD3DX12_DESCRIPTOR_RANGE srvRange;
	srvRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0); // t0

	CD3DX12_DESCRIPTOR_RANGE uavRange0;
	uavRange0.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0); // u0

	CD3DX12_DESCRIPTOR_RANGE uavRange1;
	uavRange1.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1); // u1
	CD3DX12_DESCRIPTOR_RANGE uavRange2;
	uavRange2.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2); // u2
	CD3DX12_DESCRIPTOR_RANGE uavRange3;
	uavRange3.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 3); // u3

	CD3DX12_ROOT_PARAMETER params[7];
	params[0].InitAsConstantBufferView(0);      // b0
	params[1].InitAsConstantBufferView(1);      // b1
	params[2].InitAsDescriptorTable(1, &srvRange, D3D12_SHADER_VISIBILITY_ALL);
	params[3].InitAsDescriptorTable(1, &uavRange0, D3D12_SHADER_VISIBILITY_ALL);
	params[4].InitAsDescriptorTable(1, &uavRange1, D3D12_SHADER_VISIBILITY_ALL);
	params[5].InitAsDescriptorTable(1, &uavRange2, D3D12_SHADER_VISIBILITY_ALL);
	params[6].InitAsDescriptorTable(1, &uavRange3, D3D12_SHADER_VISIBILITY_ALL);

	CD3DX12_ROOT_SIGNATURE_DESC rsDesc(_countof(params), params, 0, nullptr,
		D3D12_ROOT_SIGNATURE_FLAG_NONE);

	ComPtr<ID3DBlob> blob, error;
	HRESULT hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1,
		&blob, &error);
	if (FAILED(hr)) {
		if (error) {
			wprintf(L"RootSignature初期化失敗: %s\n", (char*)error->GetBufferPointer());
		}
		return;
	}

	hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
		IID_PPV_ARGS(&m_computeRS));
	if (FAILED(hr)) {
		wprintf(L"RootSignature生成失敗: 0x%08X\n", hr);
		return;
	}

	// ---------------------------------------------------------------------
	// ComputePSO
	m_computePS.SetDevice(device);
	m_computePS.SetRootSignature(m_computeRS.Get());
	m_computePS.SetCS(L"ParticleCS.cso");
	bool computeOk = m_computePS.Create();

	m_buildGridPS.SetDevice(device);
	m_buildGridPS.SetRootSignature(m_computeRS.Get());
	m_buildGridPS.SetCS(L"BuildGridCS.cso");
	bool buildOk = m_buildGridPS.Create();

	if (!computeOk || !buildOk) {
		wprintf(L"[Warning] ComputePSO生成失敗\n");
		m_useGpu = false;
	}

	// ---------------------------------------------------------------------
	// Particleとmetabuffers
	D3D12_RESOURCE_DESC rdPart = CD3DX12_RESOURCE_DESC::Buffer(
		sizeof(FluidParticle) * maxParticles,
		D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

	D3D12_RESOURCE_DESC rdMeta = CD3DX12_RESOURCE_DESC::Buffer(
		sizeof(ParticleMeta) * maxParticles,
		D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

	CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_DEFAULT);
	device->CreateCommittedResource(
		&heapProps,
		D3D12_HEAP_FLAG_NONE,
		&rdPart,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		nullptr,
		IID_PPV_ARGS(&m_particleBuffer));
	device->CreateCommittedResource(
		&heapProps,
		D3D12_HEAP_FLAG_NONE,
		&rdMeta,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		nullptr,
		IID_PPV_ARGS(&m_metaBuffer));

	D3D12_DESCRIPTOR_HEAP_DESC hd = {};
	hd.NumDescriptors = 5; // SRV + UAV + UAV + UAV + UAV
	hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_uavHeap));

	UINT handleSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	D3D12_CPU_DESCRIPTOR_HANDLE handle = m_uavHeap->GetCPUDescriptorHandleForHeapStart();

	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srvDesc.Buffer.NumElements = maxParticles;
	srvDesc.Buffer.StructureByteStride = sizeof(FluidParticle);
	srvDesc.Format = DXGI_FORMAT_UNKNOWN;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	device->CreateShaderResourceView(m_particleBuffer.Get(), &srvDesc, handle);

	handle.ptr += handleSize;
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavd = {};
	uavd.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavd.Buffer.NumElements = maxParticles;
	uavd.Buffer.StructureByteStride = sizeof(FluidParticle);
	device->CreateUnorderedAccessView(m_particleBuffer.Get(), nullptr, &uavd, handle);

	handle.ptr += handleSize;
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavMeta = {};
	uavMeta.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavMeta.Buffer.NumElements = maxParticles;
	uavMeta.Buffer.StructureByteStride = sizeof(ParticleMeta);
	device->CreateUnorderedAccessView(m_metaBuffer.Get(), nullptr, &uavMeta, handle);

	handle.ptr += handleSize;

	// 描画用パイプライン生成
	// ルートシグネチャ
	CD3DX12_DESCRIPTOR_RANGE graRange;
	graRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
	CD3DX12_ROOT_PARAMETER graRootParam[2];
	graRootParam[0].InitAsDescriptorTable(1, &graRange, D3D12_SHADER_VISIBILITY_PIXEL);
	graRootParam[1].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_PIXEL);
	CD3DX12_ROOT_SIGNATURE_DESC rsd(2, graRootParam, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
	ComPtr<ID3DBlob> graBlob, graErr;
	D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &graBlob, &graErr);
	device->CreateRootSignature(0, graBlob->GetBufferPointer(), graBlob->GetBufferSize(), IID_PPV_ARGS(&m_graphicsRS));

	// PSO
	m_graphicsPS = new PipelineState();
	m_graphicsPS->SetRootSignature(m_graphicsRS.Get());
	D3D12_INPUT_LAYOUT_DESC nullIL{ nullptr, 0 };
	m_graphicsPS->SetInputLayout(nullIL);
	m_graphicsPS->SetVS(L"MetaBallVS.cso");
	m_graphicsPS->SetPS(L"MetaBallPS.cso");
	m_graphicsPS->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);

	m_viewWidth = g_Engine->FrameBufferWidth();
	m_viewHeight = g_Engine->FrameBufferHeight();

	// SSAリソース＆PSO生成（低解像度RTとPSO）
	CreateSSAResources(device, rtvFormat, m_viewWidth, m_viewHeight);
	CreateSSAPipelines(device, DXGI_FORMAT_R16_FLOAT);

	// SRV ヒープ
	D3D12_DESCRIPTOR_HEAP_DESC hd2 = {};
	hd2.NumDescriptors = 1;
	hd2.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	hd2.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	device->CreateDescriptorHeap(&hd2, IID_PPV_ARGS(&m_graphicsSrvHeap));
	D3D12_SHADER_RESOURCE_VIEW_DESC srvd = {};
	srvd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srvd.Buffer.NumElements = maxParticles;
	srvd.Buffer.StructureByteStride = sizeof(ParticleMeta);
	srvd.Format = DXGI_FORMAT_UNKNOWN;
        srvd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        device->CreateShaderResourceView(m_metaBuffer.Get(), &srvd,
                m_graphicsSrvHeap->GetCPUDescriptorHandleForHeapStart());

        // 保持しておき、後で描画時に設定する
        m_particleSRV = m_graphicsSrvHeap->GetGPUDescriptorHandleForHeapStart();

	// 定数バッファ
	D3D12_HEAP_PROPERTIES hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
	D3D12_RESOURCE_DESC   cbd = CD3DX12_RESOURCE_DESC::Buffer(
		sizeof(DirectX::XMFLOAT4X4) +
		sizeof(DirectX::XMFLOAT3) +
		sizeof(float) +
		sizeof(UINT) +
		sizeof(DirectX::XMFLOAT3));
	device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &cbd,
		D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
		IID_PPV_ARGS(&m_graphicsCB));

	// アップロードヒープ (CPU→GPU 転送用)
	CD3DX12_HEAP_PROPERTIES upProps(D3D12_HEAP_TYPE_UPLOAD);
	D3D12_RESOURCE_DESC uploadDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(FluidParticle) * maxParticles);
	device->CreateCommittedResource(&upProps, D3D12_HEAP_FLAG_NONE, &uploadDesc,
		D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
		IID_PPV_ARGS(&m_particleUpload));
	uploadDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(ParticleMeta) * maxParticles);
	device->CreateCommittedResource(&upProps, D3D12_HEAP_FLAG_NONE, &uploadDesc,
		D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
		IID_PPV_ARGS(&m_metaUpload));

	auto cmd = g_Engine->CommandList();
	auto allocator = g_Engine->CommandAllocator(g_Engine->CurrentBackBufferIndex());
	cmd->Reset(allocator, nullptr);

	auto barrier0 = CD3DX12_RESOURCE_BARRIER::Transition(
		m_particleBuffer.Get(),
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		D3D12_RESOURCE_STATE_COPY_DEST);
	auto metaToCopy = CD3DX12_RESOURCE_BARRIER::Transition(
		m_metaBuffer.Get(),
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		D3D12_RESOURCE_STATE_COPY_DEST);
	CD3DX12_RESOURCE_BARRIER toCopy[] = { barrier0, metaToCopy };
	cmd->ResourceBarrier(_countof(toCopy), toCopy);

	D3D12_SUBRESOURCE_DATA src = {};
	src.pData = m_cpuParticles.data();
	src.RowPitch = sizeof(FluidParticle) * m_maxParticles;
	src.SlicePitch = src.RowPitch;
	UpdateSubresources<1>(cmd, m_particleBuffer.Get(), m_particleUpload.Get(), 0, 0, 1, &src);

	std::vector<ParticleMeta> initMeta(m_maxParticles);
	for (UINT i = 0; i < m_maxParticles; ++i) {
		initMeta[i].pos = m_cpuParticles[i].position;
		initMeta[i].r = m_params.radius;
	}
	D3D12_SUBRESOURCE_DATA metaSrc = {};
	metaSrc.pData = initMeta.data();
	metaSrc.RowPitch = sizeof(ParticleMeta) * m_maxParticles;
	metaSrc.SlicePitch = metaSrc.RowPitch;
	UpdateSubresources<1>(cmd, m_metaBuffer.Get(), m_metaUpload.Get(), 0, 0, 1, &metaSrc);

	auto barrier1 = CD3DX12_RESOURCE_BARRIER::Transition(
		m_particleBuffer.Get(),
		D3D12_RESOURCE_STATE_COPY_DEST,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	auto metaBack = CD3DX12_RESOURCE_BARRIER::Transition(
		m_metaBuffer.Get(),
		D3D12_RESOURCE_STATE_COPY_DEST,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	CD3DX12_RESOURCE_BARRIER toUa[] = { barrier1, metaBack };
	cmd->ResourceBarrier(_countof(toUa), toUa);

	cmd->Close();
	ID3D12CommandList* lists[] = { cmd };
	g_Engine->CommandQueue()->ExecuteCommandLists(1, lists);
	g_Engine->Flush();


	// Compute用定数バッファ作成
	m_sphParamCB = new ConstantBuffer(sizeof(SPHParamsCB));
	m_viewProjCB = new ConstantBuffer(sizeof(ViewProjCB));

	// 初期値設定
	if (m_sphParamCB && m_sphParamCB->IsValid()) {
		auto* cb = m_sphParamCB->GetPtr<SPHParamsCB>();
		cb->restDensity = 1000.0f;
		cb->particleMass = 1.0f;
		cb->viscosity = 1.0f;
		cb->stiffness = 200.0f;
		cb->radius = 0.1f;
		cb->timeStep = 0.016f;
		cb->particleCount = m_maxParticles;
		cb->pad0 = 0;
		cb->gridMin = m_gridMin;

		m_params.restDensity = cb->restDensity;
		m_params.particleMass = cb->particleMass;
		m_params.viscosity = cb->viscosity;
		m_params.stiffness = cb->stiffness;
		m_params.radius = cb->radius;
		m_params.timeStep = cb->timeStep;
		m_grid.SetCellSize(cb->radius);

		m_gridDimX = static_cast<UINT>(ceil((2.0f) / cb->radius)) + 1;
		m_gridDimY = static_cast<UINT>(ceil((6.0f) / cb->radius)) + 1;
		m_gridDimZ = static_cast<UINT>(ceil((2.0f) / cb->radius)) + 1;
		cb->gridDim = DirectX::XMUINT3(m_gridDimX, m_gridDimY, m_gridDimZ);
		cb->pad1 = cb->pad2 = 0;
		m_cellCount = m_gridDimX * m_gridDimY * m_gridDimZ;

		D3D12_RESOURCE_DESC rdCount = CD3DX12_RESOURCE_DESC::Buffer(
			sizeof(UINT) * m_cellCount,
			D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		D3D12_RESOURCE_DESC rdTable = CD3DX12_RESOURCE_DESC::Buffer(
			sizeof(UINT) * m_cellCount * MAX_PARTICLES_PER_CELL,
			D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &rdCount,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
			IID_PPV_ARGS(&m_gridCount));
		device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &rdTable,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
			IID_PPV_ARGS(&m_gridTable));


		D3D12_CPU_DESCRIPTOR_HANDLE gridHandle = m_uavHeap->GetCPUDescriptorHandleForHeapStart();
		gridHandle.ptr += handleSize * 3;
		D3D12_UNORDERED_ACCESS_VIEW_DESC uavCount = {};
		uavCount.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		uavCount.Buffer.NumElements = m_cellCount;
		uavCount.Buffer.StructureByteStride = sizeof(UINT);
		device->CreateUnorderedAccessView(m_gridCount.Get(), nullptr, &uavCount, gridHandle);

		gridHandle.ptr += handleSize;
		D3D12_UNORDERED_ACCESS_VIEW_DESC uavTable = {};
		uavTable.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		uavTable.Buffer.NumElements = m_cellCount * MAX_PARTICLES_PER_CELL;
		uavTable.Buffer.StructureByteStride = sizeof(UINT);
		device->CreateUnorderedAccessView(m_gridTable.Get(), nullptr, &uavTable, gridHandle);
	}

	if (m_viewProjCB && m_viewProjCB->IsValid()) {
		auto* cb = m_viewProjCB->GetPtr<ViewProjCB>();
		DirectX::XMStoreFloat4x4(&cb->viewProj, DirectX::XMMatrixIdentity());
	}
}

// シミュレーション実行
void FluidSystem::Simulate(ID3D12GraphicsCommandList* cmd, float dt) {
	if (m_useGpu) {
		// GPU シミュレーション
		if (m_metaInSrvState) {
			auto toUav = CD3DX12_RESOURCE_BARRIER::Transition(
				m_metaBuffer.Get(),
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
				D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
			cmd->ResourceBarrier(1, &toUav);
			m_metaInSrvState = false;
		}
		if (m_particleInSrvState) {
			auto toUav = CD3DX12_RESOURCE_BARRIER::Transition(
				m_particleBuffer.Get(),
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
				D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
			cmd->ResourceBarrier(1, &toUav);
			m_particleInSrvState = false;
		}

		CD3DX12_RESOURCE_BARRIER barriers[] = {
				CD3DX12_RESOURCE_BARRIER::UAV(m_particleBuffer.Get()),
				CD3DX12_RESOURCE_BARRIER::UAV(m_metaBuffer.Get()),
				CD3DX12_RESOURCE_BARRIER::UAV(m_gridCount.Get()),
				CD3DX12_RESOURCE_BARRIER::UAV(m_gridTable.Get())
		};
		cmd->ResourceBarrier(_countof(barriers), barriers);
		cmd->SetComputeRootSignature(m_computeRS.Get());
		ID3D12DescriptorHeap* heaps[] = { m_uavHeap.Get() };
		cmd->SetDescriptorHeaps(1, heaps);

		UINT handleSize = g_Engine->Device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = m_uavHeap->GetGPUDescriptorHandleForHeapStart();
		D3D12_GPU_DESCRIPTOR_HANDLE uavParticle = srvHandle;
		uavParticle.ptr += handleSize;
		D3D12_GPU_DESCRIPTOR_HANDLE uavMeta = uavParticle;
		uavMeta.ptr += handleSize;
		D3D12_GPU_DESCRIPTOR_HANDLE uavCount = uavMeta;
		uavCount.ptr += handleSize;
		D3D12_GPU_DESCRIPTOR_HANDLE uavTable = uavCount;
		uavTable.ptr += handleSize;

		UINT maxThreads = (m_cellCount > m_maxParticles) ? m_cellCount : m_maxParticles;
		UINT buildGroups = (maxThreads + 255) / 256;
		cmd->SetPipelineState(m_buildGridPS.Get());
		cmd->SetComputeRootConstantBufferView(0, m_sphParamCB->GetAddress());
		cmd->SetComputeRootConstantBufferView(1, m_viewProjCB->GetAddress());
		cmd->SetComputeRootDescriptorTable(2, srvHandle);
		cmd->SetComputeRootDescriptorTable(3, uavParticle);
		cmd->SetComputeRootDescriptorTable(4, uavMeta);
		cmd->SetComputeRootDescriptorTable(5, uavCount);
		cmd->SetComputeRootDescriptorTable(6, uavTable);
		cmd->Dispatch(buildGroups, 1, 1);

		CD3DX12_RESOURCE_BARRIER gridBarriers[] = {
				CD3DX12_RESOURCE_BARRIER::UAV(m_gridCount.Get()),
				CD3DX12_RESOURCE_BARRIER::UAV(m_gridTable.Get())
		};
		cmd->ResourceBarrier(2, gridBarriers);

		// ルートパラメーター: b0, b1, t0, u0, u1
		// 定数バッファ更新
		if (m_sphParamCB && m_sphParamCB->IsValid()) {
			auto* cb = m_sphParamCB->GetPtr<SPHParamsCB>();
			cb->timeStep = dt;
			cb->particleCount = m_maxParticles;
		}

		if (m_viewProjCB && m_viewProjCB->IsValid()) {
			auto* cam = g_Engine->GetObj<Camera>("Camera");
			if (cam) {
				auto mat = cam->GetViewMatrix() * cam->GetProjMatrix();
				auto* cb = m_viewProjCB->GetPtr<ViewProjCB>();
				DirectX::XMStoreFloat4x4(&cb->viewProj, mat);
			}
		}

		cmd->SetComputeRootConstantBufferView(0, m_sphParamCB->GetAddress());
		cmd->SetComputeRootConstantBufferView(1, m_viewProjCB->GetAddress());
		cmd->SetComputeRootDescriptorTable(2, srvHandle);
		cmd->SetComputeRootDescriptorTable(3, uavParticle);
		cmd->SetComputeRootDescriptorTable(4, uavMeta);
		cmd->SetComputeRootDescriptorTable(5, uavCount);
		cmd->SetComputeRootDescriptorTable(6, uavTable);
		cmd->SetPipelineState(m_computePS.Get());
		UINT groups = (m_maxParticles + 255) / 256;
		cmd->Dispatch(groups, 1, 1);

		auto toSrv = CD3DX12_RESOURCE_BARRIER::Transition(
			m_metaBuffer.Get(),
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
			D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		cmd->ResourceBarrier(1, &toSrv);
		m_metaInSrvState = true;
	}
	else {
		// CPU シミュレーション
		const float PI = 3.14159265358979323846f;
		float radius = m_params.radius;
		float radius2 = radius * radius;
		float radius6 = radius2 * radius2 * radius2;
		float radius9 = radius6 * radius2 * radius;

		m_grid.Clear();
		for (UINT i = 0; i < m_maxParticles; ++i) {
			m_grid.Insert(i, m_cpuParticles[i].position);
		}

		std::fill(m_density.begin(), m_density.end(), 0.0f);
		auto& neigh = m_neighborBuffer;
		for (UINT i = 0; i < m_maxParticles; ++i) {
			m_grid.Query(m_cpuParticles[i].position, radius, neigh);
			float d = 0.f;
			for (size_t j : neigh) {
				DirectX::XMFLOAT3 rij{
						m_cpuParticles[i].position.x - m_cpuParticles[j].position.x,
						m_cpuParticles[i].position.y - m_cpuParticles[j].position.y,
						m_cpuParticles[i].position.z - m_cpuParticles[j].position.z };
				float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
				if (r2 < radius2) {
					float x = radius2 - r2;
					d += m_params.particleMass * (315.0f / (64.0f * PI * radius9)) * x * x * x;
				}
			}
			m_density[i] = d;
		}

		for (UINT i = 0; i < m_maxParticles; ++i) {

			m_grid.Query(m_cpuParticles[i].position, radius, neigh);    // 近傍に限定
			float pressure = m_params.stiffness * (m_density[i] - m_params.restDensity);
			DirectX::XMFLOAT3 force{ 0.0f, -9.8f * m_density[i], 0.0f };

			// 近傍粒子からの影響範囲内を計算
			for (size_t j : neigh) {
				if (j == i) continue;
				DirectX::XMFLOAT3 rij{
						m_cpuParticles[i].position.x - m_cpuParticles[j].position.x,
						m_cpuParticles[i].position.y - m_cpuParticles[j].position.y,
						m_cpuParticles[i].position.z - m_cpuParticles[j].position.z };
				float r = std::sqrt(rij.x * rij.x + rij.y * rij.y + rij.z * rij.z);
				if (r > 0 && r < radius) {
					float coeff = -45.0f / (PI * radius6) * (radius - r) * (radius - r);
					DirectX::XMFLOAT3 grad{ coeff * rij.x / r, coeff * rij.y / r, coeff * rij.z / r };
					float pTerm = (pressure + m_params.stiffness * (m_density[j] - m_params.restDensity)) / (2 * m_density[j]);
					force.x += -m_params.particleMass * pTerm * grad.x;
					force.y += -m_params.particleMass * pTerm * grad.y;
					force.z += -m_params.particleMass * pTerm * grad.z;
					float lap = 45.0f / (PI * radius6) * (radius - r);
					force.x += m_params.viscosity * m_params.particleMass * (m_cpuParticles[j].velocity.x - m_cpuParticles[i].velocity.x) * (lap / m_density[j]);
					force.y += m_params.viscosity * m_params.particleMass * (m_cpuParticles[j].velocity.y - m_cpuParticles[i].velocity.y) * (lap / m_density[j]);
					force.z += m_params.viscosity * m_params.particleMass * (m_cpuParticles[j].velocity.z - m_cpuParticles[i].velocity.z) * (lap / m_density[j]);
				}
			}

			// 速度、位置更新
			float invD = 1.0f / m_density[i];
			m_cpuParticles[i].velocity.x += force.x * invD * m_params.timeStep;
			m_cpuParticles[i].velocity.y += force.y * invD * m_params.timeStep;
			m_cpuParticles[i].velocity.z += force.z * invD * m_params.timeStep;

			m_cpuParticles[i].position.x += m_cpuParticles[i].velocity.x * m_params.timeStep;
			m_cpuParticles[i].position.y += m_cpuParticles[i].velocity.y * m_params.timeStep;
			m_cpuParticles[i].position.z += m_cpuParticles[i].velocity.z * m_params.timeStep;

			if (m_cpuParticles[i].position.x < -1 || m_cpuParticles[i].position.x > 1) m_cpuParticles[i].velocity.x *= -0.1f;
			if (m_cpuParticles[i].position.y < -1 || m_cpuParticles[i].position.y > 5) m_cpuParticles[i].velocity.y *= -0.1f;
			if (m_cpuParticles[i].position.z < -1 || m_cpuParticles[i].position.z > 1) m_cpuParticles[i].velocity.z *= -0.1f;
		}

		// CPU→GPU 転送 (particles)
		auto toCopyParticle = CD3DX12_RESOURCE_BARRIER::Transition(
			m_particleBuffer.Get(),
			m_particleInSrvState ?
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
			D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE :
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			D3D12_RESOURCE_STATE_COPY_DEST);
		auto toCopyMeta = CD3DX12_RESOURCE_BARRIER::Transition(
			m_metaBuffer.Get(),
			m_metaInSrvState ?
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
			D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE :
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			D3D12_RESOURCE_STATE_COPY_DEST);
		CD3DX12_RESOURCE_BARRIER preCopy[] = { toCopyParticle, toCopyMeta };
		cmd->ResourceBarrier(2, preCopy);

		D3D12_SUBRESOURCE_DATA srcParticle = {};
		srcParticle.pData = m_cpuParticles.data();
		srcParticle.RowPitch = sizeof(FluidParticle) * m_maxParticles;
		srcParticle.SlicePitch = srcParticle.RowPitch;
		UpdateSubresources<1>(cmd, m_particleBuffer.Get(), m_particleUpload.Get(), 0, 0, 1, &srcParticle);

		std::vector<ParticleMeta> metas(m_maxParticles);
		for (UINT i = 0; i < m_maxParticles; ++i) {
			metas[i].pos = m_cpuParticles[i].position;
			metas[i].r = radius;
		}
		D3D12_SUBRESOURCE_DATA srcMeta = {};
		srcMeta.pData = metas.data();
		srcMeta.RowPitch = sizeof(ParticleMeta) * m_maxParticles;
		srcMeta.SlicePitch = srcMeta.RowPitch;
		UpdateSubresources<1>(cmd, m_metaBuffer.Get(), m_metaUpload.Get(), 0, 0, 1, &srcMeta);

		auto toSrvParticle = CD3DX12_RESOURCE_BARRIER::Transition(
			m_particleBuffer.Get(),
			D3D12_RESOURCE_STATE_COPY_DEST,
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
			D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		auto toSrvMeta = CD3DX12_RESOURCE_BARRIER::Transition(
			m_metaBuffer.Get(),
			D3D12_RESOURCE_STATE_COPY_DEST,
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
			D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		CD3DX12_RESOURCE_BARRIER postCopy[] = { toSrvParticle, toSrvMeta };
		cmd->ResourceBarrier(2, postCopy);
		m_particleInSrvState = true;
		m_metaInSrvState = true;
	}
}

// 描画
void FluidSystem::Render(ID3D12GraphicsCommandList* cmd, const DirectX::XMFLOAT4X4& invViewProj, const DirectX::XMFLOAT3& camPos, float isoLevel) {
	// 定数バッファ更新
	struct MetaCB {
		DirectX::XMFLOAT4X4 invVP;
		DirectX::XMFLOAT3 cam;
		float iso;
		UINT count;
		DirectX::XMFLOAT3 pad;
	} cb;
	cb.invVP = invViewProj;
	cb.cam = camPos;
	cb.iso = isoLevel;
	cb.count = m_maxParticles;
	cb.pad = DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);
	void* p;
	m_graphicsCB->Map(0, nullptr, &p);
	memcpy(p, &cb, sizeof(cb));
	m_graphicsCB->Unmap(0, nullptr);

	if (m_useScreenSpace) { RenderSSA(cmd); return; }


	// 描画
	cmd->SetDescriptorHeaps(1, m_graphicsSrvHeap.GetAddressOf());
	cmd->SetGraphicsRootSignature(m_graphicsRS.Get());
	cmd->SetPipelineState(m_graphicsPS->Get());
	cmd->SetGraphicsRootDescriptorTable(0, m_graphicsSrvHeap->GetGPUDescriptorHandleForHeapStart());
	cmd->SetGraphicsRootConstantBufferView(1, m_graphicsCB->GetGPUVirtualAddress());
	cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	cmd->DrawInstanced(3, 1, 0, 0);
}

// マウスクリックでパーティクルを選択
void FluidSystem::StartDrag(int mouseX, int mouseY, Camera* cam)
{
	if (m_useGpu) return; // GPUシミュレーション時は未対応

	UINT width = g_Engine->FrameBufferWidth();
	UINT height = g_Engine->FrameBufferHeight();
	auto view = cam->GetViewMatrix();
	auto proj = cam->GetProjMatrix();
	auto viewProj = view * proj;

	float best = 15.0f * 15.0f; // 距離閾値(px^2)
	m_dragIndex = -1;
	using namespace DirectX;
	for (UINT i = 0; i < m_maxParticles; ++i) {
		XMVECTOR p = XMLoadFloat3(&m_cpuParticles[i].position);
		XMVECTOR clip = XMVector3TransformCoord(p, viewProj);
		float sx = (XMVectorGetX(clip) * 0.5f + 0.5f) * static_cast<float>(width);
		float sy = (-XMVectorGetY(clip) * 0.5f + 0.5f) * static_cast<float>(height);
		float dx = sx - static_cast<float>(mouseX);
		float dy = sy - static_cast<float>(mouseY);
		float dist = dx * dx + dy * dy;
		if (dist < best) {
			best = dist;
			m_dragIndex = static_cast<int>(i);
			XMVECTOR eye = cam->GetEyePos();
			m_dragDepth = XMVectorGetX(XMVector3Length(p - eye));
		}
	}
}

// マウスドラッグでパーティクルを移動
void FluidSystem::Drag(int mouseX, int mouseY, Camera* cam)
{
	if (m_useGpu) return;        // GPUシミュレーション時は未対応
	if (m_dragIndex < 0) return; // クリックでパーティクルが選択されていない

	UINT width = g_Engine->FrameBufferWidth();
	UINT height = g_Engine->FrameBufferHeight();
	auto invVP = cam->GetInvViewProj();
	using namespace DirectX;
	XMMATRIX inv = XMLoadFloat4x4(&invVP);

	float x = (2.0f * mouseX / static_cast<float>(width)) - 1.0f;
	float y = 1.0f - (2.0f * mouseY / static_cast<float>(height));
	XMVECTOR nearP = XMVector3TransformCoord(XMVectorSet(x, y, 0.0f, 1.0f), inv);
	XMVECTOR farP = XMVector3TransformCoord(XMVectorSet(x, y, 1.0f, 1.0f), inv);
	XMVECTOR dir = XMVector3Normalize(farP - nearP);
	XMVECTOR eye = cam->GetEyePos();
	XMVECTOR newPos = eye + dir * m_dragDepth;
	XMVECTOR curPos = XMLoadFloat3(&m_cpuParticles[m_dragIndex].position);
	XMVECTOR delta = newPos - curPos;
	XMVECTOR vel = XMLoadFloat3(&m_cpuParticles[m_dragIndex].velocity);

	// パーティクルを移動
	vel += delta * 0.1f; // 調整用係数
	XMStoreFloat3(&m_cpuParticles[m_dragIndex].velocity, vel);

	// パーティクルを引っ張る
	float dragRadius = 0.2f; // パーティクル操作の影響範囲
	float radius2 = dragRadius * dragRadius;
	for (UINT i = 0; i < m_maxParticles; ++i) {
		if (i == static_cast<UINT>(m_dragIndex)) continue;
		XMVECTOR pos = XMLoadFloat3(&m_cpuParticles[i].position);
		XMVECTOR diff = pos - curPos;
		float dist2 = XMVectorGetX(XMVector3LengthSq(diff));
		if (dist2 < radius2) {
			float dist = std::sqrt(dist2);
			float w = 1.0f - (dist / dragRadius);
			XMVECTOR v = XMLoadFloat3(&m_cpuParticles[i].velocity);
			v += delta * (0.1f * w);
			XMStoreFloat3(&m_cpuParticles[i].velocity, v);
		}
	}
}

// ドラッグ終了
void FluidSystem::EndDrag()
{
	m_dragIndex = -1;
}


static void CreateUploadCB(ID3D12Device* device, size_t size, Microsoft::WRL::ComPtr<ID3D12Resource>& cb)
{
	CD3DX12_HEAP_PROPERTIES hp(D3D12_HEAP_TYPE_UPLOAD);
	CD3DX12_RESOURCE_DESC   rd = CD3DX12_RESOURCE_DESC::Buffer((size + 255) & ~255);
	device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
		D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&cb));
}

void FluidSystem::DestroySSAResources()
{
	// --- RT / SRV ヒープ / CB ---
	m_accumTex.Reset();
	m_blurTex.Reset();

	m_rtvHeapSSA.Reset();
	m_srvHeapSSA.Reset();

	m_cbAccum.Reset();
	m_cbBlur.Reset();
	m_cbComposite.Reset();

	// --- Root Signatures ---
	m_rsAccum.Reset();
	m_rsBlur.Reset();
	m_rsComposite.Reset();

	// --- 無効値に戻す ---
	m_accumRTV = {};
	m_blurRTV = {};

	m_accumSRV = {};
	m_blurSRV = {};
	m_particleSRV = {};
}

void FluidSystem::CreateSSAResources(ID3D12Device* device, DXGI_FORMAT mainRTFormat, UINT viewW, UINT viewH)
{
	DestroySSAResources();

	UINT w = max(1u, viewW / m_ssaScale);
	UINT h = max(1u, viewH / m_ssaScale);

	// RTVヒープ（accum, blur）
	D3D12_DESCRIPTOR_HEAP_DESC rtvHd = {};
	rtvHd.NumDescriptors = 2;
	rtvHd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	rtvHd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	device->CreateDescriptorHeap(&rtvHd, IID_PPV_ARGS(&m_rtvHeapSSA));
	auto rtvStart = m_rtvHeapSSA->GetCPUDescriptorHandleForHeapStart();
	UINT rtvInc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
	m_accumRTV = rtvStart;
	m_blurRTV = { rtvStart.ptr + rtvInc };

	// SRVヒープ（GPU可視）：（accumSRV, blurSRV）
	D3D12_DESCRIPTOR_HEAP_DESC srvHd = {};
	srvHd.NumDescriptors = 2;
	srvHd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	srvHd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	device->CreateDescriptorHeap(&srvHd, IID_PPV_ARGS(&m_srvHeapSSA));
	auto srvStart = m_srvHeapSSA->GetGPUDescriptorHandleForHeapStart();
	UINT srvInc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	m_accumSRV = srvStart;
	m_blurSRV = { srvStart.ptr + srvInc };

	// 2Dテクスチャ作成（R16_FLOAT）
	DXGI_FORMAT accumFmt = m_mainRTFormat;
        CD3DX12_RESOURCE_DESC rd = CD3DX12_RESOURCE_DESC::Tex2D(accumFmt, w, h, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        FLOAT clearV[4] = { 0,0,0,0 };
	D3D12_CLEAR_VALUE cv = { accumFmt, {clearV[0],clearV[1],clearV[2],clearV[3]} };
	CD3DX12_HEAP_PROPERTIES hp(D3D12_HEAP_TYPE_DEFAULT);

	device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &cv, IID_PPV_ARGS(&m_accumTex));
	device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &cv, IID_PPV_ARGS(&m_blurTex));

	// RTV
	device->CreateRenderTargetView(m_accumTex.Get(), nullptr, m_accumRTV);
	device->CreateRenderTargetView(m_blurTex.Get(), nullptr, m_blurRTV);

	// SRV
	D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
	sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	sd.Format = accumFmt;
	sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	sd.Texture2D.MipLevels = 1;

	// CPU側ハンドルが必要なら別途CPUヒープを用意してください。ここではGPUハンドルだけを保持します。
	auto srvCPU = CD3DX12_CPU_DESCRIPTOR_HANDLE(
		m_srvHeapSSA->GetCPUDescriptorHandleForHeapStart());
	device->CreateShaderResourceView(m_accumTex.Get(), &sd, srvCPU);
	srvCPU.ptr += srvInc;
	device->CreateShaderResourceView(m_blurTex.Get(), &sd, srvCPU);

	// 定数バッファ
	CreateUploadCB(device, sizeof(float) * (16 + 16 + 4 + 4), m_cbAccum);     // AccumCB
	CreateUploadCB(device, sizeof(float) * 4, m_cbBlur);   // BlurCB
	CreateUploadCB(device, sizeof(float) * 8, m_cbComposite); // CompositeCB
}

//void FluidSystem::CreateSSAPipelines(ID3D12Device* device, DXGI_FORMAT /*accumFormat*/)
//{
//    using Microsoft::WRL::ComPtr;
//
//    // ------------------------------------------------------------
//    // Root Signatures
//    // ------------------------------------------------------------
//
//    // [Accum] VSでb0(CB), t0(SRV:粒子メタ)を読む
//    {
//        CD3DX12_ROOT_PARAMETER rp[2];
//        rp[0].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_VERTEX); // b0 (VS)
//        CD3DX12_DESCRIPTOR_RANGE range{};
//        range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0);                 // t0
//        rp[1].InitAsDescriptorTable(1, &range, D3D12_SHADER_VISIBILITY_VERTEX);
//
//        // サンプラは使わないので0本
//        CD3DX12_ROOT_SIGNATURE_DESC rsd(
//            _countof(rp), rp,
//            0, nullptr,
//            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
//        );
//
//        ComPtr<ID3DBlob> sig, err;
//        HRESULT hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
//        if (FAILED(hr)) { if (err) OutputDebugStringA((char*)err->GetBufferPointer()); throw std::runtime_error("RS serialize (Accum)"); }
//        hr = device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_rsAccum));
//        if (FAILED(hr)) throw std::runtime_error("CreateRootSignature (Accum)");
//    }
//
//    // [Blur] PSでb0(CB:Texel/Dir), t0(SRV:Src), s0(sampler)を読む
//    {
//        CD3DX12_ROOT_PARAMETER rp[2];
//        rp[0].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_PIXEL); // b0 (PS)
//        CD3DX12_DESCRIPTOR_RANGE range{};
//        range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0);               // t0
//        rp[1].InitAsDescriptorTable(1, &range, D3D12_SHADER_VISIBILITY_PIXEL);
//
//        D3D12_STATIC_SAMPLER_DESC ss{};
//        ss.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
//        ss.AddressU = ss.AddressV = ss.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
//        ss.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
//        ss.ShaderRegister = 0;
//
//        CD3DX12_ROOT_SIGNATURE_DESC rsd(
//            _countof(rp), rp,
//            1, &ss,
//            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
//        );
//
//        ComPtr<ID3DBlob> sig, err;
//        HRESULT hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
//        if (FAILED(hr)) { if (err) OutputDebugStringA((char*)err->GetBufferPointer()); throw std::runtime_error("RS serialize (Blur)"); }
//        hr = device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_rsBlur));
//        if (FAILED(hr)) throw std::runtime_error("CreateRootSignature (Blur)");
//    }
//
//    // [Composite] PSでb0(CB:Threshold等), t0(SRV:Density), s0(sampler)
//    {
//        CD3DX12_ROOT_PARAMETER rp[2];
//        rp[0].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_PIXEL); // b0 (PS)
//        CD3DX12_DESCRIPTOR_RANGE range{};
//        range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0);               // t0
//        rp[1].InitAsDescriptorTable(1, &range, D3D12_SHADER_VISIBILITY_PIXEL);
//
//        D3D12_STATIC_SAMPLER_DESC ss{};
//        ss.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
//        ss.AddressU = ss.AddressV = ss.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
//        ss.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
//        ss.ShaderRegister = 0;
//
//        CD3DX12_ROOT_SIGNATURE_DESC rsd(
//            _countof(rp), rp,
//            1, &ss,
//            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
//        );
//
//        ComPtr<ID3DBlob> sig, err;
//        HRESULT hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
//        if (FAILED(hr)) { if (err) OutputDebugStringA((char*)err->GetBufferPointer()); throw std::runtime_error("RS serialize (Composite)"); }
//        hr = device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_rsComposite));
//        if (FAILED(hr)) throw std::runtime_error("CreateRootSignature (Composite)");
//    }
//
//    // ------------------------------------------------------------
//    //  PipelineStates
//    // ------------------------------------------------------------
//    D3D12_INPUT_LAYOUT_DESC emptyLayout{};
//    emptyLayout.pInputElementDescs = nullptr;
//    emptyLayout.NumElements = 0;
//
//    // Accum（粒子→低解像度RTへ加算）
//    {
//        m_psoAccum = std::make_unique<PipelineState>();
//        m_psoAccum->SetInputLayout(emptyLayout);
//        m_psoAccum->SetRootSignature(m_rsAccum.Get());
//        m_psoAccum->SetVS(L"AccumBillboardVS.cso");
//        m_psoAccum->SetPS(L"AccumBillboardPS.cso");
//        m_psoAccum->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
//        if (!m_psoAccum->IsValid()) throw std::runtime_error("Accum PSO invalid");
//    }
//
//    // Blur（横/縦 共通。DirをCBで切り替えて同じPSOを使う）
//    {
//        m_psoBlur = std::make_unique<PipelineState>();
//        m_psoBlur->SetInputLayout(emptyLayout);
//        m_psoBlur->SetRootSignature(m_rsBlur.Get());
//        m_psoBlur->SetVS(L"FullscreenVS.cso");
//        m_psoBlur->SetPS(L"BlurPS.cso");
//        m_psoBlur->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
//        if (!m_psoBlur->IsValid()) throw std::runtime_error("Blur PSO invalid");
//    }
//
//    // Composite（密度→最終RTへ合成）
//    {
//        m_psoComposite = std::make_unique<PipelineState>();
//        m_psoComposite->SetInputLayout(emptyLayout);
//        m_psoComposite->SetRootSignature(m_rsComposite.Get());
//        m_psoComposite->SetVS(L"FullscreenVS.cso");
//        m_psoComposite->SetPS(L"CompositePS.cso");
//        m_psoComposite->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
//        if (!m_psoComposite->IsValid()) throw std::runtime_error("Composite PSO invalid");
//    }
//}

void FluidSystem::CreateSSAPipelines(ID3D12Device* device, DXGI_FORMAT)
{
	using Microsoft::WRL::ComPtr;

	// ===== RootSignatures =====
	// Accum: VS で b0(CB) / t0(StructuredBuffer:粒子)
	{
		CD3DX12_ROOT_PARAMETER rp[2];
		rp[0].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_VERTEX);
		CD3DX12_DESCRIPTOR_RANGE range{}; range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0);
		rp[1].InitAsDescriptorTable(1, &range, D3D12_SHADER_VISIBILITY_VERTEX);

		CD3DX12_ROOT_SIGNATURE_DESC rsd(_countof(rp), rp, 0, nullptr,
			D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		ComPtr<ID3DBlob> sig, err;
		HRESULT hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
		if (FAILED(hr)) { if (err) OutputDebugStringA((char*)err->GetBufferPointer()); return; }
		hr = device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_rsAccum));
		if (FAILED(hr)) return;
	}

	// Blur: PS で b0(CB:Texel/Dir) / t0(SRV:Src) / s0
	{
		CD3DX12_ROOT_PARAMETER rp[2];
		rp[0].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_PIXEL);
		CD3DX12_DESCRIPTOR_RANGE range{}; range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0);
		rp[1].InitAsDescriptorTable(1, &range, D3D12_SHADER_VISIBILITY_PIXEL);

		D3D12_STATIC_SAMPLER_DESC ss{};
		ss.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
		ss.AddressU = ss.AddressV = ss.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
		ss.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
		ss.ShaderRegister = 0;

		CD3DX12_ROOT_SIGNATURE_DESC rsd(_countof(rp), rp, 1, &ss,
			D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		ComPtr<ID3DBlob> sig, err;
		HRESULT hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
		if (FAILED(hr)) { if (err) OutputDebugStringA((char*)err->GetBufferPointer()); return; }
		hr = device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_rsBlur));
		if (FAILED(hr)) return;
	}

	// Composite: PS で b0(CB:Threshold等) / t0(SRV:Density) / s0
	{
		CD3DX12_ROOT_PARAMETER rp[2];
		rp[0].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_PIXEL);
		CD3DX12_DESCRIPTOR_RANGE range{}; range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0);
		rp[1].InitAsDescriptorTable(1, &range, D3D12_SHADER_VISIBILITY_PIXEL);

		D3D12_STATIC_SAMPLER_DESC ss{};
		ss.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
		ss.AddressU = ss.AddressV = ss.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
		ss.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
		ss.ShaderRegister = 0;

		CD3DX12_ROOT_SIGNATURE_DESC rsd(_countof(rp), rp, 1, &ss,
			D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		ComPtr<ID3DBlob> sig, err;
		HRESULT hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
		if (FAILED(hr)) { if (err) OutputDebugStringA((char*)err->GetBufferPointer()); return; }
		hr = device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_rsComposite));
		if (FAILED(hr)) return;
	}

	// ===== PSO =====
	// Accum（加算）: 出力RT = R16_FLOAT
	m_psoAccum.SetRootSignature(m_rsAccum.Get());
	m_psoAccum.SetShaders(L"AccumBillboardVS.cso", L"AccumBillboardPS.cso");
	m_psoAccum.SetFormats(DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_UNKNOWN);
	m_psoAccum.SetBlend(FullscreenPSO::Blend::Additive);
	m_psoAccum.SetTopology(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
	if (!m_psoAccum.Create(device)) return;

	// Blur（通常）: 出力RT = R16_FLOAT
	m_psoBlur.SetRootSignature(m_rsBlur.Get());
	m_psoBlur.SetShaders(L"FullscreenVS.cso", L"BlurPS.cso");
	m_psoBlur.SetFormats(DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_UNKNOWN);
	m_psoBlur.SetBlend(FullscreenPSO::Blend::Opaque);
	m_psoBlur.SetTopology(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
	if (!m_psoBlur.Create(device)) return;

	// Composite（不透明 or α）: 出力RT = 最終RTフォーマット
	m_psoComposite.SetRootSignature(m_rsComposite.Get());
	m_psoComposite.SetShaders(L"FullscreenVS.cso", L"CompositePS.cso");
	m_psoComposite.SetFormats(m_mainRTFormat, DXGI_FORMAT_UNKNOWN);
	m_psoComposite.SetBlend(FullscreenPSO::Blend::Alpha); // 透過合成ならAlpha/Opaqueは不透明
	m_psoComposite.SetTopology(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
	if (!m_psoComposite.Create(device)) return;
}

void FluidSystem::UpdateSSAConstantBuffers(ID3D12GraphicsCommandList* cmd)
{
	// AccumCB への書き込み
	struct AccumCB {
		float View[16]; float Proj[16];
		float CameraRight[4];
		float CameraUp[4];
		float ViewportSize[2]; float _padA[2];
		float PixelScale; float _padB[3];
	} acb = {};

	// 既存のカメラから行列と Right/Up を取得してください（プロジェクト依存）
	// 下は疑似コード：
	// auto V = camera->GetViewMatrix(); auto P = camera->GetProjMatrix();
	// auto right = camera->GetRight();  auto up = camera->GetUp();

	// acb.View[...] = ...
	// acb.Proj[...] = ...
	// acb.CameraRight[0..2] = right; acb.CameraUp[0..2] = up;

	acb.ViewportSize[0] = float(max(1u, m_viewWidth / m_ssaScale));
	acb.ViewportSize[1] = float(max(1u, m_viewHeight / m_ssaScale));
	acb.PixelScale = 8.0f; // 画面半径スケール（必要に応じて調整）

	void* p; m_cbAccum->Map(0, nullptr, &p); memcpy(p, &acb, sizeof(acb)); m_cbAccum->Unmap(0, nullptr);

	// BlurCB
	struct BlurCB { float TexelSize[2]; float Dir[2]; } bcb{};
	bcb.TexelSize[0] = 1.0f / acb.ViewportSize[0];
	bcb.TexelSize[1] = 1.0f / acb.ViewportSize[1];
	// Dirは描画直前に (1,0) と (0,1) をそれぞれ設定
	// ここではデフォルトだけ入れておく
	bcb.Dir[0] = 1.0f; bcb.Dir[1] = 0.0f;
	m_cbBlur->Map(0, nullptr, &p); memcpy(p, &bcb, sizeof(bcb)); m_cbBlur->Unmap(0, nullptr);

	// CompositeCB
	struct CompositeCB { float Threshold; float Softness; float BaseColor[3]; float Opacity; } ccb{};
	ccb.Threshold = 0.25f; ccb.Softness = 0.10f;
	ccb.BaseColor[0] = 0.25f; ccb.BaseColor[1] = 0.5f; ccb.BaseColor[2] = 1.0f;
	ccb.Opacity = 1.0f;
	m_cbComposite->Map(0, nullptr, &p); memcpy(p, &ccb, sizeof(ccb)); m_cbComposite->Unmap(0, nullptr);
}

// 画面蓄積の描画一式
void FluidSystem::RenderSSA(ID3D12GraphicsCommandList* cmd)
{
	// accum
	{
		// Dispatch直後：UAV→SRV
		auto toSRVParticles = CD3DX12_RESOURCE_BARRIER::Transition(
			m_metaBuffer.Get(),
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		cmd->ResourceBarrier(1, &toSRVParticles);

		// accumulate へ
		auto toRTaccum = CD3DX12_RESOURCE_BARRIER::Transition(
			m_accumTex.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
		cmd->ResourceBarrier(1, &toRTaccum);

		const float CLEAR0[4] = { 0,0,0,0 };
		cmd->OMSetRenderTargets(1, &m_accumRTV, FALSE, nullptr);
		cmd->ClearRenderTargetView(m_accumRTV, CLEAR0, 0, nullptr);

		cmd->SetPipelineState(m_psoAccum.Get());
		cmd->SetGraphicsRootSignature(m_rsAccum.Get());
		cmd->SetGraphicsRootConstantBufferView(0, m_cbAccum->GetGPUVirtualAddress());
               ID3D12DescriptorHeap* heaps[] = { m_graphicsSrvHeap.Get() };
               cmd->SetDescriptorHeaps(1, heaps);
               cmd->SetGraphicsRootDescriptorTable(1, m_particleSRV); // ★ 粒子SRV

		cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
		cmd->DrawInstanced(4, m_maxParticles, 0, 0);

		auto backToSRV = CD3DX12_RESOURCE_BARRIER::Transition(
			m_accumTex.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		cmd->ResourceBarrier(1, &backToSRV);
	}


	// 横（Src=accum → Dst=blur）
	{
		auto toRT = CD3DX12_RESOURCE_BARRIER::Transition(
			m_blurTex.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
		cmd->ResourceBarrier(1, &toRT);

		const float CLEAR0[4] = { 0,0,0,0 };
		cmd->OMSetRenderTargets(1, &m_blurRTV, FALSE, nullptr);
		cmd->ClearRenderTargetView(m_blurRTV, CLEAR0, 0, nullptr);

		D3D12_VIEWPORT vp{ 0,0,(float)(m_viewWidth / m_ssaScale),(float)(m_viewHeight / m_ssaScale),0,1 };
		D3D12_RECT sc{ 0,0,(LONG)(m_viewWidth / m_ssaScale),(LONG)(m_viewHeight / m_ssaScale) };
		cmd->RSSetViewports(1, &vp);
		cmd->RSSetScissorRects(1, &sc);

		// CB: Dir=(1,0)
		struct BlurCB { float TexelSize[2]; float Dir[2]; } b{};
		b.TexelSize[0] = 1.0f / (m_viewWidth / m_ssaScale);
		b.TexelSize[1] = 1.0f / (m_viewHeight / m_ssaScale);
		b.Dir[0] = 1.0f; b.Dir[1] = 0.0f;
		void* p; m_cbBlur->Map(0, nullptr, &p); memcpy(p, &b, sizeof(b)); m_cbBlur->Unmap(0, nullptr);

		cmd->SetPipelineState(m_psoBlur.Get());
		cmd->SetGraphicsRootSignature(m_rsBlur.Get());
		ID3D12DescriptorHeap* heaps[] = { m_srvHeapSSA.Get() };
		cmd->SetDescriptorHeaps(1, heaps);
		cmd->SetGraphicsRootConstantBufferView(0, m_cbBlur->GetGPUVirtualAddress());
		cmd->SetGraphicsRootDescriptorTable(1, m_accumSRV);

		// 念のため再度RTへ（状態が怪しいときの保険）
		auto fixRT = CD3DX12_RESOURCE_BARRIER::Transition(
			m_blurTex.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
		cmd->ResourceBarrier(1, &fixRT);

		cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		cmd->DrawInstanced(3, 1, 0, 0);

		auto toSRV = CD3DX12_RESOURCE_BARRIER::Transition(
			m_blurTex.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		cmd->ResourceBarrier(1, &toSRV);
	}

	// 縦（Src=blur → Dst=accum）
	{
		auto toRT2 = CD3DX12_RESOURCE_BARRIER::Transition(
			m_accumTex.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
		cmd->ResourceBarrier(1, &toRT2);

		const float CLEAR0[4] = { 0,0,0,0 };
		cmd->OMSetRenderTargets(1, &m_accumRTV, FALSE, nullptr);
		cmd->ClearRenderTargetView(m_accumRTV, CLEAR0, 0, nullptr);

		// CB: Dir=(0,1)
		struct BlurCB { float TexelSize[2]; float Dir[2]; } b{};
		b.TexelSize[0] = 1.0f / (m_viewWidth / m_ssaScale);
		b.TexelSize[1] = 1.0f / (m_viewHeight / m_ssaScale);
		b.Dir[0] = 0.0f; b.Dir[1] = 1.0f;
		void* p; m_cbBlur->Map(0, nullptr, &p); memcpy(p, &b, sizeof(b)); m_cbBlur->Unmap(0, nullptr);

		cmd->SetPipelineState(m_psoBlur.Get());
		cmd->SetGraphicsRootSignature(m_rsBlur.Get());
		ID3D12DescriptorHeap* heaps[] = { m_srvHeapSSA.Get() };
		cmd->SetDescriptorHeaps(1, heaps);
		cmd->SetGraphicsRootConstantBufferView(0, m_cbBlur->GetGPUVirtualAddress());
		cmd->SetGraphicsRootDescriptorTable(1, m_blurSRV);

		auto fixRT2 = CD3DX12_RESOURCE_BARRIER::Transition(
			m_accumTex.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
		cmd->ResourceBarrier(1, &fixRT2);

		cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		cmd->DrawInstanced(3, 1, 0, 0);

		auto toSRV2 = CD3DX12_RESOURCE_BARRIER::Transition(
			m_accumTex.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		cmd->ResourceBarrier(1, &toSRV2);
	}


	// 合成
	{
		// 最終RTへ（外側のレンダリングループでセット済みなら不要）
		cmd->SetPipelineState(m_psoComposite.Get());
		cmd->SetGraphicsRootSignature(m_rsComposite.Get());
		cmd->SetGraphicsRootConstantBufferView(0, m_cbComposite->GetGPUVirtualAddress());
		ID3D12DescriptorHeap* heaps2[] = { m_srvHeapSSA.Get() };
		cmd->SetDescriptorHeaps(1, heaps2);
		cmd->SetGraphicsRootDescriptorTable(1, m_accumSRV);

		cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		cmd->DrawInstanced(3, 1, 0, 0);
	}
}
