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

        // UAVヒープのハンドルを取得
        UINT uavHandleSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE uavHandle = m_uavHeap->GetCPUDescriptorHandleForHeapStart();

	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srvDesc.Buffer.NumElements = maxParticles;
	srvDesc.Buffer.StructureByteStride = sizeof(FluidParticle);
	srvDesc.Format = DXGI_FORMAT_UNKNOWN;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        device->CreateShaderResourceView(m_particleBuffer.Get(), &srvDesc, uavHandle);

        uavHandle.ptr += uavHandleSize;
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavd = {};
	uavd.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavd.Buffer.NumElements = maxParticles;
	uavd.Buffer.StructureByteStride = sizeof(FluidParticle);
        device->CreateUnorderedAccessView(m_particleBuffer.Get(), nullptr, &uavd, uavHandle);

        uavHandle.ptr += uavHandleSize;
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavMeta = {};
	uavMeta.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavMeta.Buffer.NumElements = maxParticles;
	uavMeta.Buffer.StructureByteStride = sizeof(ParticleMeta);
        device->CreateUnorderedAccessView(m_metaBuffer.Get(), nullptr, &uavMeta, uavHandle);

        uavHandle.ptr += uavHandleSize;

        // --------------------------------------------------
        // Compute用定数バッファとグリッド関連リソースの生成
        // --------------------------------------------------
        m_sphParamCB = new ConstantBuffer(sizeof(SPHParamsCB));
        m_viewProjCB = new ConstantBuffer(sizeof(ViewProjCB));

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

                // グリッド用バッファを生成
                device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &rdCount,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                        IID_PPV_ARGS(&m_gridCount));
                device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &rdTable,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                        IID_PPV_ARGS(&m_gridTable));

                // UAVを連続したハンドルで作成
                D3D12_CPU_DESCRIPTOR_HANDLE gridHandle = m_uavHeap->GetCPUDescriptorHandleForHeapStart();
                gridHandle.ptr += uavHandleSize * 3;
                D3D12_UNORDERED_ACCESS_VIEW_DESC uavCount = {};
                uavCount.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                uavCount.Buffer.NumElements = m_cellCount;
                uavCount.Buffer.StructureByteStride = sizeof(UINT);
                device->CreateUnorderedAccessView(m_gridCount.Get(), nullptr, &uavCount, gridHandle);

                gridHandle.ptr += uavHandleSize;
                D3D12_UNORDERED_ACCESS_VIEW_DESC uavTable = {};
                uavTable.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                uavTable.Buffer.NumElements = m_cellCount * MAX_PARTICLES_PER_CELL;
                uavTable.Buffer.StructureByteStride = sizeof(UINT);
                device->CreateUnorderedAccessView(m_gridTable.Get(), nullptr, &uavTable, gridHandle);
        }

        // 描画用パイプライン生成
        // ルートシグネチャ
        CD3DX12_DESCRIPTOR_RANGE graRanges[3];
	graRanges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0); // t0
	graRanges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1); // t1
	graRanges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2); // t2

	CD3DX12_ROOT_PARAMETER graRootParam[4];
	graRootParam[0].InitAsDescriptorTable(1, &graRanges[0], D3D12_SHADER_VISIBILITY_PIXEL);
	graRootParam[1].InitAsDescriptorTable(1, &graRanges[1], D3D12_SHADER_VISIBILITY_PIXEL);
	graRootParam[2].InitAsDescriptorTable(1, &graRanges[2], D3D12_SHADER_VISIBILITY_PIXEL);
	graRootParam[3].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_PIXEL);

	CD3DX12_ROOT_SIGNATURE_DESC rsd(_countof(graRootParam), graRootParam, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
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


	// SRVヒープ
	D3D12_DESCRIPTOR_HEAP_DESC hd2 = {};
	hd2.NumDescriptors = 3;
	hd2.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	hd2.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        device->CreateDescriptorHeap(&hd2, IID_PPV_ARGS(&m_graphicsSrvHeap));

        // 描画用SRVを連続したハンドルで作成
        D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = m_graphicsSrvHeap->GetCPUDescriptorHandleForHeapStart();
        UINT srvHandleSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        // MetaBuffer用SRV(0番目)
        D3D12_SHADER_RESOURCE_VIEW_DESC srvd = {};
        srvd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvd.Buffer.NumElements = maxParticles;
        srvd.Buffer.StructureByteStride = sizeof(ParticleMeta);
        srvd.Format = DXGI_FORMAT_UNKNOWN;
        srvd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        device->CreateShaderResourceView(m_metaBuffer.Get(), &srvd, srvHandle);

        // GridCount用SRV(1番目)
        srvHandle.ptr += srvHandleSize;
        D3D12_SHADER_RESOURCE_VIEW_DESC srvd_count = {};
        srvd_count.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvd_count.Buffer.NumElements = m_cellCount;
        srvd_count.Buffer.StructureByteStride = sizeof(UINT);
        srvd_count.Format = DXGI_FORMAT_UNKNOWN;
        srvd_count.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        device->CreateShaderResourceView(m_gridCount.Get(), &srvd_count, srvHandle);

        // GridTable用SRV(2番目)
        srvHandle.ptr += srvHandleSize;
        D3D12_SHADER_RESOURCE_VIEW_DESC srvd_table = {};
        srvd_table.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvd_table.Buffer.NumElements = m_cellCount * MAX_PARTICLES_PER_CELL;
        srvd_table.Buffer.StructureByteStride = sizeof(UINT);
        srvd_table.Format = DXGI_FORMAT_UNKNOWN;
        srvd_table.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        device->CreateShaderResourceView(m_gridTable.Get(), &srvd_table, srvHandle);


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
        // グリッド用アップロードバッファ
        uploadDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(UINT) * m_cellCount);
        device->CreateCommittedResource(&upProps, D3D12_HEAP_FLAG_NONE, &uploadDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                IID_PPV_ARGS(&m_gridCountUpload));
        uploadDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(UINT) * m_cellCount * MAX_PARTICLES_PER_CELL);
        device->CreateCommittedResource(&upProps, D3D12_HEAP_FLAG_NONE, &uploadDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                IID_PPV_ARGS(&m_gridTableUpload));

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

                // ルートパラメーター: b0, b1, t0, u0, u1, u2, u3
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
		// 粒子バッファも読み取り用にSRV状態へ戻す
		auto particleToSrv = CD3DX12_RESOURCE_BARRIER::Transition(
			m_particleBuffer.Get(),
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
			D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		cmd->ResourceBarrier(1, &particleToSrv);
		m_particleInSrvState = true;
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

                // --------------------------------------------------
                // m_gridに格納した粒子IDをGPUバッファへコピー
                // --------------------------------------------------
                std::vector<UINT> gridCount(m_cellCount, 0);
                std::vector<UINT> gridTable(m_cellCount * MAX_PARTICLES_PER_CELL, 0);

                int ox = static_cast<int>(std::floor(m_gridMin.x / radius));
                int oy = static_cast<int>(std::floor(m_gridMin.y / radius));
                int oz = static_cast<int>(std::floor(m_gridMin.z / radius));

                for (const auto& cell : m_grid.Cells()) {
                        int gx = cell.first.x - ox;
                        int gy = cell.first.y - oy;
                        int gz = cell.first.z - oz;
                        if (gx < 0 || gy < 0 || gz < 0 ||
                                gx >= static_cast<int>(m_gridDimX) ||
                                gy >= static_cast<int>(m_gridDimY) ||
                                gz >= static_cast<int>(m_gridDimZ)) {
                                continue; // グリッド範囲外はスキップ
                        }
                        UINT cellId = gx + m_gridDimX * (gy + m_gridDimY * gz);
                        UINT count = static_cast<UINT>(std::min<size_t>(cell.second.size(), MAX_PARTICLES_PER_CELL));
                        gridCount[cellId] = count;
                        for (UINT i = 0; i < count; ++i) {
                                gridTable[cellId * MAX_PARTICLES_PER_CELL + i] = static_cast<UINT>(cell.second[i]);
                        }
                }

                auto toCopyGrid = {
                        CD3DX12_RESOURCE_BARRIER::Transition(
                                m_gridCount.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST),
                        CD3DX12_RESOURCE_BARRIER::Transition(
                                m_gridTable.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST)
                };
                cmd->ResourceBarrier(toCopyGrid.size(), toCopyGrid.begin());

                D3D12_SUBRESOURCE_DATA srcCount = {};
                srcCount.pData = gridCount.data();
                srcCount.RowPitch = sizeof(UINT) * m_cellCount;
                srcCount.SlicePitch = srcCount.RowPitch;
                UpdateSubresources<1>(cmd, m_gridCount.Get(), m_gridCountUpload.Get(), 0, 0, 1, &srcCount);

                D3D12_SUBRESOURCE_DATA srcTable = {};
                srcTable.pData = gridTable.data();
                srcTable.RowPitch = sizeof(UINT) * m_cellCount * MAX_PARTICLES_PER_CELL;
                srcTable.SlicePitch = srcTable.RowPitch;
                UpdateSubresources<1>(cmd, m_gridTable.Get(), m_gridTableUpload.Get(), 0, 0, 1, &srcTable);

                auto toSrvGrid = {
                        CD3DX12_RESOURCE_BARRIER::Transition(
                                m_gridCount.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
                        CD3DX12_RESOURCE_BARRIER::Transition(
                                m_gridTable.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
                };
                cmd->ResourceBarrier(toSrvGrid.size(), toSrvGrid.begin());
                m_gridInSrvState = true;
        }
}

// 描画
void FluidSystem::Render(ID3D12GraphicsCommandList* cmd, const DirectX::XMFLOAT4X4& invViewProj, const DirectX::XMFLOAT3& camPos, float isoLevel) {
        // CPUモードではSimulate内でSRVへの遷移を済ませている
        if (!m_gridInSrvState) {
                auto gridToSrv = {
                        CD3DX12_RESOURCE_BARRIER::Transition(
                                m_gridCount.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
                        CD3DX12_RESOURCE_BARRIER::Transition(
                                m_gridTable.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
                };
                cmd->ResourceBarrier(gridToSrv.size(), gridToSrv.begin());
                m_gridInSrvState = true;
        }

	// 定数バッファ更新
	struct MetaCB {
		DirectX::XMFLOAT4X4 invVP;
		DirectX::XMFLOAT3 cam;
		float iso;
		UINT count;
		DirectX::XMFLOAT3 gridMin;
		DirectX::XMUINT3 gridDim;
		float radius;
		float pad;
	} cb;
	cb.invVP = invViewProj;
	cb.cam = camPos;
	cb.iso = isoLevel;
	cb.count = m_maxParticles;
	cb.gridMin = m_gridMin;
	cb.gridDim = { m_gridDimX, m_gridDimY, m_gridDimZ };
	cb.radius = m_params.radius;

	void* p;
	m_graphicsCB->Map(0, nullptr, &p);
	memcpy(p, &cb, sizeof(cb));
	m_graphicsCB->Unmap(0, nullptr);

	// スクリーンスペース処理を行わず直接描画する
	cmd->SetDescriptorHeaps(1, m_graphicsSrvHeap.GetAddressOf());
	cmd->SetGraphicsRootSignature(m_graphicsRS.Get());
	cmd->SetPipelineState(m_graphicsPS->Get());

	// 3つのディスクリプタテーブルをセット
	UINT handleSize = g_Engine->Device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	auto gpuHandle = m_graphicsSrvHeap->GetGPUDescriptorHandleForHeapStart();
	cmd->SetGraphicsRootDescriptorTable(0, gpuHandle); // t0: Meta
	gpuHandle.ptr += handleSize;
	cmd->SetGraphicsRootDescriptorTable(1, gpuHandle); // t1: GridCount
	gpuHandle.ptr += handleSize;
	cmd->SetGraphicsRootDescriptorTable(2, gpuHandle); // t2: GridTable

	cmd->SetGraphicsRootConstantBufferView(3, m_graphicsCB->GetGPUVirtualAddress()); // CBVはindex3に
	cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	cmd->DrawInstanced(3, 1, 0, 0);
	
	// 次フレームのシミュレーションのためにSRVからUAVステートへ戻す
        auto gridToUav = {
                CD3DX12_RESOURCE_BARRIER::Transition(
                        m_gridCount.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                CD3DX12_RESOURCE_BARRIER::Transition(
                        m_gridTable.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        };
        cmd->ResourceBarrier(gridToUav.size(), gridToUav.begin());
        m_gridInSrvState = false;
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


