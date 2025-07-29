#include "FluidSystem.h"
#include "d3dx12.h"
#include <algorithm>

void FluidSystem::Init(ID3D12Device* device, DXGI_FORMAT rtvFormat,
	UINT maxParticles, UINT threadGroupCount) {
	m_maxParticles = maxParticles;
	m_threadGroupCount = threadGroupCount;
	m_cpuParticles.resize(maxParticles);

	CD3DX12_DESCRIPTOR_RANGE range;
	range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);

	CD3DX12_ROOT_PARAMETER rp;
	rp.InitAsDescriptorTable(1, &range, D3D12_SHADER_VISIBILITY_ALL);

	CD3DX12_ROOT_SIGNATURE_DESC rsDesc(1, &rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

	// シリアライズ
	ComPtr<ID3DBlob> blob, errBlob;
	HRESULT hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errBlob);
	if (FAILED(hr)) {
		if (errBlob) {
			OutputDebugStringA((char*)errBlob->GetBufferPointer());
		}
		wprintf(L"RootSignature シリアライズ失敗: 0x%08X\n", hr);
		return;
	}

	// ルートシグネチャ生成
	hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&m_computeRS));
	if (FAILED(hr)) {
		wprintf(L"[Error] RootSignature 生成失敗: 0x%08X\n", hr);
		return;
	}

	// ComputePSOを初期化
	m_computePS.SetDevice(device);
	m_computePS.SetRootSignature(m_computeRS.Get());
	m_computePS.SetCS(L"ParticleCS.cso");
	m_computePS.Create();

	// 粒子バッファとUAVヒープ
	D3D12_RESOURCE_DESC rd = CD3DX12_RESOURCE_DESC::Buffer(sizeof(ParticleMeta) * maxParticles, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
	CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_DEFAULT);
	device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_particleBuffer));
	D3D12_DESCRIPTOR_HEAP_DESC hd = {};
	hd.NumDescriptors = 1;
	hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_uavHeap));

	D3D12_UNORDERED_ACCESS_VIEW_DESC uavd = {};
	uavd.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavd.Buffer.NumElements = maxParticles;
	uavd.Buffer.StructureByteStride = sizeof(ParticleMeta);
	device->CreateUnorderedAccessView(m_particleBuffer.Get(), nullptr, &uavd, m_uavHeap->GetCPUDescriptorHandleForHeapStart());

	// 描画用パイプライン生成
	// ルートシグネチャ
	CD3DX12_DESCRIPTOR_RANGE graRange;
	range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
	CD3DX12_ROOT_PARAMETER graRootParam[2];
	graRootParam[0].InitAsDescriptorTable(1, &range, D3D12_SHADER_VISIBILITY_PIXEL);
	graRootParam[1].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_PIXEL);
	CD3DX12_ROOT_SIGNATURE_DESC rsd(2, graRootParam, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
	ComPtr<ID3DBlob> graBlob, graErr;
	D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &graBlob, &graErr);
	device->CreateRootSignature(0, graBlob->GetBufferPointer(), graBlob->GetBufferSize(), IID_PPV_ARGS(&m_graphicsRS));

	// PipelineState
	m_graphicsPS = new PipelineState();
	m_graphicsPS->SetRootSignature(m_graphicsRS.Get());
	D3D12_INPUT_LAYOUT_DESC nullIL{ nullptr, 0 };
	m_graphicsPS->SetInputLayout(nullIL);
	m_graphicsPS->SetVS(L"MetaBallVS.cso");
	m_graphicsPS->SetPS(L"MetaBallPS.cso");
	m_graphicsPS->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);

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
	device->CreateShaderResourceView(m_particleBuffer.Get(), &srvd,
		m_graphicsSrvHeap->GetCPUDescriptorHandleForHeapStart());

	// 定数バッファ
	D3D12_HEAP_PROPERTIES hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
	D3D12_RESOURCE_DESC   cbd = CD3DX12_RESOURCE_DESC::Buffer(sizeof(DirectX::XMFLOAT4X4) +
		sizeof(DirectX::XMFLOAT3) + sizeof(UINT) + 4);
	device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &cbd,
		D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
		IID_PPV_ARGS(&m_graphicsCB));

	// アップロードヒープを作成
	CD3DX12_HEAP_PROPERTIES uploadProps(D3D12_HEAP_TYPE_UPLOAD);
	CD3DX12_RESOURCE_DESC   uploadDesc =
		CD3DX12_RESOURCE_DESC::Buffer(sizeof(ParticleMeta) * m_maxParticles);
	device->CreateCommittedResource(
		&uploadProps,
		D3D12_HEAP_FLAG_NONE,
		&uploadDesc,
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&m_uploadHeap)
	);
}

void FluidSystem::Simulate(ID3D12GraphicsCommandList* cmd, float dt) {
	if (m_useGpu) {
		// GPU シミュレーション
		auto uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_particleBuffer.Get());
		cmd->ResourceBarrier(1, &uavBarrier);
		cmd->SetComputeRootSignature(m_computeRS.Get());
		ID3D12DescriptorHeap* heaps[] = { m_uavHeap.Get() };
		cmd->SetDescriptorHeaps(1, heaps);
		cmd->SetComputeRootDescriptorTable(0, m_uavHeap->GetGPUDescriptorHandleForHeapStart());
		cmd->SetPipelineState(m_computePS.Get());
		cmd->SetComputeRoot32BitConstants(1, 1, &dt, 0);
		cmd->Dispatch(m_threadGroupCount, 1, 1);
	}
	else {
		// CPU シミュレーション
		for (UINT i = 0; i < m_maxParticles; ++i) {
			// m_cpuParticles[i].pos を更新 (簡易例：Y+=dt)
			m_cpuParticles[i].pos.y += dt;
		}
		// CPU→GPU 転送
		D3D12_SUBRESOURCE_DATA srcData = {};
		srcData.pData = m_cpuParticles.data();
		srcData.RowPitch = sizeof(ParticleMeta) * m_maxParticles;
		srcData.SlicePitch = srcData.RowPitch;
		UpdateSubresources<1>(cmd, m_particleBuffer.Get(), m_uploadHeap.Get(), 0, 0, 1, &srcData);
	}
}

void FluidSystem::Render(ID3D12GraphicsCommandList* cmd, const DirectX::XMFLOAT4X4& invViewProj, const DirectX::XMFLOAT3& camPos, float isoLevel) {
	// 定数バッファ更新
	struct MetaCB { DirectX::XMFLOAT4X4 invVP; DirectX::XMFLOAT3 cam; float iso; UINT count; } cb;
	cb.invVP = invViewProj;
	cb.cam = camPos;
	cb.iso = isoLevel;
	cb.count = m_maxParticles;
	void* p;
	m_graphicsCB->Map(0, nullptr, &p);
	memcpy(p, &cb, sizeof(cb));
	m_graphicsCB->Unmap(0, nullptr);

	// 描画
	cmd->SetDescriptorHeaps(1, m_graphicsSrvHeap.GetAddressOf());
	cmd->SetGraphicsRootSignature(m_graphicsRS.Get());
	cmd->SetPipelineState(m_graphicsPS->Get());
	cmd->SetGraphicsRootDescriptorTable(0, m_graphicsSrvHeap->GetGPUDescriptorHandleForHeapStart());
	cmd->SetGraphicsRootConstantBufferView(1, m_graphicsCB->GetGPUVirtualAddress());
	cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	cmd->DrawInstanced(3, 1, 0, 0);
}
