#include "FluidSystem.h"
#include "d3dx12.h"
#include <algorithm>
#include <d3dcompiler.h>
#include <vector>
#include "Engine.h"
#include "Camera.h"

struct SPHParamsCB {
    float restDensity;
    float particleMass;
    float viscosity;
    float stiffness;
    float radius;
    float timeStep;
    uint32_t particleCount;
    float pad; // padding to 16 bytes multiple
};

struct ViewProjCB {
    DirectX::XMFLOAT4X4 viewProj;
};

void FluidSystem::Init(ID3D12Device* device, DXGI_FORMAT rtvFormat,
        UINT maxParticles, UINT threadGroupCount) {
        m_maxParticles = maxParticles;
        m_threadGroupCount = threadGroupCount;
        m_cpuParticles.resize(maxParticles);

        // ---------------------------------------------------------------------
        // Compute root signature
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

        CD3DX12_ROOT_PARAMETER params[5];
        params[0].InitAsConstantBufferView(0);      // b0
        params[1].InitAsConstantBufferView(1);      // b1
        params[2].InitAsDescriptorTable(1, &srvRange, D3D12_SHADER_VISIBILITY_ALL);
        params[3].InitAsDescriptorTable(1, &uavRange0, D3D12_SHADER_VISIBILITY_ALL);
        params[4].InitAsDescriptorTable(1, &uavRange1, D3D12_SHADER_VISIBILITY_ALL);

        CD3DX12_ROOT_SIGNATURE_DESC rsDesc(_countof(params), params, 0, nullptr,
                D3D12_ROOT_SIGNATURE_FLAG_NONE);

        ComPtr<ID3DBlob> blob, error;
        HRESULT hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1,
                &blob, &error);
        if (FAILED(hr)) {
                if (error) {
                        wprintf(L"[Error] RootSignature Serialize failed: %s\n", (char*)error->GetBufferPointer());
                }
                return;
        }

        hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                IID_PPV_ARGS(&m_computeRS));
        if (FAILED(hr)) {
                wprintf(L"[Error] RootSignature 生成失敗: 0x%08X\n", hr);
                return;
        }

        // ---------------------------------------------------------------------
        // Compute PSO
        m_computePS.SetDevice(device);
        m_computePS.SetRootSignature(m_computeRS.Get());
        m_computePS.SetCS(L"ParticleCS.cso");
        m_computePS.Create();

        // ---------------------------------------------------------------------
        // Particle and meta buffers
        D3D12_RESOURCE_DESC rdPart = CD3DX12_RESOURCE_DESC::Buffer(
                sizeof(Particle) * maxParticles,
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
        hd.NumDescriptors = 3; // SRV + UAV + UAV
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_uavHeap));

        UINT handleSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE handle = m_uavHeap->GetCPUDescriptorHandleForHeapStart();

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.NumElements = maxParticles;
        srvDesc.Buffer.StructureByteStride = sizeof(Particle);
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        device->CreateShaderResourceView(m_particleBuffer.Get(), &srvDesc, handle);

        handle.ptr += handleSize;
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavd = {};
        uavd.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavd.Buffer.NumElements = maxParticles;
        uavd.Buffer.StructureByteStride = sizeof(Particle);
        device->CreateUnorderedAccessView(m_particleBuffer.Get(), nullptr, &uavd, handle);

        handle.ptr += handleSize;
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavMeta = {};
        uavMeta.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavMeta.Buffer.NumElements = maxParticles;
        uavMeta.Buffer.StructureByteStride = sizeof(ParticleMeta);
        device->CreateUnorderedAccessView(m_metaBuffer.Get(), nullptr, &uavMeta, handle);

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
        device->CreateShaderResourceView(m_metaBuffer.Get(), &srvd,
                m_graphicsSrvHeap->GetCPUDescriptorHandleForHeapStart());

        // 定数バッファ
        D3D12_HEAP_PROPERTIES hp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        D3D12_RESOURCE_DESC   cbd = CD3DX12_RESOURCE_DESC::Buffer(sizeof(DirectX::XMFLOAT4X4) +
                sizeof(DirectX::XMFLOAT3) + sizeof(UINT) + 4);
        device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &cbd,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                IID_PPV_ARGS(&m_graphicsCB));

        // アップロードヒープ (CPU→GPU 転送用)
        CD3DX12_HEAP_PROPERTIES upProps(D3D12_HEAP_TYPE_UPLOAD);
        D3D12_RESOURCE_DESC uploadDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(Particle) * maxParticles);
        device->CreateCommittedResource(&upProps, D3D12_HEAP_FLAG_NONE, &uploadDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                IID_PPV_ARGS(&m_particleUpload));
        uploadDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(ParticleMeta) * maxParticles);
        device->CreateCommittedResource(&upProps, D3D12_HEAP_FLAG_NONE, &uploadDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                IID_PPV_ARGS(&m_metaUpload));

        // Compute用定数バッファ作成
        m_sphParamCB = new ConstantBuffer(sizeof(SPHParamsCB));
        m_viewProjCB = new ConstantBuffer(sizeof(ViewProjCB));

        // 初期値設定
        if (m_sphParamCB && m_sphParamCB->IsValid()) {
            auto* cb = m_sphParamCB->GetPtr<SPHParamsCB>();
            cb->restDensity   = 1000.0f;
            cb->particleMass  = 1.0f;
            cb->viscosity     = 1.0f;
            cb->stiffness     = 200.0f;
            cb->radius        = 0.1f;
            cb->timeStep      = 0.016f;
            cb->particleCount = m_maxParticles;
        }

        if (m_viewProjCB && m_viewProjCB->IsValid()) {
            auto* cb = m_viewProjCB->GetPtr<ViewProjCB>();
            DirectX::XMStoreFloat4x4(&cb->viewProj, DirectX::XMMatrixIdentity());
        }
}

void FluidSystem::Simulate(ID3D12GraphicsCommandList* cmd, float dt) {
        if (m_useGpu) {
                // GPU シミュレーション
                CD3DX12_RESOURCE_BARRIER barriers[] = {
                    CD3DX12_RESOURCE_BARRIER::UAV(m_particleBuffer.Get()),
                    CD3DX12_RESOURCE_BARRIER::UAV(m_metaBuffer.Get())
                };
                cmd->ResourceBarrier(2, barriers);
                cmd->SetComputeRootSignature(m_computeRS.Get());
                ID3D12DescriptorHeap* heaps[] = { m_uavHeap.Get() };
                cmd->SetDescriptorHeaps(1, heaps);

                UINT handleSize = g_Engine->Device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = m_uavHeap->GetGPUDescriptorHandleForHeapStart();
                D3D12_GPU_DESCRIPTOR_HANDLE uavParticle = srvHandle;
                uavParticle.ptr += handleSize;
                D3D12_GPU_DESCRIPTOR_HANDLE uavMeta = uavParticle;
                uavMeta.ptr += handleSize;

                // Root parameter order: b0, b1, t0, u0, u1
                // 定数バッファ更新
                if (m_sphParamCB && m_sphParamCB->IsValid()) {
                    auto* cb = m_sphParamCB->GetPtr<SPHParamsCB>();
                    cb->timeStep      = dt;
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
                cmd->SetPipelineState(m_computePS.Get());
                cmd->Dispatch(m_threadGroupCount, 1, 1);
        }
	else {
                // CPU シミュレーション
                for (UINT i = 0; i < m_maxParticles; ++i) {
                        m_cpuParticles[i].position.y += dt;
                }

                // CPU→GPU 転送 (particles)
                D3D12_SUBRESOURCE_DATA srcParticle = {};
                srcParticle.pData = m_cpuParticles.data();
                srcParticle.RowPitch = sizeof(Particle) * m_maxParticles;
                srcParticle.SlicePitch = srcParticle.RowPitch;
                UpdateSubresources<1>(cmd, m_particleBuffer.Get(), m_particleUpload.Get(), 0, 0, 1, &srcParticle);

                std::vector<ParticleMeta> metas(m_maxParticles);
                for (UINT i = 0; i < m_maxParticles; ++i) {
                        metas[i].pos = m_cpuParticles[i].position;
                        metas[i].r   = 0.1f;
                }
                D3D12_SUBRESOURCE_DATA srcMeta = {};
                srcMeta.pData = metas.data();
                srcMeta.RowPitch = sizeof(ParticleMeta) * m_maxParticles;
                srcMeta.SlicePitch = srcMeta.RowPitch;
                UpdateSubresources<1>(cmd, m_metaBuffer.Get(), m_metaUpload.Get(), 0, 0, 1, &srcMeta);
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
