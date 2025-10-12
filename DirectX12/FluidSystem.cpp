#include "FluidSystem.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <d3dx12.h>
#include "Engine.h"

using namespace DirectX;

namespace {
        constexpr XMFLOAT4 kSurfaceColor{ 0.0f, 0.45f, 0.8f, 0.9f };
        constexpr float kCameraNear = 0.1f;
        constexpr float kCameraFar = 1000.0f;

        struct ParticleDepthConstant
        {
                XMFLOAT4X4 view;
                XMFLOAT4X4 proj;
                XMFLOAT2 clipZ;
                XMFLOAT2 pad;
        };

        struct ParticleCompositeConstant
        {
                XMFLOAT4 misc0; // x=1/width, y=1/height, z=半径, w=法線強調
                XMFLOAT4 misc1; // x,y,z=流体色, w=不透明度基準
                XMFLOAT4 misc2; // x,y,z=ライト方向, w=ハイライト硬さ
        };
}

FluidSystem::FluidSystem() = default;
FluidSystem::~FluidSystem() = default;

bool FluidSystem::Init(ID3D12Device* device, const Bounds& bounds, size_t particleCount)
{
	m_device = device;
	m_bounds = bounds;
	m_waterLevel = (bounds.min.y + bounds.max.y) * 0.5f;
	m_timeSeconds = 0.0f;

	// --- モード判定 ---
	m_mode = (particleCount > 0) ? SimMode::Particles : SimMode::Heightfield;
	m_particleCap = (int)particleCount;

	// --- グリッド共通の初期化（XZセル幅の基準）
	const float gridWidth = m_bounds.max.x - m_bounds.min.x;
	const float gridDepth = m_bounds.max.z - m_bounds.min.z;
	m_simMin = XMFLOAT3(m_bounds.min.x, m_bounds.min.y, m_bounds.min.z);
	m_cellDx = (m_resolution > 1) ? gridWidth / float(m_resolution - 1) : 0.0f;
	m_cellDz = (m_resolution > 1) ? gridDepth / float(m_resolution - 1) : 0.0f;

        if (m_mode == SimMode::Particles) {
                InitParticles(std::max(100, m_particleCap)); // 粒子スポーン
                if (!BuildParticleRenderResources()) return false; // SSFR描画用リソース生成
                return true;
        }

	// ---- 高さ場モード----
	if (!BuildSimulationResources()) return false;
	if (!BuildRenderResources())     return false;

	ResetWaveState();
	UpdateVertexBuffer();
	return true;
}

// 計算に必要なリソースを作成
bool FluidSystem::BuildSimulationResources()
{
	const size_t total = (size_t)m_resolution * (size_t)m_resolution;
	m_height.assign(total, 0.0f);
	m_velocity.assign(total, 0.0f);
	m_vertices.assign(total, {});

	m_indices.clear();
	m_indices.reserve((m_resolution - 1) * (m_resolution - 1) * 6);
	for (int z = 0; z < m_resolution - 1; ++z)
		for (int x = 0; x < m_resolution - 1; ++x)
		{
			uint32_t i0 = (uint32_t)Index(x, z);
			uint32_t i1 = (uint32_t)Index(x + 1, z);
			uint32_t i2 = (uint32_t)Index(x, z + 1);
			uint32_t i3 = (uint32_t)Index(x + 1, z + 1);
			m_indices.push_back(i0); m_indices.push_back(i1); m_indices.push_back(i2);
			m_indices.push_back(i1); m_indices.push_back(i3); m_indices.push_back(i2);
		}
	return true;
}

bool FluidSystem::BuildParticleRenderResources()
{
        // 既存リソースを初期化してから再構築する
        m_particleQuadVB.reset();
        m_particleQuadIB.reset();
        m_particleInstanceVB.reset();
        m_particleDepthRoot.reset();
        m_particleDepthPSO.reset();
        for (auto& cb : m_particleDepthCB) cb.reset();
        m_particleCompositeRoot.reset();
        m_particleCompositePSO.reset();
        for (auto& cb : m_particleCompositeCB) cb.reset();
        m_particleRtvHeap.Reset();
        m_particleDepthTexture.Reset();
        m_particleDepthSrv = nullptr;
        m_particleDepthState = D3D12_RESOURCE_STATE_COMMON;

        if (!m_device || m_particleCap <= 0) {
                return false;
        }

        // 画面に張り付けるビルボード頂点（4隅）
        struct QuadVertex { XMFLOAT2 corner; };
        const QuadVertex quad[4] = {
                { XMFLOAT2(-1.0f, -1.0f) },
                { XMFLOAT2( 1.0f, -1.0f) },
                { XMFLOAT2( 1.0f,  1.0f) },
                { XMFLOAT2(-1.0f,  1.0f) },
        };
        m_particleQuadVB = std::make_unique<VertexBuffer>(sizeof(quad), sizeof(QuadVertex), quad);
        if (!m_particleQuadVB || !m_particleQuadVB->IsValid()) {
                return false;
        }

        const uint32_t quadIndices[6] = { 0,1,2, 0,2,3 };
        m_particleQuadIB = std::make_unique<IndexBuffer>(sizeof(quadIndices), quadIndices);
        if (!m_particleQuadIB || !m_particleQuadIB->IsValid()) {
                return false;
        }

        // 粒子インスタンス用頂点バッファ（毎フレーム更新）
        m_particleInstanceVB = std::make_unique<VertexBuffer>(sizeof(ParticleInstanceGPU) * (size_t)m_particleCap, sizeof(ParticleInstanceGPU), nullptr);
        if (!m_particleInstanceVB || !m_particleInstanceVB->IsValid()) {
                return false;
        }

        // 深度パス用ルートシグネチャ
        CD3DX12_ROOT_PARAMETER depthParams[1];
        depthParams[0].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_ALL);

        D3D12_ROOT_SIGNATURE_DESC depthDesc{};
        depthDesc.NumParameters = _countof(depthParams);
        depthDesc.pParameters = depthParams;
        depthDesc.NumStaticSamplers = 0;
        depthDesc.pStaticSamplers = nullptr;
        depthDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        m_particleDepthRoot = std::make_unique<RootSignature>();
        if (!m_particleDepthRoot || !m_particleDepthRoot->Init(depthDesc) || !m_particleDepthRoot->IsValid()) {
                return false;
        }

        // 深度パイプラインステート
        m_particleDepthPSO = std::make_unique<PipelineState>();
        if (!m_particleDepthPSO) {
                return false;
        }

        D3D12_INPUT_ELEMENT_DESC depthLayout[] = {
                { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,   0 },
                { "POSITION", 1, DXGI_FORMAT_R32G32B32_FLOAT, 1, 0, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
                { "TEXCOORD", 0, DXGI_FORMAT_R32_FLOAT,       1, sizeof(XMFLOAT3), D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
        };
        m_particleDepthPSO->SetInputLayout({ depthLayout, _countof(depthLayout) });
        m_particleDepthPSO->SetRootSignature(m_particleDepthRoot->Get());
        m_particleDepthPSO->SetVS(L"ParticlesDepthVS.cso");
        m_particleDepthPSO->SetPS(L"ParticlesDepthPS.cso");
        m_particleDepthPSO->SetDepthStencilFormat(DXGI_FORMAT_UNKNOWN); // 深度バッファ非使用
        m_particleDepthPSO->SetRenderTargetFormat(DXGI_FORMAT_R32_FLOAT);
        m_particleDepthPSO->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
        if (!m_particleDepthPSO->IsValid()) {
                return false;
        }

        for (auto& cb : m_particleDepthCB) {
                cb = std::make_unique<ConstantBuffer>(sizeof(ParticleDepthConstant));
                if (!cb || !cb->IsValid()) {
                        return false;
                }
        }

        // 深度結果を書き込むレンダーターゲット
        const UINT width = std::max(1u, g_Engine->FrameBufferWidth());
        const UINT height = std::max(1u, g_Engine->FrameBufferHeight());

        CD3DX12_RESOURCE_DESC depthTexDesc = CD3DX12_RESOURCE_DESC::Tex2D(
                DXGI_FORMAT_R32_FLOAT, width, height, 1, 1, 1, 0,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);

        D3D12_CLEAR_VALUE clearValue{};
        clearValue.Format = DXGI_FORMAT_R32_FLOAT;
        clearValue.Color[0] = 0.0f; // 深度テクスチャ初期値

        auto heapProp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        HRESULT hr = m_device->CreateCommittedResource(
                &heapProp,
                D3D12_HEAP_FLAG_NONE,
                &depthTexDesc,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                &clearValue,
                IID_PPV_ARGS(m_particleDepthTexture.ReleaseAndGetAddressOf()));
        if (FAILED(hr)) {
                return false;
        }
        m_particleDepthState = D3D12_RESOURCE_STATE_RENDER_TARGET;

        // RTVヒープを粒子専用で確保
        D3D12_DESCRIPTOR_HEAP_DESC rtvDesc{};
        rtvDesc.NumDescriptors = 1;
        rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        rtvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        hr = m_device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(m_particleRtvHeap.ReleaseAndGetAddressOf()));
        if (FAILED(hr)) {
                return false;
        }
        m_particleDepthRtv = m_particleRtvHeap->GetCPUDescriptorHandleForHeapStart();
        m_device->CreateRenderTargetView(m_particleDepthTexture.Get(), nullptr, m_particleDepthRtv);

        // SRV登録
        if (!g_Engine->CbvSrvUavHeap()) {
                return false;
        }
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
        srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Texture2D.MipLevels = 1;
        m_particleDepthSrv = g_Engine->CbvSrvUavHeap()->Register(m_particleDepthTexture.Get(), srvDesc);
        if (!m_particleDepthSrv) {
                return false;
        }

        // 合成パス用ルートシグネチャ
        CD3DX12_DESCRIPTOR_RANGE range{};
        range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);

        CD3DX12_ROOT_PARAMETER compParams[2];
        compParams[0].InitAsDescriptorTable(1, &range, D3D12_SHADER_VISIBILITY_PIXEL);
        compParams[1].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_PIXEL);

        CD3DX12_STATIC_SAMPLER_DESC sampler(0, D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,
                D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);

        D3D12_ROOT_SIGNATURE_DESC compDesc{};
        compDesc.NumParameters = _countof(compParams);
        compDesc.pParameters = compParams;
        compDesc.NumStaticSamplers = 1;
        compDesc.pStaticSamplers = &sampler;
        compDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        m_particleCompositeRoot = std::make_unique<RootSignature>();
        if (!m_particleCompositeRoot || !m_particleCompositeRoot->Init(compDesc) || !m_particleCompositeRoot->IsValid()) {
                return false;
        }

        m_particleCompositePSO = std::make_unique<PipelineState>();
        if (!m_particleCompositePSO) {
                return false;
        }
        m_particleCompositePSO->SetInputLayout({ nullptr, 0 });
        m_particleCompositePSO->SetRootSignature(m_particleCompositeRoot->Get());
        m_particleCompositePSO->SetVS(L"ParticleCompositeVS.cso");
        m_particleCompositePSO->SetPS(L"ParticleCompositePS.cso");
        m_particleCompositePSO->SetDepthStencilFormat(DXGI_FORMAT_D32_FLOAT);

        // 半透明合成のためアルファブレンドを有効化
        D3D12_BLEND_DESC blendDesc{};
        blendDesc.AlphaToCoverageEnable = FALSE;
        blendDesc.IndependentBlendEnable = FALSE;
        auto& rtBlend = blendDesc.RenderTarget[0];
        rtBlend.BlendEnable = TRUE;
        rtBlend.SrcBlend = D3D12_BLEND_SRC_ALPHA;
        rtBlend.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        rtBlend.BlendOp = D3D12_BLEND_OP_ADD;
        rtBlend.SrcBlendAlpha = D3D12_BLEND_ONE;
        rtBlend.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
        rtBlend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        rtBlend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        m_particleCompositePSO->SetBlendState(blendDesc);

        m_particleCompositePSO->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
        if (!m_particleCompositePSO->IsValid()) {
                return false;
        }

        for (auto& cb : m_particleCompositeCB) {
                cb = std::make_unique<ConstantBuffer>(sizeof(ParticleCompositeConstant));
                if (!cb || !cb->IsValid()) {
                        return false;
                }
        }

        return true;
}

// 描画に必要なリソースを作成
bool FluidSystem::BuildRenderResources()
{
	CD3DX12_ROOT_PARAMETER params[1] = {};
	params[0].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_ALL);

	D3D12_ROOT_SIGNATURE_DESC desc{};
	desc.NumParameters = _countof(params);
	desc.pParameters = params;
	desc.NumStaticSamplers = 0;
	desc.pStaticSamplers = nullptr;
	desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

	m_rootSignature = std::make_unique<RootSignature>();
	if (!m_rootSignature || !m_rootSignature->Init(desc) || !m_rootSignature->IsValid()) return false;

	m_pipelineState = std::make_unique<PipelineState>();
	if (!m_pipelineState) return false;

	D3D12_INPUT_ELEMENT_DESC layout[] = {
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, offsetof(OceanVertex, position), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT,    0, offsetof(OceanVertex, normal),   D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,       0, offsetof(OceanVertex, uv),       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, offsetof(OceanVertex, color),    D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
	};
	m_pipelineState->SetInputLayout({ layout, _countof(layout) });
	m_pipelineState->SetRootSignature(m_rootSignature->Get());
	m_pipelineState->SetVS(L"OceanVS.cso");
	m_pipelineState->SetPS(L"OceanPS.cso");
	m_pipelineState->SetDepthStencilFormat(DXGI_FORMAT_D32_FLOAT);
	m_pipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
	if (!m_pipelineState->IsValid()) return false;

	size_t vbSize = m_vertices.size() * sizeof(OceanVertex);
	m_vertexBuffer = std::make_unique<VertexBuffer>(vbSize, sizeof(OceanVertex), m_vertices.data());
	if (!m_vertexBuffer || !m_vertexBuffer->IsValid()) return false;

	size_t ibSize = m_indices.size() * sizeof(uint32_t);
	m_indexBuffer = std::make_unique<IndexBuffer>(ibSize, m_indices.data());
	if (!m_indexBuffer || !m_indexBuffer->IsValid()) return false;

	for (auto& cb : m_constantBuffers) {
		cb = std::make_unique<ConstantBuffer>(sizeof(OceanConstant));
		if (!cb || !cb->IsValid()) return false;
	}
	return true;
}

// 初期状態にリセット
void FluidSystem::ResetWaveState()
{
	std::fill(m_height.begin(), m_height.end(), 0.0f);
	std::fill(m_velocity.begin(), m_velocity.end(), 0.0f);
}

// 
void FluidSystem::ApplyPendingDrops()
{
	for (const auto& d : m_pendingDrops) ApplyDrop(d);
	m_pendingDrops.clear();
}

// 
void FluidSystem::ApplyDrop(const DropRequest& drop)
{
	float fx = std::clamp(drop.uv.x * float(m_resolution - 1), 0.0f, float(m_resolution - 1));
	float fz = std::clamp(drop.uv.y * float(m_resolution - 1), 0.0f, float(m_resolution - 1));
	int cx = (int)std::round(fx), cz = (int)std::round(fz);

	const float r = drop.radius * float(m_resolution);
	const float r2 = r * r;

	for (int z = std::max(0, cz - (int)r); z <= std::min(m_resolution - 1, cz + (int)r); ++z)
		for (int x = std::max(0, cx - (int)r); x <= std::min(m_resolution - 1, cx + (int)r); ++x)
		{
			float dx = float(x) - fx, dz = float(z) - fz;
			float d2 = dx * dx + dz * dz;
			if (d2 > r2) continue;
			float falloff = 1.0f - (d2 / r2);
			m_velocity[Index((size_t)x, (size_t)z)] += drop.strength * falloff;
		}
}

// 水の計算
void FluidSystem::StepSimulation(float deltaTime)
{
	const float gridWidth = m_bounds.max.x - m_bounds.min.x;
	const float gridDepth = m_bounds.max.z - m_bounds.min.z;
	if (gridWidth <= 0.0f || gridDepth <= 0.0f) return;

	const float dtMax = std::min(deltaTime, 0.033f);
	const float dx = gridWidth / float(m_resolution - 1);
	const float dz = gridDepth / float(m_resolution - 1);
	const float hcell = std::max(1e-5f, std::min(dx, dz));
	const float c = std::max(1e-4f, m_waveSpeed);
	const float dtCFL = 0.5f * (hcell / c);
	int   steps = std::max(1, (int)std::ceil(dtMax / dtCFL));
	steps = std::min(steps, 8);
	const float sdt = dtMax / float(steps);
	const float dampPerStep = std::pow(m_damping, sdt / (1.0f / 60.0f));
	const float coeff = (m_waveSpeed * m_waveSpeed);

	for (int s = 0; s < steps; ++s)
	{
		std::vector<float> newHeight(m_height.size(), 0.0f);
		for (int z = 0; z < m_resolution; ++z)
			for (int x = 0; x < m_resolution; ++x)
			{
				size_t idx = Index((size_t)x, (size_t)z);
				float h = m_height[idx];

				auto sample = [&](int sx, int sz) {
					sx = std::clamp(sx, 0, m_resolution - 1);
					sz = std::clamp(sz, 0, m_resolution - 1);
					return m_height[Index((size_t)sx, (size_t)sz)];
					};

				float lap = sample(x - 1, z) + sample(x + 1, z) + sample(x, z - 1) + sample(x, z + 1) - 4.0f * h;
				float accel = coeff * lap;
				m_velocity[idx] += accel * sdt;
				m_velocity[idx] *= dampPerStep;
				newHeight[idx] = h + m_velocity[idx] * sdt;
			}
		m_height.swap(newHeight);
	}
}

void FluidSystem::UpdateVertexBuffer()
{
	if (m_cellDx <= 0.0f || m_cellDz <= 0.0f) return;

	for (int z = 0; z < m_resolution; ++z)
		for (int x = 0; x < m_resolution; ++x)
		{
			const size_t idx = Index((size_t)x, (size_t)z);
			const float wx = m_simMin.x + m_cellDx * float(x);
			const float wz = m_simMin.z + m_cellDz * float(z);
			const float wy = m_waterLevel + m_height[idx];

			auto& v = m_vertices[idx];
			v.position = XMFLOAT3(wx, wy, wz);
			const float fx = float(x) / float(m_resolution - 1);
			const float fz = float(z) / float(m_resolution - 1);
			v.uv = XMFLOAT2(fx, fz);
			v.color = kSurfaceColor;
		}

	const float inv2dx = 1.0f / (2.0f * m_cellDx);
	const float inv2dz = 1.0f / (2.0f * m_cellDz);
	for (int z = 0; z < m_resolution; ++z)
		for (int x = 0; x < m_resolution; ++x)
		{
			const int xm = std::max(0, x - 1), xp = std::min(m_resolution - 1, x + 1);
			const int zm = std::max(0, z - 1), zp = std::min(m_resolution - 1, z + 1);
			const float hx = m_height[Index((size_t)xp, (size_t)z)] - m_height[Index((size_t)xm, (size_t)z)];
			const float hz = m_height[Index((size_t)x, (size_t)zp)] - m_height[Index((size_t)x, (size_t)zm)];
			const float dhdx = hx * inv2dx, dhdz = hz * inv2dz;
			XMFLOAT3 nLocal(-dhdx, 1.0f, -dhdz);
			XMVECTOR n = XMVector3Normalize(XMLoadFloat3(&nLocal));
			XMStoreFloat3(&m_vertices[Index((size_t)x, (size_t)z)].normal, n);
		}

	if (m_vertexBuffer && m_vertexBuffer->IsValid()) {
		void* mapped = nullptr;
		auto resource = m_vertexBuffer->GetResource();
		if (resource && SUCCEEDED(resource->Map(0, nullptr, &mapped))) {
			std::memcpy(mapped, m_vertices.data(), m_vertices.size() * sizeof(OceanVertex));
			resource->Unmap(0, nullptr);
		}
	}
}

void FluidSystem::UpdateCameraCB(const Camera& camera)
{
	UINT frameIndex = g_Engine->CurrentBackBufferIndex();
	auto& cb = m_constantBuffers[frameIndex];
	if (!cb) return;
	auto* constant = cb->GetPtr<OceanConstant>();
	if (!constant) return;
	XMStoreFloat4x4(&constant->world, XMMatrixTranspose(XMMatrixIdentity()));
	XMStoreFloat4x4(&constant->view, XMMatrixTranspose(camera.GetViewMatrix()));
	XMStoreFloat4x4(&constant->proj, XMMatrixTranspose(camera.GetProjMatrix()));
	constant->color = kSurfaceColor;
	constant->color.w = m_timeSeconds * m_waveTimeScale; // アニメ用時刻
}

void FluidSystem::Draw(ID3D12GraphicsCommandList* cmd, const Camera& camera)
{
	if (m_mode == SimMode::Particles) {
		DrawParticles(cmd, camera); // 粒子描画（今はGameScene側で可視化するので空実装でもOK）
		return;
	}

	if (!cmd || !m_pipelineState || !m_pipelineState->IsValid()) return;

	UpdateCameraCB(camera);

	UINT frameIndex = g_Engine->CurrentBackBufferIndex();
	auto& cb = m_constantBuffers[frameIndex];
	if (!cb) return;

	cmd->SetGraphicsRootSignature(m_rootSignature->Get());
	cmd->SetPipelineState(m_pipelineState->Get());
	cmd->SetGraphicsRootConstantBufferView(0, cb->GetAddress());

	auto vbView = m_vertexBuffer->View();
	auto ibView = m_indexBuffer->View();
	cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	cmd->IASetVertexBuffers(0, 1, &vbView);
	cmd->IASetIndexBuffer(&ibView);
	cmd->DrawIndexedInstanced((UINT)m_indices.size(), 1, 0, 0, 0);
}

// 壁による境界調整(水の大きさも変化)
void FluidSystem::AdjustWall(const XMFLOAT3& direction, float amount)
{
	if (amount == 0.0f) return;

	XMVECTOR dirVec = XMLoadFloat3(&direction);
	if (XMVector3LengthSq(dirVec).m128_f32[0] < 1e-6f) return;
	dirVec = XMVector3Normalize(dirVec);

	XMFLOAT3 dir{}; XMStoreFloat3(&dir, dirVec);
	float absX = std::fabs(dir.x), absY = std::fabs(dir.y), absZ = std::fabs(dir.z);

	if (absX >= absY && absX >= absZ) {
		if (dir.x > 0.0f) m_bounds.max.x += amount; else m_bounds.min.x += amount;
	}
	else if (absZ >= absX && absZ >= absY) {
		if (dir.z > 0.0f) m_bounds.max.z += amount; else m_bounds.min.z += amount;
	}
	else {
		if (dir.y > 0.0f) m_bounds.max.y += amount; else m_bounds.min.y += amount;
		m_waterLevel = (m_bounds.min.y + m_bounds.max.y) * 0.5f;
	}

	if (m_bounds.max.x - m_bounds.min.x < m_minWallExtent) {
		float c = (m_bounds.max.x + m_bounds.min.x) * 0.5f;
		m_bounds.min.x = c - m_minWallExtent * 0.5f; m_bounds.max.x = c + m_minWallExtent * 0.5f;
	}
	if (m_bounds.max.z - m_bounds.min.z < m_minWallExtent) {
		float c = (m_bounds.max.z + m_bounds.min.z) * 0.5f;
		m_bounds.min.z = c - m_minWallExtent * 0.5f; m_bounds.max.z = c + m_minWallExtent * 0.5f;
	}
	if (m_bounds.max.y - m_bounds.min.y < 0.2f) {
		float c = (m_bounds.max.y + m_bounds.min.y) * 0.5f;
		m_bounds.min.y = c - 0.1f; m_bounds.max.y = c + 0.1f; m_waterLevel = c;
	}

	if (m_mode == SimMode::Heightfield) UpdateVertexBuffer();
}

// 壁による境界調整(水は固定)
void FluidSystem::AdjustWall(const DirectX::XMFLOAT3& direction, float amount, float deltaTime)
{
	if (amount == 0.0f) return;

	Bounds prev = m_bounds;

	using namespace DirectX;
	XMVECTOR dirVec = XMLoadFloat3(&direction);
	if (XMVector3LengthSq(dirVec).m128_f32[0] < 1e-6f) return;
	dirVec = XMVector3Normalize(dirVec);
	XMFLOAT3 dir{}; XMStoreFloat3(&dir, dirVec);

	float absX = std::fabs(dir.x), absY = std::fabs(dir.y), absZ = std::fabs(dir.z);

	if (absX >= absY && absX >= absZ) { if (dir.x > 0.0f) m_bounds.max.x += amount; else m_bounds.min.x += amount; }
	else if (absZ >= absX && absZ >= absY) { if (dir.z > 0.0f) m_bounds.max.z += amount; else m_bounds.min.z += amount; }
	else { if (dir.y > 0.0f) m_bounds.max.y += amount; else m_bounds.min.y += amount; m_waterLevel = (m_bounds.min.y + m_bounds.max.y) * 0.5f; }

	if (m_bounds.max.x - m_bounds.min.x < m_minWallExtent) { float c = (m_bounds.max.x + m_bounds.min.x) * 0.5f; m_bounds.min.x = c - m_minWallExtent * 0.5f; m_bounds.max.x = c + m_minWallExtent * 0.5f; }
	if (m_bounds.max.z - m_bounds.min.z < m_minWallExtent) { float c = (m_bounds.max.z + m_bounds.min.z) * 0.5f; m_bounds.min.z = c - m_minWallExtent * 0.5f; m_bounds.max.z = c + m_minWallExtent * 0.5f; }
	if (m_bounds.max.y - m_bounds.min.y < 0.2f) { float c = (m_bounds.max.y + m_bounds.min.y) * 0.5f; m_bounds.min.y = c - 0.1f; m_bounds.max.y = c + 0.1f; m_waterLevel = c; }

	// 高さ場：壁速度を波へ
	if (m_mode == SimMode::Heightfield) {
		ApplyWallImpulse(prev, m_bounds, deltaTime);
		UpdateVertexBuffer();
	}
	else {
		// 粒子：境界を縮めた分、はみ出し粒子を押し戻し
		for (auto& p : m_particles) ResolveWall(p.pos, p.vel);
	}
}

// レイと境界ボックスの交差判定
bool FluidSystem::RayIntersectBounds(const XMFLOAT3& origin, const XMFLOAT3& direction, XMFLOAT3& hitPoint) const
{
	XMFLOAT3 dir = direction;
	if (std::fabs(dir.x) < 1e-6f && std::fabs(dir.y) < 1e-6f && std::fabs(dir.z) < 1e-6f) return false;

	float tMin = 0.0f, tMax = FLT_MAX;
	auto updateInterval = [&](float ro, float rd, float bmin, float bmax)->bool {
		if (std::fabs(rd) < 1e-6f) return ro >= bmin && ro <= bmax;
		float inv = 1.0f / rd; float t0 = (bmin - ro) * inv, t1 = (bmax - ro) * inv; if (t0 > t1) std::swap(t0, t1);
		tMin = std::max(tMin, t0); tMax = std::min(tMax, t1); return tMax > tMin;
		};
	if (!updateInterval(origin.x, dir.x, m_bounds.min.x, m_bounds.max.x)) return false;
	if (!updateInterval(origin.y, dir.y, m_bounds.min.y, m_bounds.max.y)) return false;
	if (!updateInterval(origin.z, dir.z, m_bounds.min.z, m_bounds.max.z)) return false;

	float t = tMin;
	hitPoint = XMFLOAT3(origin.x + dir.x * t, origin.y + dir.y * t, origin.z + dir.z * t);
	return true;
}

// カメラ位置からの吸引・噴出リクエスト
void FluidSystem::SetCameraLiftRequest(const XMFLOAT3& origin, const XMFLOAT3& direction, float deltaTime)
{
	XMFLOAT3 hit{}; if (!RayIntersectBounds(origin, direction, hit)) return;

	float u = (hit.x - m_bounds.min.x) / std::max(m_bounds.max.x - m_bounds.min.x, 1e-3f);
	float v = (hit.z - m_bounds.min.z) / std::max(m_bounds.max.z - m_bounds.min.z, 1e-3f);
	u = std::clamp(u, 0.0f, 1.0f); v = std::clamp(v, 0.0f, 1.0f);

	if (m_mode == SimMode::Heightfield) {
		DropRequest d{}; d.uv = XMFLOAT2(u, v); d.strength = deltaTime * 14.0f; d.radius = 0.030f;
		m_pendingDrops.push_back(d); m_liftRequested = true;
	}
	else {
		// 粒子モード：中心へ吸引（半径は適当）
		const float kGatherR = 0.25f;
		ParticleAttractDisc(XMFLOAT2{ hit.x, hit.z }, kGatherR, /*strength*/4.0f, deltaTime);
	}
}
void FluidSystem::ClearCameraLiftRequest() { m_liftRequested = false; }

// 壁の移動による波を発生
void FluidSystem::ApplyWallImpulse(const Bounds& prev, const Bounds& curr, float dt)
{
	if (dt <= 0.0f || m_resolution <= 1 || m_cellDx <= 0.0f || m_cellDz <= 0.0f) return;

	const float invDt = 1.0f / dt;
	const float gain = 0.1f; const int band = 2;
	auto clampi = [&](int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); };

	auto addStripX = [&](int x, float vwall) {
		x = clampi(x, 0, m_resolution - 1);
		for (int b = 0; b < band; ++b) {
			int xi = clampi(x + b, 0, m_resolution - 1);
			for (int z = 0; z < m_resolution; ++z)
				m_velocity[Index((size_t)xi, (size_t)z)] += gain * vwall;
		}
		};
	auto addStripZ = [&](int z, float vwall) {
		z = clampi(z, 0, m_resolution - 1);
		for (int b = 0; b < band; ++b) {
			int zi = clampi(z + b, 0, m_resolution - 1);
			for (int x = 0; x < m_resolution; ++x)
				m_velocity[Index((size_t)x, (size_t)zi)] += gain * vwall;
		}
		};

	if (curr.min.x != prev.min.x) { float vwall = (curr.min.x - prev.min.x) * invDt; int strip = (int)std::round((curr.min.x - m_simMin.x) / m_cellDx); addStripX(strip, +vwall); }
	if (curr.max.x != prev.max.x) { float vwall = (prev.max.x - curr.max.x) * invDt; int strip = (int)std::round((curr.max.x - m_simMin.x) / m_cellDx) - 1; addStripX(strip, +vwall); }
	if (curr.min.z != prev.min.z) { float vwall = (curr.min.z - prev.min.z) * invDt; int strip = (int)std::round((curr.min.z - m_simMin.z) / m_cellDz); addStripZ(strip, +vwall); }
	if (curr.max.z != prev.max.z) { float vwall = (prev.max.z - curr.max.z) * invDt; int strip = (int)std::round((curr.max.z - m_simMin.z) / m_cellDz) - 1; addStripZ(strip, +vwall); }

	if (curr.min.y != prev.min.y) {
		float vwallY = (curr.min.y - prev.min.y) * invDt;
		for (int z = 0; z < m_resolution; ++z) for (int x = 0; x < m_resolution; ++x)
			m_velocity[Index((size_t)x, (size_t)z)] += 0.5f * gain * vwallY;
	}
	if (curr.max.y != prev.max.y) {
		float vwallY = (prev.max.y - curr.max.y) * invDt;
		for (int z = 0; z < m_resolution; ++z) for (int x = 0; x < m_resolution; ++x)
			m_velocity[Index((size_t)x, (size_t)z)] += 0.5f * gain * vwallY;
	}
}

// 円盤状の領域に波を発生（高さと速度）
void FluidSystem::ApplyDiscImpulse(const XMFLOAT2& c, float radius, float addHeight, float addVel)
{
	if (m_resolution <= 1 || radius <= 0.0f) return;

	const int xr = std::max(1, int(std::ceil(radius / m_cellDx)));
	const int zr = std::max(1, int(std::ceil(radius / m_cellDz)));
	int cx = int(std::round((c.x - m_simMin.x) / m_cellDx));
	int cz = int(std::round((c.y - m_simMin.z) / m_cellDz));
	auto clampi = [&](int v) { return std::max(0, std::min(m_resolution - 1, v)); };
	const float r2 = radius * radius;

	for (int z = clampi(cz - zr); z <= clampi(cz + zr); ++z)
		for (int x = clampi(cx - xr); x <= clampi(cx + xr); ++x)
		{
			float wx = m_simMin.x + m_cellDx * float(x);
			float wz = m_simMin.z + m_cellDz * float(z);
			float dx = wx - c.x, dz = wz - c.y;
			float d2 = dx * dx + dz * dz; if (d2 > r2) continue;
			float w = 1.0f - std::sqrt(d2) / radius; w = w * w * (3.0f - 2.0f * w);
			size_t idx = Index((size_t)x, (size_t)z);
			if (addHeight != 0.0f) m_height[idx] += addHeight * w;
			if (addVel != 0.0f) m_velocity[idx] += addVel * w;
		}
}

// ======= 水を操作系 =======
void FluidSystem::BeginGrab(const XMFLOAT2& xz, float radius) { m_gatherActive = true; m_gatherCenter = xz; m_gatherRadius = radius; m_gatheredVolume = 0.0f; }
void FluidSystem::UpdateGrab(const XMFLOAT2& xz, float liftPerSec, float dt) {
	if (!m_gatherActive) return; m_gatherCenter = xz;
	ApplyDiscImpulse(m_gatherCenter, m_gatherRadius, liftPerSec * dt, 0.0f);
}
void FluidSystem::EndGrab(const XMFLOAT2& throwDirXZ, float throwSpeed)
{
	if (!m_gatherActive) return; m_gatherActive = false;
	WavePacket p{}; p.center = m_gatherCenter;
	XMVECTOR d = XMVector2Normalize(XMLoadFloat2(&throwDirXZ)); XMStoreFloat2(&p.vel, XMVectorScale(d, throwSpeed));
	p.radius = m_packetDefaultRadius; p.amp = m_packetDefaultAmp; p.life = 1.2f; m_packets.push_back(p);
}
void FluidSystem::CutWater(const XMFLOAT2& a, const XMFLOAT2& b, float radius, float depth)
{
	const int N = 10;
	for (int i = 0; i <= N; ++i) {
		float t = float(i) / float(N);
		XMFLOAT2 p{ a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t };
		ApplyDiscImpulse(p, radius, -depth, 0.0f);
	}
}

// ===========================

// 波パケットと飛沫(まだ未実装：重すぎた為調整中)
void FluidSystem::UpdateInteractions(float dt)
{
	// 波パケットと飛沫の更新
	for (size_t i = 0; i < m_packets.size(); )
	{
		auto& p = m_packets[i];
		p.center.x += p.vel.x * dt; p.center.y += p.vel.y * dt;
		p.vel.x *= std::max(0.0f, 1.0f - m_packetFriction * dt);
		p.vel.y *= std::max(0.0f, 1.0f - m_packetFriction * dt);
		p.amp *= std::pow(m_packetDecay, dt);
		ApplyDiscImpulse(p.center, p.radius, 0.0f, +2.5f * p.amp);

		// 飛沫生成
		float dist = std::sqrt(p.vel.x * p.vel.x + p.vel.y * p.vel.y) * dt;
		int spawnN = int(m_sprayRate * dist);
		for (int n = 0; n < spawnN; ++n) {
			float ang = (float)n / std::max(1, spawnN) * 6.28318f;
			float rx = p.center.x + p.radius * std::cos(ang);
			float rz = p.center.y + p.radius * std::sin(ang);
			int ix = std::clamp(int(std::round((rx - m_simMin.x) / m_cellDx)), 0, m_resolution - 1);
			int iz = std::clamp(int(std::round((rz - m_simMin.z) / m_cellDz)), 0, m_resolution - 1);
			float y = m_waterLevel + m_height[Index((size_t)ix, (size_t)iz)];
			SprayParticle sp{}; sp.pos = XMFLOAT3(rx, y, rz);
			sp.vel = XMFLOAT3(p.vel.x * 0.4f, 3.0f + 0.6f * std::abs(p.amp), p.vel.y * 0.4f);
			sp.life = m_sprayLife; m_spray.push_back(sp);
		}

		p.life -= dt;
		if (p.life <= 0.0f || p.amp < 0.01f) m_packets.erase(m_packets.begin() + i); else ++i;
	}

	// 飛沫の更新
	for (size_t i = 0; i < m_spray.size(); )
	{
		auto& s = m_spray[i];
		s.vel.y += m_sprayGravity * dt;
		s.vel.x *= std::max(0.0f, 1.0f - m_sprayDrag * dt);
		s.vel.z *= std::max(0.0f, 1.0f - m_sprayDrag * dt);
		s.pos.x += s.vel.x * dt; s.pos.y += s.vel.y * dt; s.pos.z += s.vel.z * dt;
		s.life -= dt;

		int ix = std::clamp(int(std::round((s.pos.x - m_simMin.x) / m_cellDx)), 0, m_resolution - 1);
		int iz = std::clamp(int(std::round((s.pos.z - m_simMin.z) / m_cellDz)), 0, m_resolution - 1);
		float y = m_waterLevel + m_height[Index((size_t)ix, (size_t)iz)];
		if (s.life <= 0.0f || s.pos.y <= y) m_spray.erase(m_spray.begin() + i); else ++i;
	}
}

// ====== 粒子（SPHライト） ======
void FluidSystem::InitParticles(int count)
{
	m_particles.clear(); m_particles.reserve(count);

	const float s = m_kernelH * 0.9f; // 粒径間隔
	int nx = std::max(1, (int)((m_bounds.max.x - m_bounds.min.x) / s) - 2);
	int ny = std::max(1, (int)((m_bounds.max.y - m_bounds.min.y) / s) - 2);
	int nz = std::max(1, (int)((m_bounds.max.z - m_bounds.min.z) / s) - 2);

	for (int z = 1; z < nz && (int)m_particles.size() < count; ++z)
		for (int y = 1; y < ny && (int)m_particles.size() < count; ++y)
			for (int x = 1; x < nx && (int)m_particles.size() < count; ++x)
			{
				Particle p;
				p.pos = { m_bounds.min.x + x * s, m_bounds.min.y + y * s, m_bounds.min.z + z * s };
				p.vel = { 0,0,0 };
				m_particles.push_back(p);
				if ((int)m_particles.size() >= count) break;
			}

	m_particleRadius = s * 0.5f;
	float Vp = s * s * s * 0.8f;      // 充填率0.8
	m_mass = m_rho0 * Vp;
}

void FluidSystem::BuildGrid()
{
	m_grid.clear(); const float invH = 1.0f / m_kernelH;
	for (int i = 0; i < (int)m_particles.size(); ++i) {
		auto& p = m_particles[i];
		int ix = (int)std::floor((p.pos.x - m_bounds.min.x) * invH);
		int iy = (int)std::floor((p.pos.y - m_bounds.min.y) * invH);
		int iz = (int)std::floor((p.pos.z - m_bounds.min.z) * invH);
		m_grid[HashCell(ix, iy, iz)].push_back(i);
	}
}

void FluidSystem::ForNeighbors(int i, const std::function<void(int j, const XMFLOAT3& rij, float rlen)>& fn)
{
	auto& pi = m_particles[i];
	const float invH = 1.0f / m_kernelH;
	int ix = (int)std::floor((pi.pos.x - m_bounds.min.x) * invH);
	int iy = (int)std::floor((pi.pos.y - m_bounds.min.y) * invH);
	int iz = (int)std::floor((pi.pos.z - m_bounds.min.z) * invH);

	for (int dz = -1; dz <= 1; ++dz)
		for (int dy = -1; dy <= 1; ++dy)
			for (int dx = -1; dx <= 1; ++dx)
			{
				auto it = m_grid.find(HashCell(ix + dx, iy + dy, iz + dz));
				if (it == m_grid.end()) continue;
				for (int j : it->second) {
					if (j == i) continue;
					XMFLOAT3 rij{ pi.pos.x - m_particles[j].pos.x,
								  pi.pos.y - m_particles[j].pos.y,
								  pi.pos.z - m_particles[j].pos.z };
					float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
					if (r2 < m_kernelH * m_kernelH) fn(j, rij, std::sqrt(r2));
				}
			}
}

void FluidSystem::SPH_DensityPressure()
{
	const float h = m_kernelH, h2 = h * h;
	const float poly6 = 315.0f / (64.0f * 3.14159265f * std::pow(h, 9));

	for (auto& p : m_particles) p.density = 0.0f;

	for (int i = 0; i < (int)m_particles.size(); ++i) {
		float rho = 0.0f;
		ForNeighbors(i, [&](int j, const XMFLOAT3& rij, float r) {
			float t = h2 - r * r; if (t > 0) rho += m_mass * poly6 * t * t * t;
			});
		// self term
		rho += m_mass * poly6 * h2 * h2 * h2;
		m_particles[i].density = std::max(rho, m_rho0 * 0.1f);
		m_particles[i].pressure = m_stiffness * std::max(m_particles[i].density - m_rho0, 0.0f);
	}
}

void FluidSystem::SPH_ForcesIntegrate(float dt)
{
	using namespace DirectX;
	const float h = m_kernelH;
	const float spiky = -45.0f / (3.14159265f * std::pow(h, 6));
	const float visc = 45.0f / (3.14159265f * std::pow(h, 6));

	for (int i = 0; i < (int)m_particles.size(); ++i) {
		XMVECTOR Fi = XMVectorSet(0, -9.8f * m_particles[i].density, 0, 0);
		XMVECTOR vi = XMLoadFloat3(&m_particles[i].vel);
		XMVECTOR xi = XMLoadFloat3(&m_particles[i].pos);

		ForNeighbors(i, [&](int j, const XMFLOAT3& rij, float r) {
			if (r <= 1e-6f) return;
			float pij = (m_particles[i].pressure + m_particles[j].pressure) / (2.0f * m_particles[j].density);
			XMVECTOR grad = XMVectorScale(XMVectorSet(rij.x, rij.y, rij.z, 0) / r, spiky * (h - r) * (h - r));
			Fi += XMVectorScale(grad, -m_mass * pij);

			XMVECTOR vj = XMLoadFloat3(&m_particles[j].vel);
			XMVECTOR viscF = XMVectorScale((vj - vi), m_viscosity * m_mass * visc * (h - r) / m_particles[j].density);
			Fi += viscF;
			});

		// XSPH
		XMVECTOR vxsph = XMVectorZero(); float wsum = 0.0f;
		ForNeighbors(i, [&](int j, const XMFLOAT3& rij, float r) {
			float w = std::max(0.0f, 1.0f - r / h);
			XMVECTOR vj = XMLoadFloat3(&m_particles[j].vel);
			vxsph += w * (vj - vi); wsum += w;
			});
		if (wsum > 0) vxsph *= (m_xsph / wsum);

		XMVECTOR ai = Fi / std::max(m_particles[i].density, 1.0f);
		vi += ai * dt + vxsph;
		xi += vi * dt;

		XMFLOAT3 v, x; XMStoreFloat3(&v, vi); XMStoreFloat3(&x, xi);
		ResolveWall(x, v);
		m_particles[i].vel = v; m_particles[i].pos = x;
	}
}

void FluidSystem::ResolveWall(XMFLOAT3& p, XMFLOAT3& v, float restitution, float friction)
{
	auto hitAxis = [&](float& coord, float& vel, float minv, float maxv) {
		if (coord < minv) { coord = minv; vel = -vel * restitution; }
		if (coord > maxv) { coord = maxv; vel = -vel * restitution; }
		};
	hitAxis(p.x, v.x, m_bounds.min.x + m_particleRadius, m_bounds.max.x - m_particleRadius);
	hitAxis(p.y, v.y, m_bounds.min.y + m_particleRadius, m_bounds.max.y - m_particleRadius);
	hitAxis(p.z, v.z, m_bounds.min.z + m_particleRadius, m_bounds.max.z - m_particleRadius);
	v.x *= (1.0f - friction * 0.02f);
	v.z *= (1.0f - friction * 0.02f);
}

void FluidSystem::UpdateParticles(float dt)
{
	float remain = dt;
	while (remain > 1e-6f) {
		float sdt = std::min(remain, 0.016f);

		float vmax = 0.1f;
		for (auto& p : m_particles) {
			float s = std::sqrt(p.vel.x * p.vel.x + p.vel.y * p.vel.y + p.vel.z * p.vel.z);
			vmax = std::max(vmax, s);
		}
		float cfl = 0.4f * m_kernelH / std::max(0.5f, vmax);
		sdt = std::min(sdt, cfl);

		BuildGrid();
		SPH_DensityPressure();
		SPH_ForcesIntegrate(sdt);

		remain -= sdt;
	}
}

void FluidSystem::DrawParticles(ID3D12GraphicsCommandList* cmd, const Camera& cam)
{
        if (!cmd || !m_particleDepthPSO || !m_particleDepthPSO->IsValid() ||
                !m_particleCompositePSO || !m_particleCompositePSO->IsValid()) {
                return;
        }

        const UINT particleCount = static_cast<UINT>(m_particles.size());
        if (particleCount == 0) {
                return;
        }

        if (!m_particleInstanceVB || !m_particleQuadVB || !m_particleQuadIB || !m_particleDepthSrv) {
                return;
        }

        // インスタンスバッファへ粒子中心と半径を書き込む
        if (auto* resource = m_particleInstanceVB->GetResource()) {
                ParticleInstanceGPU* mapped = nullptr;
                if (SUCCEEDED(resource->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) {
                        for (UINT i = 0; i < particleCount; ++i) {
                                mapped[i].pos = m_particles[i].pos;
                                mapped[i].radius = m_particleRadius;
                        }
                        resource->Unmap(0, nullptr);
                }
        }

        UINT frameIndex = g_Engine->CurrentBackBufferIndex();
        auto& depthCB = m_particleDepthCB[frameIndex];
        auto& compCB = m_particleCompositeCB[frameIndex];
        if (!depthCB || !compCB) {
                return;
        }

        // 深度描画用定数バッファ更新
        if (auto* constants = depthCB->GetPtr<ParticleDepthConstant>()) {
                XMStoreFloat4x4(&constants->view, XMMatrixTranspose(cam.GetViewMatrix()));
                XMStoreFloat4x4(&constants->proj, XMMatrixTranspose(cam.GetProjMatrix()));
                constants->clipZ = XMFLOAT2(kCameraNear, kCameraFar);
                constants->pad = XMFLOAT2(0.0f, 0.0f);
        }

        // 合成用定数バッファ更新
        const float invWidth = 1.0f / std::max(1u, g_Engine->FrameBufferWidth());
        const float invHeight = 1.0f / std::max(1u, g_Engine->FrameBufferHeight());
        if (auto* constants = compCB->GetPtr<ParticleCompositeConstant>()) {
                constants->misc0 = XMFLOAT4(invWidth, invHeight, m_particleRadius, m_particleRadius * 12.0f);
                constants->misc1 = XMFLOAT4(kSurfaceColor.x, kSurfaceColor.y, kSurfaceColor.z, 0.65f);
                constants->misc2 = XMFLOAT4(0.25f, 0.85f, -0.35f, 48.0f);
        }

        // 深度テクスチャをレンダーターゲットへ遷移
        if (m_particleDepthTexture && m_particleDepthState != D3D12_RESOURCE_STATE_RENDER_TARGET) {
                auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
                        m_particleDepthTexture.Get(),
                        m_particleDepthState,
                        D3D12_RESOURCE_STATE_RENDER_TARGET);
                cmd->ResourceBarrier(1, &barrier);
                m_particleDepthState = D3D12_RESOURCE_STATE_RENDER_TARGET;
        }

        // 深度パス描画
        cmd->OMSetRenderTargets(1, &m_particleDepthRtv, FALSE, nullptr);
        const float clearColor[4] = { 0.0f,0.0f,0.0f,0.0f };
        cmd->ClearRenderTargetView(m_particleDepthRtv, clearColor, 0, nullptr);

        cmd->SetGraphicsRootSignature(m_particleDepthRoot->Get());
        cmd->SetPipelineState(m_particleDepthPSO->Get());
        cmd->SetGraphicsRootConstantBufferView(0, depthCB->GetAddress());

        D3D12_VERTEX_BUFFER_VIEW vbViews[2];
        vbViews[0] = m_particleQuadVB->View();
        vbViews[1] = m_particleInstanceVB->View();
        vbViews[1].SizeInBytes = sizeof(ParticleInstanceGPU) * particleCount; // 無駄な読みを避けるため実サイズ指定
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd->IASetVertexBuffers(0, 2, vbViews);
        auto ibView = m_particleQuadIB->View();
        cmd->IASetIndexBuffer(&ibView);
        cmd->DrawIndexedInstanced(6, particleCount, 0, 0, 0);

        // 深度テクスチャをシェーダーリソースへ遷移
        if (m_particleDepthTexture && m_particleDepthState != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE) {
                auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
                        m_particleDepthTexture.Get(),
                        m_particleDepthState,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
                cmd->ResourceBarrier(1, &barrier);
                m_particleDepthState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        }

        // バックバッファへ戻して合成パスを実行
        auto backRtv = g_Engine->CurrentBackBufferView();
        auto dsv = g_Engine->DepthStencilView();
        cmd->OMSetRenderTargets(1, &backRtv, FALSE, &dsv);

        auto* srvHeap = g_Engine->CbvSrvUavHeap();
        if (!srvHeap) {
                return;
        }
        ID3D12DescriptorHeap* heaps[] = { srvHeap->GetHeap() };
        cmd->SetDescriptorHeaps(1, heaps);

        cmd->SetGraphicsRootSignature(m_particleCompositeRoot->Get());
        cmd->SetPipelineState(m_particleCompositePSO->Get());
        cmd->SetGraphicsRootDescriptorTable(0, m_particleDepthSrv->HandleGPU);
        cmd->SetGraphicsRootConstantBufferView(1, compCB->GetAddress());
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd->DrawInstanced(3, 1, 0, 0);
}

// ===============================

// 追加インタラクション
void FluidSystem::ParticleAttractDisc(const XMFLOAT2& c, float r, float k, float dt) {
	float r2 = r * r;
	for (auto& p : m_particles) {
		float dx = p.pos.x - c.x, dz = p.pos.z - c.y; float d2 = dx * dx + dz * dz; if (d2 > r2) continue;
		float d = std::sqrt(std::max(d2, 1e-6f)); float w = 1.0f - d / r;
		float ax = -k * (dx / (d + 1e-3f)); float az = -k * (dz / (d + 1e-3f));
		p.vel.x += ax * dt * w; p.vel.z += az * dt * w;
	}
}

void FluidSystem::ParticleLaunch(const XMFLOAT2& dir, float speed) {
	for (auto& p : m_particles) { p.vel.x += dir.x * speed; p.vel.z += dir.y * speed; p.vel.y += 0.5f * speed; }
}

void FluidSystem::UpdateBlob(float dt)
{
	if (!m_blob.active || dt <= 0.0f) return;

	// 単純な重力運動
	m_blob.vel.y += -9.8f * dt;
	m_blob.pos.x += m_blob.vel.x * dt;
	m_blob.pos.y += m_blob.vel.y * dt;
	m_blob.pos.z += m_blob.vel.z * dt;

	// 画面外に飛び続けるのを防ぐ（任意のフェイルセーフ）
	if (m_blob.pos.y < m_bounds.min.y - 5.0f) {
		m_blob.active = false;
		return;
	}

	// 水面との衝突判定（高さ場がある前提：Heightfieldモードでのみ有効）
	if (m_resolution <= 1 || m_cellDx <= 0.0f || m_cellDz <= 0.0f) return;

	auto clampi = [&](int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); };

	int ix = clampi((int)std::round((m_blob.pos.x - m_simMin.x) / m_cellDx), 0, m_resolution - 1);
	int iz = clampi((int)std::round((m_blob.pos.z - m_simMin.z) / m_cellDz), 0, m_resolution - 1);
	float ySurface = m_waterLevel + m_height[Index((size_t)ix, (size_t)iz)];

	// 球の底が水面に触れたら着水
	if (m_blob.pos.y - m_blob.radius <= ySurface) {
		// 集めていた体積を水面へ戻す（少し広く分配）
		DepositVolumeGaussian(DirectX::XMFLOAT2{ m_blob.pos.x, m_blob.pos.z },
			m_blob.volume,
			/*sigma=*/m_blob.radius * 1.5f);

		// 着水インパルスで波を起こす（落下速度に応じて）
		float addVel = std::max(0.0f, -0.8f * m_blob.vel.y);
		ApplyDiscImpulse(DirectX::XMFLOAT2{ m_blob.pos.x, m_blob.pos.z },
			/*radius=*/m_blob.radius * 1.2f,
			/*addHeight=*/0.0f,
			/*addVel=*/addVel);

		m_blob.active = false;
	}
}

// 体積 [m^3] を中心XZにガウス分布で水面高さへ加算（HeightFieldモード用）
void FluidSystem::DepositVolumeGaussian(const DirectX::XMFLOAT2& centerXZ, float volume, float sigma)
{
	// HeightFieldの時だけ
	if (m_mode != SimMode::Heightfield) return;

	if (m_resolution <= 1 || m_cellDx <= 0.0f || m_cellDz <= 0.0f) return;
	if (sigma <= 0.0f || std::abs(volume) < 1e-12f) return;

	// 影響半径は3σ程度
	const float radius = 3.0f * sigma;
	const float r2 = radius * radius;

	// セル範囲に変換
	auto clampi = [&](int v) { return std::max(0, std::min(m_resolution - 1, v)); };
	const int cx = clampi((int)std::round((centerXZ.x - m_simMin.x) / m_cellDx));
	const int cz = clampi((int)std::round((centerXZ.y - m_simMin.z) / m_cellDz));
	const int rx = std::max(1, (int)std::ceil(radius / m_cellDx));
	const int rz = std::max(1, (int)std::ceil(radius / m_cellDz));

	// まず重み総和を求める（正規化のため）
	double wsum = 0.0;
	for (int z = clampi(cz - rz); z <= clampi(cz + rz); ++z)
		for (int x = clampi(cx - rx); x <= clampi(cx + rx); ++x)
		{
			const float wx = m_simMin.x + m_cellDx * float(x);
			const float wz = m_simMin.z + m_cellDz * float(z);
			const float dx = wx - centerXZ.x;
			const float dz = wz - centerXZ.y;
			const float d2 = dx * dx + dz * dz;
			if (d2 > r2) continue;
			const double w = std::exp(-0.5 * (double)d2 / (double)(sigma * sigma)); // ガウス
			wsum += w;
		}
	if (wsum <= 1e-12) return;

	// セル面積で体積→高さへ変換しつつ分配
	const double cellArea = (double)m_cellDx * (double)m_cellDz;
	for (int z = clampi(cz - rz); z <= clampi(cz + rz); ++z)
		for (int x = clampi(cx - rx); x <= clampi(cx + rx); ++x)
		{
			const float wx = m_simMin.x + m_cellDx * float(x);
			const float wz = m_simMin.z + m_cellDz * float(z);
			const float dx = wx - centerXZ.x;
			const float dz = wz - centerXZ.y;
			const float d2 = dx * dx + dz * dz;
			if (d2 > r2) continue;

			const double w = std::exp(-0.5 * (double)d2 / (double)(sigma * sigma));
			const double dh = (double)volume / cellArea * (w / wsum); // [m]

			const size_t idx = Index((size_t)x, (size_t)z);
			m_height[idx] += (float)dh;
		}
}

// 水関係の更新処理
void FluidSystem::Update(float deltaTime)
{
	if (deltaTime <= 0.0f) return;
	m_timeSeconds += deltaTime;

	if (m_mode == SimMode::Particles) {
		UpdateParticles(deltaTime);
		return;
	}

	ApplyPendingDrops();
	StepSimulation(deltaTime);
	UpdateInteractions(deltaTime);
	UpdateBlob(deltaTime);
	UpdateVertexBuffer();
}
