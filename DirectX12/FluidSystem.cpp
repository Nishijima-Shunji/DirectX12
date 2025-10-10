#include "FluidSystem.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <d3dx12.h>
#include "Engine.h"

using namespace DirectX;

namespace
{
	// 日本語コメント: 波面色を固定値で設定
	constexpr XMFLOAT4 kSurfaceColor{ 0.0f, 0.45f, 0.8f, 0.9f };
}

FluidSystem::FluidSystem() = default;
FluidSystem::~FluidSystem() = default;

bool FluidSystem::Init(ID3D12Device* device, const Bounds& bounds, size_t /*particleCount*/)
{
	m_device = device;
	m_bounds = bounds;
	m_waterLevel = (bounds.min.y + bounds.max.y) * 0.5f;

	if (!BuildSimulationResources())
	{
		return false;
	}

	// 解像度(m_resolution)と m_bounds が決まった直後に1回だけ設定
	const float gridWidth = m_bounds.max.x - m_bounds.min.x;
	const float gridDepth = m_bounds.max.z - m_bounds.min.z;

	m_simMin = DirectX::XMFLOAT3(m_bounds.min.x, m_bounds.min.y, m_bounds.min.z);
	m_cellDx = (m_resolution > 1) ? gridWidth / float(m_resolution - 1) : 0.0f;
	m_cellDz = (m_resolution > 1) ? gridDepth / float(m_resolution - 1) : 0.0f;

	if (!BuildRenderResources())
	{
		return false;
	}

	ResetWaveState();
	UpdateVertexBuffer();
	return true;
}

bool FluidSystem::BuildSimulationResources()
{
	// 日本語コメント: 格子解像度に合わせたバッファを初期化
	const size_t total = static_cast<size_t>(m_resolution) * static_cast<size_t>(m_resolution);
	m_height.assign(total, 0.0f);
	m_velocity.assign(total, 0.0f);
	m_vertices.assign(total, {});

	// 日本語コメント: インデックスは三角形リストで生成
	m_indices.clear();
	m_indices.reserve((m_resolution - 1) * (m_resolution - 1) * 6);
	for (int z = 0; z < m_resolution - 1; ++z)
	{
		for (int x = 0; x < m_resolution - 1; ++x)
		{
			uint32_t i0 = static_cast<uint32_t>(Index(x, z));
			uint32_t i1 = static_cast<uint32_t>(Index(x + 1, z));
			uint32_t i2 = static_cast<uint32_t>(Index(x, z + 1));
			uint32_t i3 = static_cast<uint32_t>(Index(x + 1, z + 1));
			m_indices.push_back(i0);
			m_indices.push_back(i1);
			m_indices.push_back(i2);
			m_indices.push_back(i1);
			m_indices.push_back(i3);
			m_indices.push_back(i2);
		}
	}
	return true;
}

bool FluidSystem::BuildRenderResources()
{
	// 日本語コメント: 専用ルートシグネチャを生成
	CD3DX12_ROOT_PARAMETER params[1] = {};
	params[0].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_ALL);

	D3D12_ROOT_SIGNATURE_DESC desc{};
	desc.NumParameters = _countof(params);
	desc.pParameters = params;
	desc.NumStaticSamplers = 0;
	desc.pStaticSamplers = nullptr;
	desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

	m_rootSignature = std::make_unique<RootSignature>();
	if (!m_rootSignature || !m_rootSignature->Init(desc) || !m_rootSignature->IsValid())
	{
		return false;
	}

	m_pipelineState = std::make_unique<PipelineState>();
	if (!m_pipelineState)
	{
		return false;
	}

	D3D12_INPUT_ELEMENT_DESC layout[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, offsetof(OceanVertex, position), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT,    0, offsetof(OceanVertex, normal),   D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,       0, offsetof(OceanVertex, uv),       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT,0, offsetof(OceanVertex, color),    D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
	};
	m_pipelineState->SetInputLayout({ layout, _countof(layout) });
	m_pipelineState->SetRootSignature(m_rootSignature->Get());
	m_pipelineState->SetVS(L"OceanVS.cso");
	m_pipelineState->SetPS(L"OceanPS.cso");
	m_pipelineState->SetDepthStencilFormat(DXGI_FORMAT_D32_FLOAT);
	m_pipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
	if (!m_pipelineState->IsValid())
	{
		return false;
	}

	size_t vbSize = m_vertices.size() * sizeof(OceanVertex);
	m_vertexBuffer = std::make_unique<VertexBuffer>(vbSize, sizeof(OceanVertex), m_vertices.data());
	if (!m_vertexBuffer || !m_vertexBuffer->IsValid())
	{
		return false;
	}

	size_t ibSize = m_indices.size() * sizeof(uint32_t);
	m_indexBuffer = std::make_unique<IndexBuffer>(ibSize, m_indices.data());
	if (!m_indexBuffer || !m_indexBuffer->IsValid())
	{
		return false;
	}

	for (auto& cb : m_constantBuffers)
	{
		cb = std::make_unique<ConstantBuffer>(sizeof(OceanConstant));
		if (!cb || !cb->IsValid())
		{
			return false;
		}
	}

	return true;
}

void FluidSystem::ResetWaveState()
{
	// 日本語コメント: 初期波面を静止状態でセット
	std::fill(m_height.begin(), m_height.end(), 0.0f);
	std::fill(m_velocity.begin(), m_velocity.end(), 0.0f);
}

void FluidSystem::Update(float deltaTime)
{
	if (deltaTime <= 0.0f)
	{
		return;
	}

	m_timeSeconds += deltaTime;

	ApplyPendingDrops();
	StepSimulation(deltaTime);
	UpdateVertexBuffer();
}

void FluidSystem::ApplyPendingDrops()
{
	for (const auto& drop : m_pendingDrops)
	{
		ApplyDrop(drop);
	}
	m_pendingDrops.clear();
}

void FluidSystem::ApplyDrop(const DropRequest& drop)
{
	// 日本語コメント: UV座標から格子インデックスへ変換
	float fx = std::clamp(drop.uv.x * static_cast<float>(m_resolution - 1), 0.0f, static_cast<float>(m_resolution - 1));
	float fz = std::clamp(drop.uv.y * static_cast<float>(m_resolution - 1), 0.0f, static_cast<float>(m_resolution - 1));
	int centerX = static_cast<int>(std::round(fx));
	int centerZ = static_cast<int>(std::round(fz));

	const float radius = drop.radius * static_cast<float>(m_resolution);
	const float radiusSq = radius * radius;

	for (int z = std::max(0, centerZ - static_cast<int>(radius)); z <= std::min(m_resolution - 1, centerZ + static_cast<int>(radius)); ++z)
	{
		for (int x = std::max(0, centerX - static_cast<int>(radius)); x <= std::min(m_resolution - 1, centerX + static_cast<int>(radius)); ++x)
		{
			float dx = static_cast<float>(x) - fx;
			float dz = static_cast<float>(z) - fz;
			float distSq = dx * dx + dz * dz;
			if (distSq > radiusSq)
			{
				continue;
			}

			float falloff = 1.0f - (distSq / radiusSq);
			size_t idx = Index(static_cast<size_t>(x), static_cast<size_t>(z));
			m_velocity[idx] += drop.strength * falloff;
		}
	}
}

void FluidSystem::StepSimulation(float deltaTime)
{
	const float gridWidth = m_bounds.max.x - m_bounds.min.x;
	const float gridDepth = m_bounds.max.z - m_bounds.min.z;
	if (gridWidth <= 0.0f || gridDepth <= 0.0f)
	{
		return;
	}

	/*const float dt = std::min(deltaTime, 0.033f);
	const float coeff = (m_waveSpeed * m_waveSpeed);

	std::vector<float> newHeight(m_height.size(), 0.0f);

	for (int z = 0; z < m_resolution; ++z)
	{
		for (int x = 0; x < m_resolution; ++x)
		{
			size_t idx = Index(static_cast<size_t>(x), static_cast<size_t>(z));
			float h = m_height[idx];

			auto sample = [&](int sx, int sz)
			{
				sx = std::clamp(sx, 0, m_resolution - 1);
				sz = std::clamp(sz, 0, m_resolution - 1);
				return m_height[Index(static_cast<size_t>(sx), static_cast<size_t>(sz))];
			};

			float lap = sample(x - 1, z) + sample(x + 1, z) + sample(x, z - 1) + sample(x, z + 1) - 4.0f * h;
			float accel = coeff * lap;
			m_velocity[idx] += accel * dt;
			m_velocity[idx] *= m_damping;
			newHeight[idx] = h + m_velocity[idx] * dt;
		}
	}*/
	//m_height.swap(newHeight);

	const float dtMax = std::min(deltaTime, 0.033f);
	const float dx = gridWidth / static_cast<float>(m_resolution - 1);
	const float dz = gridDepth / static_cast<float>(m_resolution - 1);
	const float hcell = std::max(1e-5f, std::min(dx, dz));
	const float c = std::max(1e-4f, m_waveSpeed);
	const float dtCFL = 0.5f * (hcell / c); // 安定目安
	const int   steps = std::max(1, (int)std::ceil(dtMax / dtCFL));
	const float sdt = dtMax / static_cast<float>(steps);
	// 減衰は“秒基準”で一定にする（フレームレートに依存させない）
	const float dampPerStep = std::pow(m_damping, sdt / (1.0f / 60.0f));
	const float coeff = (m_waveSpeed * m_waveSpeed);

	for (int s = 0; s < steps; ++s)
	{
		std::vector<float> newHeight(m_height.size(), 0.0f);
		for (int z = 0; z < m_resolution; ++z)
		{
			for (int x = 0; x < m_resolution; ++x)
			{
				size_t idx = Index((size_t)x, (size_t)z);
				float h = m_height[idx];

				auto sample = [&](int sx, int sz)
					{
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
		}
		m_height.swap(newHeight);
	}

}

void FluidSystem::UpdateVertexBuffer()
{
	if (m_cellDx <= 0.0f || m_cellDz <= 0.0f) return;

	// ① 頂点の XZ は「固定セル幅 × インデックス」＋ 初期原点
	for (int z = 0; z < m_resolution; ++z)
	{
		for (int x = 0; x < m_resolution; ++x)
		{
			const size_t idx = Index((size_t)x, (size_t)z);

			const float worldX = m_simMin.x + m_cellDx * float(x);
			const float worldZ = m_simMin.z + m_cellDz * float(z);
			const float worldY = m_waterLevel + m_height[idx];

			auto& v = m_vertices[idx];
			v.position = XMFLOAT3(worldX, worldY, worldZ);

			// UV は 0..1 のままでOK（固定グリッドに対して安定）
			const float fx = float(x) / float(m_resolution - 1);
			const float fz = float(z) / float(m_resolution - 1);
			v.uv = XMFLOAT2(fx, fz);
			v.color = kSurfaceColor;
		}
	}

	// ② 法線は「中央差分 / (2*dx, 2*dz)」に修正（いまは×dx,×dz になっている）
	const float inv2dx = 1.0f / (2.0f * m_cellDx);
	const float inv2dz = 1.0f / (2.0f * m_cellDz);

	for (int z = 0; z < m_resolution; ++z)
	{
		for (int x = 0; x < m_resolution; ++x)
		{
			const int xm = std::max(0, x - 1);
			const int xp = std::min(m_resolution - 1, x + 1);
			const int zm = std::max(0, z - 1);
			const int zp = std::min(m_resolution - 1, z + 1);

			const float hx =
				m_height[Index((size_t)xp, (size_t)z)] -
				m_height[Index((size_t)xm, (size_t)z)];
			const float hz =
				m_height[Index((size_t)x, (size_t)zp)] -
				m_height[Index((size_t)x, (size_t)zm)];

			// ∂h/∂x ≈ (h[x+1]-h[x-1]) / (2*dx)
			// ∂h/∂z ≈ (h[z+1]-h[z-1]) / (2*dz)
			const float dhdx = hx * inv2dx;
			const float dhdz = hz * inv2dz;

			XMFLOAT3 nLocal(-dhdx, 1.0f, -dhdz);
			XMVECTOR n = XMVector3Normalize(XMLoadFloat3(&nLocal));
			XMStoreFloat3(&m_vertices[Index((size_t)x, (size_t)z)].normal, n);
		}
	}

	if (m_vertexBuffer && m_vertexBuffer->IsValid())
	{
		void* mapped = nullptr;
		auto resource = m_vertexBuffer->GetResource();
		if (resource && SUCCEEDED(resource->Map(0, nullptr, &mapped)))
		{
			std::memcpy(mapped, m_vertices.data(),
				m_vertices.size() * sizeof(OceanVertex));
			resource->Unmap(0, nullptr);
		}
	}
}


void FluidSystem::UpdateCameraCB(const Camera& camera)
{
	UINT frameIndex = g_Engine->CurrentBackBufferIndex();
	auto& cb = m_constantBuffers[frameIndex];
	if (!cb)
	{
		return;
	}

	OceanConstant* constant = cb->GetPtr<OceanConstant>();
	if (!constant)
	{
		return;
	}

	XMStoreFloat4x4(&constant->world, XMMatrixTranspose(XMMatrixIdentity()));
	XMStoreFloat4x4(&constant->view, XMMatrixTranspose(camera.GetViewMatrix()));
	XMStoreFloat4x4(&constant->proj, XMMatrixTranspose(camera.GetProjMatrix()));
	constant->color = kSurfaceColor;

	constant->color.w = m_timeSeconds * m_waveTimeScale;

}

void FluidSystem::Draw(ID3D12GraphicsCommandList* cmd, const Camera& camera)
{
	if (!cmd || !m_pipelineState || !m_pipelineState->IsValid())
	{
		return;
	}

	UpdateCameraCB(camera);

	UINT frameIndex = g_Engine->CurrentBackBufferIndex();
	auto& cb = m_constantBuffers[frameIndex];
	if (!cb)
	{
		return;
	}

	cmd->SetGraphicsRootSignature(m_rootSignature->Get());
	cmd->SetPipelineState(m_pipelineState->Get());
	cmd->SetGraphicsRootConstantBufferView(0, cb->GetAddress());

	auto vbView = m_vertexBuffer->View();
	auto ibView = m_indexBuffer->View();
	cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	cmd->IASetVertexBuffers(0, 1, &vbView);
	cmd->IASetIndexBuffer(&ibView);
	cmd->DrawIndexedInstanced(static_cast<UINT>(m_indices.size()), 1, 0, 0, 0);
}

void FluidSystem::AdjustWall(const XMFLOAT3& direction, float amount)
{
	if (amount == 0.0f)
	{
		return;
	}

	XMVECTOR dirVec = XMLoadFloat3(&direction);
	if (XMVector3LengthSq(dirVec).m128_f32[0] < 1e-6f)
	{
		return;
	}
	dirVec = XMVector3Normalize(dirVec);

	XMFLOAT3 dir{};
	XMStoreFloat3(&dir, dirVec);

	float absX = std::fabs(dir.x);
	float absY = std::fabs(dir.y);
	float absZ = std::fabs(dir.z);

	if (absX >= absY && absX >= absZ)
	{
		if (dir.x > 0.0f)
		{
			m_bounds.max.x += amount;
		}
		else
		{
			m_bounds.min.x += amount;
		}
	}
	else if (absZ >= absX && absZ >= absY)
	{
		if (dir.z > 0.0f)
		{
			m_bounds.max.z += amount;
		}
		else
		{
			m_bounds.min.z += amount;
		}
	}
	else
	{
		if (dir.y > 0.0f)
		{
			m_bounds.max.y += amount;
		}
		else
		{
			m_bounds.min.y += amount;
		}
		m_waterLevel = (m_bounds.min.y + m_bounds.max.y) * 0.5f;
	}

	if (m_bounds.max.x - m_bounds.min.x < m_minWallExtent)
	{
		float center = (m_bounds.max.x + m_bounds.min.x) * 0.5f;
		m_bounds.min.x = center - m_minWallExtent * 0.5f;
		m_bounds.max.x = center + m_minWallExtent * 0.5f;
	}
	if (m_bounds.max.z - m_bounds.min.z < m_minWallExtent)
	{
		float center = (m_bounds.max.z + m_bounds.min.z) * 0.5f;
		m_bounds.min.z = center - m_minWallExtent * 0.5f;
		m_bounds.max.z = center + m_minWallExtent * 0.5f;
	}
	if (m_bounds.max.y - m_bounds.min.y < 0.2f)
	{
		float center = (m_bounds.max.y + m_bounds.min.y) * 0.5f;
		m_bounds.min.y = center - 0.1f;
		m_bounds.max.y = center + 0.1f;
		m_waterLevel = center;
	}

	UpdateVertexBuffer();
}

void FluidSystem::AdjustWall(const DirectX::XMFLOAT3& direction, float amount, float deltaTime)
{
	if (amount == 0.0f) return;

	// --- 前状態を保持 ---
	Bounds prev = m_bounds;

	// 入力方向を正規化し、支配軸を決める（元のロジックを踏襲）
	using namespace DirectX;
	XMVECTOR dirVec = XMLoadFloat3(&direction);
	if (XMVector3LengthSq(dirVec).m128_f32[0] < 1e-6f) return;
	dirVec = XMVector3Normalize(dirVec);

	XMFLOAT3 dir{};
	XMStoreFloat3(&dir, dirVec);

	float absX = std::fabs(dir.x);
	float absY = std::fabs(dir.y);
	float absZ = std::fabs(dir.z);

	// --- 境界を動かす（元の分岐そのまま） ---
	if (absX >= absY && absX >= absZ)
	{
		if (dir.x > 0.0f) m_bounds.max.x += amount;
		else              m_bounds.min.x += amount;
	}
	else if (absZ >= absX && absZ >= absY)
	{
		if (dir.z > 0.0f) m_bounds.max.z += amount;
		else              m_bounds.min.z += amount;
	}
	else
	{
		if (dir.y > 0.0f) m_bounds.max.y += amount;
		else              m_bounds.min.y += amount;

		// 水位は従来通りセンタートラッキング
		m_waterLevel = (m_bounds.min.y + m_bounds.max.y) * 0.5f;
	}

	// --- 最小幅の維持（元のクランプをそのまま） ---
	if (m_bounds.max.x - m_bounds.min.x < m_minWallExtent) {
		float cx = (m_bounds.max.x + m_bounds.min.x) * 0.5f;
		m_bounds.min.x = cx - m_minWallExtent * 0.5f;
		m_bounds.max.x = cx + m_minWallExtent * 0.5f;
	}
	if (m_bounds.max.z - m_bounds.min.z < m_minWallExtent) {
		float cz = (m_bounds.max.z + m_bounds.min.z) * 0.5f;
		m_bounds.min.z = cz - m_minWallExtent * 0.5f;
		m_bounds.max.z = cz + m_minWallExtent * 0.5f;
	}
	if (m_bounds.max.y - m_bounds.min.y < 0.2f) {
		float cy = (m_bounds.max.y + m_bounds.min.y) * 0.5f;
		m_bounds.min.y = cy - 0.1f;
		m_bounds.max.y = cy + 0.1f;
		m_waterLevel = cy;
	}

	// --- 壁が動いた分を“速度”として境界帯に注入（押す/引く） ---
	ApplyWallImpulse(prev, m_bounds, deltaTime);

	// --- 描画用頂点更新（A対応でグリッドは固定アンカー） ---
	UpdateVertexBuffer();
}


bool FluidSystem::RayIntersectBounds(const XMFLOAT3& origin, const XMFLOAT3& direction, XMFLOAT3& hitPoint) const
{
	XMFLOAT3 dir = direction;
	if (std::fabs(dir.x) < 1e-6f && std::fabs(dir.y) < 1e-6f && std::fabs(dir.z) < 1e-6f)
	{
		return false;
	}

	float tMin = 0.0f;
	float tMax = FLT_MAX;

	auto updateInterval = [&](float rayOrigin, float rayDir, float boxMin, float boxMax) -> bool
		{
			if (std::fabs(rayDir) < 1e-6f)
			{
				return rayOrigin >= boxMin && rayOrigin <= boxMax;
			}
			float inv = 1.0f / rayDir;
			float t0 = (boxMin - rayOrigin) * inv;
			float t1 = (boxMax - rayOrigin) * inv;
			if (t0 > t1)
			{
				std::swap(t0, t1);
			}
			tMin = std::max(tMin, t0);
			tMax = std::min(tMax, t1);
			return tMax > tMin;
		};

	if (!updateInterval(origin.x, dir.x, m_bounds.min.x, m_bounds.max.x)) return false;
	if (!updateInterval(origin.y, dir.y, m_bounds.min.y, m_bounds.max.y)) return false;
	if (!updateInterval(origin.z, dir.z, m_bounds.min.z, m_bounds.max.z)) return false;

	float t = tMin;
	hitPoint = XMFLOAT3(origin.x + dir.x * t,
		origin.y + dir.y * t,
		origin.z + dir.z * t);
	return true;
}

void FluidSystem::SetCameraLiftRequest(const XMFLOAT3& origin, const XMFLOAT3& direction, float deltaTime)
{
	XMFLOAT3 hit{};
	if (!RayIntersectBounds(origin, direction, hit))
	{
		return;
	}

	float u = (hit.x - m_bounds.min.x) / std::max(m_bounds.max.x - m_bounds.min.x, 1e-3f);
	float v = (hit.z - m_bounds.min.z) / std::max(m_bounds.max.z - m_bounds.min.z, 1e-3f);
	u = std::clamp(u, 0.0f, 1.0f);
	v = std::clamp(v, 0.0f, 1.0f);

	DropRequest drop{};
	drop.uv = XMFLOAT2(u, v);
	/*drop.strength = deltaTime * 6.0f;
	drop.radius = 0.045f;*/
	drop.strength = deltaTime * 14.0f; // ★反応を鋭く
	drop.radius = 0.030f;
	m_pendingDrops.push_back(drop);
	m_liftRequested = true;
}

void FluidSystem::ClearCameraLiftRequest()
{
	m_liftRequested = false;
}

void FluidSystem::ApplyWallImpulse(const Bounds& prev, const Bounds& curr, float dt)
{
	if (dt <= 0.0f || m_resolution <= 1 || m_cellDx <= 0.0f || m_cellDz <= 0.0f) return;

	const float invDt = 1.0f / dt;
	const float gain = 0.1f;   // 押し/引きの効き（0.4〜1.0で調整）
	const int   band = 2;      // 何セル幅に入れるか

	auto clampi = [&](int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); };

	auto addStripX = [&](int x, float vwall) {
		x = clampi(x, 0, m_resolution - 1);
		for (int b = 0; b < band; ++b) {
			int xi = clampi(x + b, 0, m_resolution - 1);
			for (int z = 0; z < m_resolution; ++z) {
				m_velocity[Index((size_t)xi, (size_t)z)] += gain * vwall;
			}
		}
		};
	auto addStripZ = [&](int z, float vwall) {
		z = clampi(z, 0, m_resolution - 1);
		for (int b = 0; b < band; ++b) {
			int zi = clampi(z + b, 0, m_resolution - 1);
			for (int x = 0; x < m_resolution; ++x) {
				m_velocity[Index((size_t)x, (size_t)zi)] += gain * vwall;
			}
		}
		};

	// X最小面（min.x）
	if (curr.min.x != prev.min.x) {
		float delta = curr.min.x - prev.min.x;     // +: 内側へ押す, -: 外へ広げる
		float vwall = delta * invDt;
		int   strip = (int)std::round((curr.min.x - m_simMin.x) / m_cellDx);
		addStripX(strip, +vwall);
	}
	// X最大面（max.x）
	if (curr.max.x != prev.max.x) {
		float delta = prev.max.x - curr.max.x;     // +: 内側へ押す, -: 外へ広げる
		float vwall = delta * invDt;
		int   strip = (int)std::round((curr.max.x - m_simMin.x) / m_cellDx) - 1;
		addStripX(strip, +vwall);
	}
	// Z最小面（min.z）
	if (curr.min.z != prev.min.z) {
		float delta = curr.min.z - prev.min.z;     // +: 内側へ押す, -: 外へ広げる
		float vwall = delta * invDt;
		int   strip = (int)std::round((curr.min.z - m_simMin.z) / m_cellDz);
		addStripZ(strip, +vwall);
	}
	// Z最大面（max.z）
	if (curr.max.z != prev.max.z) {
		float delta = prev.max.z - curr.max.z;     // +: 内側へ押す, -: 外へ広げる
		float vwall = delta * invDt;
		int   strip = (int)std::round((curr.max.z - m_simMin.z) / m_cellDz) - 1;
		addStripZ(strip, +vwall);
	}

	// Y方向（上下の壁を動かした場合）：高さ場なので全域へ一様インパルスで波を起こす
	if (curr.min.y != prev.min.y) {
		float vwallY = (curr.min.y - prev.min.y) * invDt; // +: 上へ押す
		for (int z = 0; z < m_resolution; ++z)
			for (int x = 0; x < m_resolution; ++x)
				m_velocity[Index((size_t)x, (size_t)z)] += 0.5f * gain * vwallY;
	}
	if (curr.max.y != prev.max.y) {
		float vwallY = (prev.max.y - curr.max.y) * invDt; // +: 下へ押す
		for (int z = 0; z < m_resolution; ++z)
			for (int x = 0; x < m_resolution; ++x)
				m_velocity[Index((size_t)x, (size_t)z)] += 0.5f * gain * vwallY;
	}
}

