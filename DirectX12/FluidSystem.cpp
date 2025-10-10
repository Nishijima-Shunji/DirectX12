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

    const float dt = std::min(deltaTime, 0.033f);
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
    }

    m_height.swap(newHeight);
}

void FluidSystem::UpdateVertexBuffer()
{
    const float gridWidth = m_bounds.max.x - m_bounds.min.x;
    const float gridDepth = m_bounds.max.z - m_bounds.min.z;
    if (gridWidth <= 0.0f || gridDepth <= 0.0f)
    {
        return;
    }

    const float dx = gridWidth / static_cast<float>(m_resolution - 1);
    const float dz = gridDepth / static_cast<float>(m_resolution - 1);

    for (int z = 0; z < m_resolution; ++z)
    {
        for (int x = 0; x < m_resolution; ++x)
        {
            size_t idx = Index(static_cast<size_t>(x), static_cast<size_t>(z));
            float fx = static_cast<float>(x) / static_cast<float>(m_resolution - 1);
            float fz = static_cast<float>(z) / static_cast<float>(m_resolution - 1);

            float worldX = m_bounds.min.x + gridWidth * fx;
            float worldZ = m_bounds.min.z + gridDepth * fz;
            float worldY = m_waterLevel + m_height[idx];

            auto& v = m_vertices[idx];
            v.position = XMFLOAT3(worldX, worldY, worldZ);
            v.uv = XMFLOAT2(fx, fz);
            v.color = kSurfaceColor;
        }
    }

    // 日本語コメント: 法線は中央差分で計算
    for (int z = 0; z < m_resolution; ++z)
    {
        for (int x = 0; x < m_resolution; ++x)
        {
            size_t idx = Index(static_cast<size_t>(x), static_cast<size_t>(z));
            int xm = std::max(0, x - 1);
            int xp = std::min(m_resolution - 1, x + 1);
            int zm = std::max(0, z - 1);
            int zp = std::min(m_resolution - 1, z + 1);

            float hx = m_height[Index(static_cast<size_t>(xp), static_cast<size_t>(z))] - m_height[Index(static_cast<size_t>(xm), static_cast<size_t>(z))];
            float hz = m_height[Index(static_cast<size_t>(x), static_cast<size_t>(zp))] - m_height[Index(static_cast<size_t>(x), static_cast<size_t>(zm))];

            XMFLOAT3 normal{
                -hx * 0.5f * dx,
                1.0f,
                -hz * 0.5f * dz
            };

            XMVECTOR n = XMVector3Normalize(XMLoadFloat3(&normal));
            XMStoreFloat3(&m_vertices[idx].normal, n);
        }
    }

    if (m_vertexBuffer && m_vertexBuffer->IsValid())
    {
        void* mapped = nullptr;
        auto resource = m_vertexBuffer->GetResource();
        if (resource && SUCCEEDED(resource->Map(0, nullptr, &mapped)))
        {
            std::memcpy(mapped, m_vertices.data(), m_vertices.size() * sizeof(OceanVertex));
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
    drop.strength = deltaTime * 6.0f;
    drop.radius = 0.045f;
    m_pendingDrops.push_back(drop);
    m_liftRequested = true;
}

void FluidSystem::ClearCameraLiftRequest()
{
    m_liftRequested = false;
}
