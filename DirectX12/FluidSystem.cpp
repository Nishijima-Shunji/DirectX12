#include "FluidSystem.h"
#include "Camera.h"
#include "Engine.h"
#include "SphereMeshGenerator.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <random>

using namespace DirectX;

namespace
{
    constexpr float kGravity = -9.8f;       // 重力加速度
    constexpr float kWallThickness = 0.0f;  // AABB扱いなので実体厚みは不要
}

namespace
{
    // 球メッシュ用の入力レイアウト定義
    constexpr D3D12_INPUT_ELEMENT_DESC kSphereInputElements[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "POSITION", 1, DXGI_FORMAT_R32G32B32_FLOAT, 1, 0, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32_FLOAT,        1, 12, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
    };

    constexpr D3D12_INPUT_ELEMENT_DESC kGridLineInputElements[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };
}

FluidSystem::FluidSystem()
    : m_world(XMMatrixIdentity())
    , m_grid(m_supportRadius)
{
}

bool FluidSystem::Init(ID3D12Device* device, const Bounds& initialBounds, UINT particleCount)
{
    (void)device; // DirectXリソース生成は Engine 内のヘルパーを利用するため未使用

    m_bounds = initialBounds;
    InitializeParticles(particleCount);

    m_rootSignature = std::make_unique<RootSignature>();
    if (!m_rootSignature || !m_rootSignature->IsValid())
    {
        return false;
    }

    m_pipelineState = std::make_unique<PipelineState>();
    if (!m_pipelineState)
    {
        return false;
    }
    D3D12_INPUT_LAYOUT_DESC sphereLayout{};
    sphereLayout.pInputElementDescs = kSphereInputElements;
    sphereLayout.NumElements = _countof(kSphereInputElements);
    m_pipelineState->SetInputLayout(sphereLayout);
    m_pipelineState->SetRootSignature(m_rootSignature->Get());
    m_pipelineState->SetVS(L"ParticleVS.cso");
    m_pipelineState->SetPS(L"ParticlePS.cso");
    m_pipelineState->SetDepthStencilFormat(DXGI_FORMAT_D32_FLOAT);
    m_pipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    if (!m_pipelineState->IsValid())
    {
        return false;
    }

    MeshData sphereMesh = CreateLowPolySphere(1.0f, 1);
    std::vector<SphereVertex> sphereVertices;
    sphereVertices.reserve(sphereMesh.vertices.size());
    for (const auto& v : sphereMesh.vertices)
    {
        SphereVertex sv{};
        sv.position = v.Position;
        sv.normal = v.Normal;
        sphereVertices.push_back(sv);
    }

    const size_t sphereVBSize = sphereVertices.size() * sizeof(SphereVertex);
    m_sphereVertexBuffer = std::make_unique<VertexBuffer>(sphereVBSize, sizeof(SphereVertex), sphereVertices.data());
    if (!m_sphereVertexBuffer || !m_sphereVertexBuffer->IsValid())
    {
        return false;
    }

    const size_t sphereIBSize = sphereMesh.indices.size() * sizeof(uint32_t);
    m_sphereIndexBuffer = std::make_unique<IndexBuffer>(sphereIBSize, sphereMesh.indices.data());
    if (!m_sphereIndexBuffer || !m_sphereIndexBuffer->IsValid())
    {
        return false;
    }
    m_indexCount = static_cast<UINT>(sphereMesh.indices.size());

    size_t instanceBufferSize = sizeof(ParticleInstance) * m_instances.size();
    m_instanceBuffer = std::make_unique<VertexBuffer>(instanceBufferSize, sizeof(ParticleInstance), m_instances.data());
    if (!m_instanceBuffer || !m_instanceBuffer->IsValid())
    {
        return false;
    }

    m_gridPipelineState = std::make_unique<PipelineState>();
    if (!m_gridPipelineState)
    {
        return false;
    }
    D3D12_INPUT_LAYOUT_DESC gridLayout{};
    gridLayout.pInputElementDescs = kGridLineInputElements;
    gridLayout.NumElements = _countof(kGridLineInputElements);
    m_gridPipelineState->SetInputLayout(gridLayout);
    m_gridPipelineState->SetRootSignature(m_rootSignature->Get());
    m_gridPipelineState->SetVS(L"GridLineVS.cso");
    m_gridPipelineState->SetPS(L"GridLinePS.cso");
    m_gridPipelineState->SetDepthStencilFormat(DXGI_FORMAT_D32_FLOAT);
    m_gridPipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE);
    if (!m_gridPipelineState->IsValid())
    {
        return false;
    }

    for (auto& cb : m_constantBuffers)
    {
        cb = std::make_unique<ConstantBuffer>(sizeof(FluidConstant));
        if (!cb || !cb->IsValid())
        {
            return false;
        }
    }

    return true;
}

void FluidSystem::InitializeParticles(UINT particleCount)
{
    m_particles.clear();
    m_particles.resize(particleCount);
    m_instances.clear();
    m_instances.resize(particleCount);
    m_neighborForces.assign(particleCount, DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f));

    // 初期配置は境界内に均等配置し、軽い乱数でばらけさせる
    std::mt19937 rng{ 12345u };
    std::uniform_real_distribution<float> jitter(-0.02f, 0.02f);

    const float startX = m_bounds.min.x + m_particleRadius;
    const float startY = m_bounds.min.y + m_particleRadius;
    const float startZ = m_bounds.min.z + m_particleRadius;
    const float maxX = m_bounds.max.x - m_particleRadius;
    const float maxY = m_bounds.max.y - m_particleRadius;
    const float maxZ = m_bounds.max.z - m_particleRadius;

    const UINT perRow = static_cast<UINT>(std::cbrt(static_cast<float>(particleCount))) + 1;
    const float spacing = std::max(m_particleRadius * 2.2f, 0.05f);

    UINT index = 0;
    for (UINT y = 0; y < perRow && index < particleCount; ++y)
    {
        for (UINT z = 0; z < perRow && index < particleCount; ++z)
        {
            for (UINT x = 0; x < perRow && index < particleCount; ++x)
            {
                float px = std::min(startX + x * spacing, maxX);
                float py = std::min(startY + y * spacing, maxY);
                float pz = std::min(startZ + z * spacing, maxZ);

                px += jitter(rng);
                py += jitter(rng);
                pz += jitter(rng);

                Particle particle{};
                particle.position = XMFLOAT3(px, py, pz);
                particle.velocity = XMFLOAT3(0.0f, 0.0f, 0.0f);
                m_particles[index] = particle;

                m_instances[index].position = particle.position;
                m_instances[index].radius = m_supportRadius;
                ++index;
            }
        }
    }
}

void FluidSystem::Update(float deltaTime)
{
    const float dt = std::clamp(deltaTime, 0.0f, 0.033f); // 極端なデルタタイムを抑制
    const float gravityStep = kGravity * dt;
    const float dragFactor = std::clamp(1.0f - m_drag * dt, 0.0f, 1.0f);

    // 近傍検索バッファを初期化し、粒子間の押し戻し量を算出
    m_grid.Clear();
    if (m_neighborForces.size() != m_particles.size())
    {
        m_neighborForces.assign(m_particles.size(), XMFLOAT3(0.0f, 0.0f, 0.0f));
    }
    for (size_t i = 0; i < m_particles.size(); ++i)
    {
        m_grid.Insert(i, m_particles[i].position); // グリッドでセル分割し近傍候補を限定することで計算負荷を抑える
        m_neighborForces[i] = XMFLOAT3(0.0f, 0.0f, 0.0f);
    }

    for (size_t i = 0; i < m_particles.size(); ++i)
    {
        const auto& particle = m_particles[i];
        m_grid.Query(particle.position, m_supportRadius, m_neighborIndices);

        XMVECTOR correction = XMVectorZero();
        const XMVECTOR selfPos = XMLoadFloat3(&particle.position);
        for (size_t neighborIndex : m_neighborIndices)
        {
            if (neighborIndex == i)
            {
                continue;
            }

            const XMVECTOR neighborPos = XMLoadFloat3(&m_particles[neighborIndex].position);
            const XMVECTOR diff = XMVectorSubtract(selfPos, neighborPos);
            const float distSq = XMVectorGetX(XMVector3LengthSq(diff));
            if (distSq < 1e-8f)
            {
                continue;
            }

            const float dist = std::sqrt(distSq);
            if (dist >= m_supportRadius)
            {
                continue;
            }

            const float weight = (m_supportRadius - dist) / m_supportRadius;
            const float strength = weight * weight; // 滑らかに減衰させる
            const XMVECTOR dir = XMVectorScale(diff, 1.0f / dist);
            correction = XMVectorAdd(correction, XMVectorScale(dir, strength)); // セル内の粒子数が多いときはこの内積計算が処理負荷の主因
        }

        XMStoreFloat3(&m_neighborForces[i], correction);
    }

    for (size_t i = 0; i < m_particles.size(); ++i)
    {
        auto& particle = m_particles[i];

        particle.velocity.y += gravityStep;

        particle.velocity.x += m_neighborForces[i].x * m_interactionStrength * dt;
        particle.velocity.y += m_neighborForces[i].y * m_interactionStrength * dt;
        particle.velocity.z += m_neighborForces[i].z * m_interactionStrength * dt;

        particle.velocity.x *= dragFactor;
        particle.velocity.y *= dragFactor;
        particle.velocity.z *= dragFactor;

        particle.velocity.x = std::clamp(particle.velocity.x, -m_maxVelocity, m_maxVelocity);
        particle.velocity.y = std::clamp(particle.velocity.y, -m_maxVelocity, m_maxVelocity);
        particle.velocity.z = std::clamp(particle.velocity.z, -m_maxVelocity, m_maxVelocity);

        particle.position.x += particle.velocity.x * dt;
        particle.position.y += particle.velocity.y * dt;
        particle.position.z += particle.velocity.z * dt;

        ResolveCollisions(particle);
    }

    UpdateInstanceBuffer();
    UpdateGridLines();
}

void FluidSystem::ResolveCollisions(Particle& particle) const
{
    const float minX = m_bounds.min.x + m_particleRadius + kWallThickness;
    const float maxX = m_bounds.max.x - m_particleRadius - kWallThickness;
    const float minY = m_bounds.min.y + m_particleRadius + kWallThickness;
    const float maxY = m_bounds.max.y - m_particleRadius - kWallThickness;
    const float minZ = m_bounds.min.z + m_particleRadius + kWallThickness;
    const float maxZ = m_bounds.max.z - m_particleRadius - kWallThickness;

    if (particle.position.x < minX)
    {
        particle.position.x = minX;
        if (particle.velocity.x < 0.0f)
        {
            particle.velocity.x *= -m_restitution;
        }
    }
    else if (particle.position.x > maxX)
    {
        particle.position.x = maxX;
        if (particle.velocity.x > 0.0f)
        {
            particle.velocity.x *= -m_restitution;
        }
    }

    if (particle.position.y < minY)
    {
        particle.position.y = minY;
        if (particle.velocity.y < 0.0f)
        {
            particle.velocity.y *= -m_restitution;
        }
    }
    else if (particle.position.y > maxY)
    {
        particle.position.y = maxY;
        if (particle.velocity.y > 0.0f)
        {
            particle.velocity.y *= -m_restitution;
        }
    }

    if (particle.position.z < minZ)
    {
        particle.position.z = minZ;
        if (particle.velocity.z < 0.0f)
        {
            particle.velocity.z *= -m_restitution;
        }
    }
    else if (particle.position.z > maxZ)
    {
        particle.position.z = maxZ;
        if (particle.velocity.z > 0.0f)
        {
            particle.velocity.z *= -m_restitution;
        }
    }
}

void FluidSystem::UpdateInstanceBuffer()
{
    for (size_t i = 0; i < m_particles.size(); ++i)
    {
        m_instances[i].position = m_particles[i].position;
        m_instances[i].radius = m_supportRadius; // 影響範囲の大きさで描画する
    }

    if (!m_instanceBuffer)
    {
        return;
    }

    void* mapped = nullptr;
    auto resource = m_instanceBuffer->GetResource();
    if (resource && SUCCEEDED(resource->Map(0, nullptr, &mapped)))
    {
        memcpy(mapped, m_instances.data(), sizeof(ParticleInstance) * m_instances.size());
        resource->Unmap(0, nullptr);
    }
}

void FluidSystem::UpdateGridLines()
{
    m_gridLineVertices.clear();
    m_gridLineVertexCount = 0;

    std::vector<XMFLOAT3> cellMins;
    m_grid.CollectActiveCellMins(cellMins);
    if (cellMins.empty())
    {
        return;
    }

    const float cellSize = m_grid.CellSize();
    m_gridLineVertices.reserve(cellMins.size() * 24);

    auto addEdge = [this](const XMFLOAT3& a, const XMFLOAT3& b)
    {
        GridLineVertex v0{ a };
        GridLineVertex v1{ b };
        m_gridLineVertices.push_back(v0);
        m_gridLineVertices.push_back(v1);
    };

    for (const auto& minCorner : cellMins)
    {
        const XMFLOAT3 maxCorner{
            minCorner.x + cellSize,
            minCorner.y + cellSize,
            minCorner.z + cellSize };

        const std::array<XMFLOAT3, 8> corners =
        {
            XMFLOAT3{ minCorner.x, minCorner.y, minCorner.z },
            XMFLOAT3{ maxCorner.x, minCorner.y, minCorner.z },
            XMFLOAT3{ maxCorner.x, maxCorner.y, minCorner.z },
            XMFLOAT3{ minCorner.x, maxCorner.y, minCorner.z },
            XMFLOAT3{ minCorner.x, minCorner.y, maxCorner.z },
            XMFLOAT3{ maxCorner.x, minCorner.y, maxCorner.z },
            XMFLOAT3{ maxCorner.x, maxCorner.y, maxCorner.z },
            XMFLOAT3{ minCorner.x, maxCorner.y, maxCorner.z }
        };

        // 下面
        addEdge(corners[0], corners[1]);
        addEdge(corners[1], corners[2]);
        addEdge(corners[2], corners[3]);
        addEdge(corners[3], corners[0]);

        // 上面
        addEdge(corners[4], corners[5]);
        addEdge(corners[5], corners[6]);
        addEdge(corners[6], corners[7]);
        addEdge(corners[7], corners[4]);

        // 側面
        addEdge(corners[0], corners[4]);
        addEdge(corners[1], corners[5]);
        addEdge(corners[2], corners[6]);
        addEdge(corners[3], corners[7]);
    }

    m_gridLineVertexCount = static_cast<UINT>(m_gridLineVertices.size());
    if (m_gridLineVertexCount == 0)
    {
        return;
    }

    const size_t requiredVertices = m_gridLineVertices.size();
    if (!m_gridLineBuffer || requiredVertices > m_gridLineCapacity)
    {
        size_t bufferSize = sizeof(GridLineVertex) * requiredVertices;
        m_gridLineBuffer = std::make_unique<VertexBuffer>(bufferSize, sizeof(GridLineVertex), m_gridLineVertices.data());
        if (!m_gridLineBuffer || !m_gridLineBuffer->IsValid())
        {
            m_gridLineBuffer.reset();
            m_gridLineCapacity = 0;
            m_gridLineVertexCount = 0;
            return;
        }
        m_gridLineCapacity = requiredVertices;
        return;
    }

    void* mapped = nullptr;
    auto resource = m_gridLineBuffer->GetResource();
    if (resource && SUCCEEDED(resource->Map(0, nullptr, &mapped)))
    {
        memcpy(mapped, m_gridLineVertices.data(), sizeof(GridLineVertex) * requiredVertices);
        resource->Unmap(0, nullptr);
    }
}

void FluidSystem::Draw(ID3D12GraphicsCommandList* cmd, const Camera& camera)
{
    if (!cmd || !m_instanceBuffer || !m_pipelineState || !m_rootSignature)
    {
        return;
    }

    UINT frameIndex = g_Engine->CurrentBackBufferIndex();
    auto& cb = m_constantBuffers[frameIndex];
    if (!cb)
    {
        return;
    }

    FluidConstant* constant = cb->GetPtr<FluidConstant>();
    constant->World = m_world;
    constant->View = camera.GetViewMatrix();
    constant->Proj = camera.GetProjMatrix();

    cmd->SetGraphicsRootSignature(m_rootSignature->Get());
    cmd->SetGraphicsRootConstantBufferView(0, cb->GetAddress());

    if (m_sphereVertexBuffer && m_sphereIndexBuffer && m_indexCount > 0)
    {
        cmd->SetPipelineState(m_pipelineState->Get());
        auto vbViews = std::array<D3D12_VERTEX_BUFFER_VIEW, 2>{ m_sphereVertexBuffer->View(), m_instanceBuffer->View() };
        auto ibView = m_sphereIndexBuffer->View();
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd->IASetVertexBuffers(0, static_cast<UINT>(vbViews.size()), vbViews.data());
        cmd->IASetIndexBuffer(&ibView);
        cmd->DrawIndexedInstanced(m_indexCount, static_cast<UINT>(m_particles.size()), 0, 0, 0);
    }

    if (m_gridPipelineState && m_gridPipelineState->IsValid() && m_gridLineBuffer && m_gridLineVertexCount > 0)
    {
        cmd->SetPipelineState(m_gridPipelineState->Get());
        auto gridVb = m_gridLineBuffer->View();
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_LINELIST);
        cmd->IASetVertexBuffers(0, 1, &gridVb);
        cmd->DrawInstanced(m_gridLineVertexCount, 1, 0, 0);
    }
}

void FluidSystem::SetBounds(const Bounds& bounds)
{
    m_bounds = bounds;
    for (auto& particle : m_particles)
    {
        ResolveCollisions(particle);
    }
    UpdateInstanceBuffer();
    UpdateGridLines();
}

void FluidSystem::AdjustWall(const XMFLOAT3& direction, float amount)
{
    if (std::fabs(amount) < 1e-6f)
    {
        return;
    }

    const float absX = std::fabs(direction.x);
    const float absZ = std::fabs(direction.z);
    if (absX < 1e-4f && absZ < 1e-4f)
    {
        return;
    }

    const float minExtent = m_particleRadius * 4.0f; // 壁間の最小距離を確保

    if (absX >= absZ)
    {
        if (direction.x >= 0.0f)
        {
            float newMax = m_bounds.max.x - amount;
            newMax = std::max(newMax, m_bounds.min.x + minExtent);
            m_bounds.max.x = newMax;
        }
        else
        {
            float newMin = m_bounds.min.x + amount;
            newMin = std::min(newMin, m_bounds.max.x - minExtent);
            m_bounds.min.x = newMin;
        }
    }
    else
    {
        if (direction.z >= 0.0f)
        {
            float newMax = m_bounds.max.z - amount;
            newMax = std::max(newMax, m_bounds.min.z + minExtent);
            m_bounds.max.z = newMax;
        }
        else
        {
            float newMin = m_bounds.min.z + amount;
            newMin = std::min(newMin, m_bounds.max.z - minExtent);
            m_bounds.min.z = newMin;
        }
    }

    for (auto& particle : m_particles)
    {
        ResolveCollisions(particle);
    }
    UpdateInstanceBuffer();
    UpdateGridLines();
}
