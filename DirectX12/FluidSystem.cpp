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
{
    // グリッド間隔は粒子の影響半径に合わせて初期化し、MLS-MPM 用の節点を確保する準備を行う
    m_gridSpacing = std::max(m_supportRadius, 0.01f);
}

bool FluidSystem::Init(ID3D12Device* device, const Bounds& initialBounds, UINT particleCount)
{
    (void)device; // DirectXリソース生成は Engine 内のヘルパーを利用するため未使用

    m_bounds = initialBounds;
    InitializeParticles(particleCount);
    InitializeGrid();

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
                m_instances[index].radius = m_supportRadius / std::max(m_ssfrResolutionScale, 1e-3f);
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

    EnsureGridReady();
    ClearGridNodes();

    const float invSpacing = 1.0f / std::max(m_gridSpacing, 1e-4f);
    const int maxX = std::max(m_gridResolution.x - 1, 0);
    const int maxY = std::max(m_gridResolution.y - 1, 0);
    const int maxZ = std::max(m_gridResolution.z - 1, 0);

    // 粒子からグリッド節点へ速度と質量を散布（近傍探索の代わりに MLS-MPM を利用する）
    for (const auto& particle : m_particles)
    {
        const float fx = std::clamp((particle.position.x - m_gridOrigin.x) * invSpacing, 0.0f, static_cast<float>(maxX));
        const float fy = std::clamp((particle.position.y - m_gridOrigin.y) * invSpacing, 0.0f, static_cast<float>(maxY));
        const float fz = std::clamp((particle.position.z - m_gridOrigin.z) * invSpacing, 0.0f, static_cast<float>(maxZ));

        const int baseX = std::clamp(static_cast<int>(std::floor(fx)), 0, maxX);
        const int baseY = std::clamp(static_cast<int>(std::floor(fy)), 0, maxY);
        const int baseZ = std::clamp(static_cast<int>(std::floor(fz)), 0, maxZ);

        const float fracX = std::clamp(fx - baseX, 0.0f, 1.0f);
        const float fracY = std::clamp(fy - baseY, 0.0f, 1.0f);
        const float fracZ = std::clamp(fz - baseZ, 0.0f, 1.0f);

        const int nodeX[2] = { baseX, std::min(baseX + 1, maxX) };
        const int nodeY[2] = { baseY, std::min(baseY + 1, maxY) };
        const int nodeZ[2] = { baseZ, std::min(baseZ + 1, maxZ) };
        const float weightX[2] = { 1.0f - fracX, fracX };
        const float weightY[2] = { 1.0f - fracY, fracY };
        const float weightZ[2] = { 1.0f - fracZ, fracZ };

        for (int ix = 0; ix < 2; ++ix)
        {
            for (int iy = 0; iy < 2; ++iy)
            {
                for (int iz = 0; iz < 2; ++iz)
                {
                    const float weight = weightX[ix] * weightY[iy] * weightZ[iz];
                    if (weight <= 0.0f)
                    {
                        continue;
                    }

                    auto& node = m_gridNodes[GridIndex(nodeX[ix], nodeY[iy], nodeZ[iz])];
                    node.mass += weight;
                    node.velocity.x += particle.velocity.x * weight;
                    node.velocity.y += particle.velocity.y * weight;
                    node.velocity.z += particle.velocity.z * weight;
                }
            }
        }
    }

    for (auto& node : m_gridNodes)
    {
        if (node.mass <= 1e-6f)
        {
            node.pressure = 0.0f;
            node.pressureGradient = XMFLOAT3(0.0f, 0.0f, 0.0f);
            continue;
        }

        const float invMass = 1.0f / node.mass;
        node.velocity.x = node.velocity.x * invMass;
        node.velocity.y = (node.velocity.y * invMass) + gravityStep;
        node.velocity.z = node.velocity.z * invMass;

        // 粒子が密集し過ぎた節点に圧力を設定し、後段で押し戻す力を計算する
        const float density = node.mass;
        const float compression = std::max(density - m_restDensity, 0.0f);
        node.pressure = compression * m_pressureStiffness;
    }

    const float safeSpacing = std::max(m_gridSpacing, 1e-4f);
    auto samplePressure = [this](int x, int y, int z)
    {
        return m_gridNodes[GridIndex(x, y, z)].pressure;
    };

    for (int z = 0; z < m_gridResolution.z; ++z)
    {
        for (int y = 0; y < m_gridResolution.y; ++y)
        {
            for (int x = 0; x < m_gridResolution.x; ++x)
            {
                auto& node = m_gridNodes[GridIndex(x, y, z)];
                if (node.mass <= 1e-6f)
                {
                    node.pressureGradient = XMFLOAT3(0.0f, 0.0f, 0.0f);
                    continue;
                }

                const float pxPlus = samplePressure(x + 1, y, z);
                const float pxMinus = samplePressure(x - 1, y, z);
                const float pyPlus = samplePressure(x, y + 1, z);
                const float pyMinus = samplePressure(x, y - 1, z);
                const float pzPlus = samplePressure(x, y, z + 1);
                const float pzMinus = samplePressure(x, y, z - 1);

                const float gradScale = 0.5f / safeSpacing;
                node.pressureGradient = XMFLOAT3(
                    (pxPlus - pxMinus) * gradScale,
                    (pyPlus - pyMinus) * gradScale,
                    (pzPlus - pzMinus) * gradScale);
            }
        }
    }

    for (auto& particle : m_particles)
    {
        const float fx = std::clamp((particle.position.x - m_gridOrigin.x) * invSpacing, 0.0f, static_cast<float>(maxX));
        const float fy = std::clamp((particle.position.y - m_gridOrigin.y) * invSpacing, 0.0f, static_cast<float>(maxY));
        const float fz = std::clamp((particle.position.z - m_gridOrigin.z) * invSpacing, 0.0f, static_cast<float>(maxZ));

        const int baseX = std::clamp(static_cast<int>(std::floor(fx)), 0, maxX);
        const int baseY = std::clamp(static_cast<int>(std::floor(fy)), 0, maxY);
        const int baseZ = std::clamp(static_cast<int>(std::floor(fz)), 0, maxZ);

        const float fracX = std::clamp(fx - baseX, 0.0f, 1.0f);
        const float fracY = std::clamp(fy - baseY, 0.0f, 1.0f);
        const float fracZ = std::clamp(fz - baseZ, 0.0f, 1.0f);

        const int nodeX[2] = { baseX, std::min(baseX + 1, maxX) };
        const int nodeY[2] = { baseY, std::min(baseY + 1, maxY) };
        const int nodeZ[2] = { baseZ, std::min(baseZ + 1, maxZ) };
        const float weightX[2] = { 1.0f - fracX, fracX };
        const float weightY[2] = { 1.0f - fracY, fracY };
        const float weightZ[2] = { 1.0f - fracZ, fracZ };

        XMFLOAT3 newVelocity{ 0.0f, 0.0f, 0.0f };
        XMFLOAT3 pressureGradient{ 0.0f, 0.0f, 0.0f };
        float density = 0.0f;

        for (int ix = 0; ix < 2; ++ix)
        {
            for (int iy = 0; iy < 2; ++iy)
            {
                for (int iz = 0; iz < 2; ++iz)
                {
                    const float weight = weightX[ix] * weightY[iy] * weightZ[iz];
                    if (weight <= 0.0f)
                    {
                        continue;
                    }

                    const auto& node = m_gridNodes[GridIndex(nodeX[ix], nodeY[iy], nodeZ[iz])];
                    if (node.mass <= 1e-6f)
                    {
                        continue;
                    }

                    newVelocity.x += node.velocity.x * weight;
                    newVelocity.y += node.velocity.y * weight;
                    newVelocity.z += node.velocity.z * weight;
                    pressureGradient.x += node.pressureGradient.x * weight;
                    pressureGradient.y += node.pressureGradient.y * weight;
                    pressureGradient.z += node.pressureGradient.z * weight;
                    density += node.mass * weight;
                }
            }
        }

        if (density > 1e-4f)
        {
            const float invDensity = 1.0f / density;
            // 圧力勾配に基づく反力を速度へ加算して粒子同士の押し戻しを再現する
            newVelocity.x += (-pressureGradient.x * invDensity) * dt;
            newVelocity.y += (-pressureGradient.y * invDensity) * dt;
            newVelocity.z += (-pressureGradient.z * invDensity) * dt;
        }

        particle.velocity.x = std::clamp(newVelocity.x * dragFactor, -m_maxVelocity, m_maxVelocity);
        particle.velocity.y = std::clamp(newVelocity.y * dragFactor, -m_maxVelocity, m_maxVelocity);
        particle.velocity.z = std::clamp(newVelocity.z * dragFactor, -m_maxVelocity, m_maxVelocity);

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
        // SSFR を半解像度で回すため、レンダリング時にカバー範囲を広げてピクセル欠けを抑える
        m_instances[i].radius = m_supportRadius / std::max(m_ssfrResolutionScale, 1e-3f);
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

void FluidSystem::InitializeGrid()
{
    // 粒子の境界を元に MLS-MPM 用のグリッド解像度を決定し、近傍探索を省いた構造へ移行する
    m_gridSpacing = std::max(m_supportRadius, 0.01f);
    const float extentX = std::max(m_bounds.max.x - m_bounds.min.x, m_gridSpacing);
    const float extentY = std::max(m_bounds.max.y - m_bounds.min.y, m_gridSpacing);
    const float extentZ = std::max(m_bounds.max.z - m_bounds.min.z, m_gridSpacing);

    const int cellsX = std::max(static_cast<int>(std::ceil(extentX / m_gridSpacing)) + 1, 1);
    const int cellsY = std::max(static_cast<int>(std::ceil(extentY / m_gridSpacing)) + 1, 1);
    const int cellsZ = std::max(static_cast<int>(std::ceil(extentZ / m_gridSpacing)) + 1, 1);

    m_gridResolution = XMINT3(cellsX, cellsY, cellsZ);
    m_gridOrigin = m_bounds.min;
    m_gridNodes.assign(
        static_cast<size_t>(cellsX) * static_cast<size_t>(cellsY) * static_cast<size_t>(cellsZ),
        GridNode{ XMFLOAT3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f, XMFLOAT3(0.0f, 0.0f, 0.0f) });
}

void FluidSystem::ClearGridNodes()
{
    // 毎フレーム節点の質量と速度をクリアし、MLS-MPM の再計算に備える
    for (auto& node : m_gridNodes)
    {
        node.velocity = XMFLOAT3(0.0f, 0.0f, 0.0f);
        node.mass = 0.0f;
        node.pressure = 0.0f;
        node.pressureGradient = XMFLOAT3(0.0f, 0.0f, 0.0f);
    }
}

size_t FluidSystem::GridIndex(int x, int y, int z) const
{
    const int clampedX = std::clamp(x, 0, std::max(m_gridResolution.x - 1, 0));
    const int clampedY = std::clamp(y, 0, std::max(m_gridResolution.y - 1, 0));
    const int clampedZ = std::clamp(z, 0, std::max(m_gridResolution.z - 1, 0));

    return (static_cast<size_t>(clampedZ) * static_cast<size_t>(m_gridResolution.y) + static_cast<size_t>(clampedY)) * static_cast<size_t>(m_gridResolution.x) + static_cast<size_t>(clampedX);
}

void FluidSystem::EnsureGridReady()
{
    if (m_gridNodes.empty())
    {
        InitializeGrid();
    }
}

void FluidSystem::UpdateGridLines()
{
    // 近傍探索グリッドを廃止したため、節点表示は無効化しておく（描画負荷と齟齬の回避が目的）。
    m_gridLineVertices.clear();
    m_gridLineVertexCount = 0;
    m_gridLineCapacity = 0;
    m_gridLineBuffer.reset();
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
    InitializeGrid();
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
