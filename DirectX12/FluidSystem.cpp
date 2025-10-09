#include "FluidSystem.h"
#include "Camera.h"
#include "Engine.h"
#include "SphereMeshGenerator.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>
#include <Windows.h>
#include <d3dcompiler.h>
#include <d3dx12.h>

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

    constexpr D3D12_INPUT_ELEMENT_DESC kPointInputElements[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32_FLOAT,        0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };

    constexpr D3D12_INPUT_ELEMENT_DESC kMarchingInputElements[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };
}


FluidSystem::FluidSystem()
    : m_world(XMMatrixIdentity())
{
    // グリッド間隔は粒子の影響半径に合わせて初期化し、MLS-MPM 用の節点を確保する準備を行う
    m_gridSpacing = std::max(m_supportRadius, 0.01f);
}

bool FluidSystem::Init(ID3D12Device* device, const Bounds& initialBounds, UINT particleCount, RenderMode renderMode)
{
    (void)device; // DirectXリソース生成は Engine 内のヘルパーを利用するため未使用

    m_bounds = initialBounds;
    m_renderMode = renderMode;
    m_useSSFR = (m_renderMode == RenderMode::SSFR);
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

    m_pointPipelineState = std::make_unique<PipelineState>();
    if (!m_pointPipelineState)
    {
        return false;
    }
    D3D12_INPUT_LAYOUT_DESC pointLayout{};
    pointLayout.pInputElementDescs = kPointInputElements;
    pointLayout.NumElements = _countof(kPointInputElements);
    m_pointPipelineState->SetInputLayout(pointLayout);
    m_pointPipelineState->SetRootSignature(m_rootSignature->Get());
    m_pointPipelineState->SetVS(L"ParticlePointVS.cso");
    m_pointPipelineState->SetPS(L"ParticlePointPS.cso");
    m_pointPipelineState->SetDepthStencilFormat(DXGI_FORMAT_D32_FLOAT);
    m_pointPipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT);
    if (!m_pointPipelineState->IsValid())
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

    if (auto* heap = g_Engine->CbvSrvUavHeap())
    {
        // ※粒子インスタンス情報を SRV として参照できるよう登録（SSFR で StructuredBuffer を読むため）
        m_descriptorIncrement = g_Engine->Device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        if (!m_particleBufferSrv)
        {
            m_particleBufferSrv = heap->RegisterBuffer(
                m_instanceBuffer->GetResource(),
                static_cast<UINT>(m_instances.size()),
                sizeof(ParticleInstance));
            if (!m_particleBufferSrv)
            {
                return false;
            }
        }
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

    m_marchingPipelineState = std::make_unique<PipelineState>();
    if (!m_marchingPipelineState)
    {
        return false;
    }
    D3D12_INPUT_LAYOUT_DESC marchingLayout{};
    marchingLayout.pInputElementDescs = kMarchingInputElements;
    marchingLayout.NumElements = _countof(kMarchingInputElements);
    m_marchingPipelineState->SetInputLayout(marchingLayout);
    m_marchingPipelineState->SetRootSignature(m_rootSignature->Get());
    m_marchingPipelineState->SetVS(L"MarchingCubesVS.cso");
    m_marchingPipelineState->SetPS(L"MarchingCubesPS.cso");
    m_marchingPipelineState->SetDepthStencilFormat(DXGI_FORMAT_D32_FLOAT);
    m_marchingPipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    if (!m_marchingPipelineState->IsValid())
    {
        return false;
    }

    if (m_useSSFR && !EnsureSSFRResources())
    {
        return false; // ※SSFR 初期化に失敗した場合は描画へ進めない
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
    GenerateMarchingCubesMesh(); // マーチングキューブ用メッシュを更新して粒子表面を再構築する
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

float FluidSystem::SampleGridDensity(const XMFLOAT3& position) const
{
    if (m_gridNodes.empty())
    {
        return 0.0f;
    }

    const float safeSpacing = std::max(m_gridSpacing, 1e-4f);
    const float invSpacing = 1.0f / safeSpacing;
    const int maxX = std::max(m_gridResolution.x - 1, 0);
    const int maxY = std::max(m_gridResolution.y - 1, 0);
    const int maxZ = std::max(m_gridResolution.z - 1, 0);

    const float fx = std::clamp((position.x - m_gridOrigin.x) * invSpacing, 0.0f, static_cast<float>(maxX));
    const float fy = std::clamp((position.y - m_gridOrigin.y) * invSpacing, 0.0f, static_cast<float>(maxY));
    const float fz = std::clamp((position.z - m_gridOrigin.z) * invSpacing, 0.0f, static_cast<float>(maxZ));

    const int x0 = std::clamp(static_cast<int>(std::floor(fx)), 0, maxX);
    const int y0 = std::clamp(static_cast<int>(std::floor(fy)), 0, maxY);
    const int z0 = std::clamp(static_cast<int>(std::floor(fz)), 0, maxZ);
    const int x1 = std::min(x0 + 1, maxX);
    const int y1 = std::min(y0 + 1, maxY);
    const int z1 = std::min(z0 + 1, maxZ);

    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const float tz = fz - static_cast<float>(z0);

    auto density = [&](int x, int y, int z)
    {
        return m_gridNodes[GridIndex(x, y, z)].mass;
    };

    const float c000 = density(x0, y0, z0);
    const float c100 = density(x1, y0, z0);
    const float c010 = density(x0, y1, z0);
    const float c110 = density(x1, y1, z0);
    const float c001 = density(x0, y0, z1);
    const float c101 = density(x1, y0, z1);
    const float c011 = density(x0, y1, z1);
    const float c111 = density(x1, y1, z1);

    const float c00 = c000 * (1.0f - tx) + c100 * tx;
    const float c10 = c010 * (1.0f - tx) + c110 * tx;
    const float c01 = c001 * (1.0f - tx) + c101 * tx;
    const float c11 = c011 * (1.0f - tx) + c111 * tx;

    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;

    return c0 * (1.0f - tz) + c1 * tz;
}

XMFLOAT3 FluidSystem::SampleGridGradient(const XMFLOAT3& position) const
{
    const float safeSpacing = std::max(m_gridSpacing, 1e-3f);
    const float h = safeSpacing * 0.5f;

    const float dx = SampleGridDensity(XMFLOAT3(position.x + h, position.y, position.z)) -
        SampleGridDensity(XMFLOAT3(position.x - h, position.y, position.z));
    const float dy = SampleGridDensity(XMFLOAT3(position.x, position.y + h, position.z)) -
        SampleGridDensity(XMFLOAT3(position.x, position.y - h, position.z));
    const float dz = SampleGridDensity(XMFLOAT3(position.x, position.y, position.z + h)) -
        SampleGridDensity(XMFLOAT3(position.x, position.y, position.z - h));

    XMVECTOR grad = XMVectorSet(dx, dy, dz, 0.0f);
    grad = XMVector3Normalize(grad);

    XMFLOAT3 normal{};
    XMStoreFloat3(&normal, grad);
    if (std::isnan(normal.x) || std::isnan(normal.y) || std::isnan(normal.z))
    {
        normal = XMFLOAT3(0.0f, 1.0f, 0.0f);
    }
    return normal;
}

void FluidSystem::UpdateMarchingBuffers()
{
    m_marchingIndexCount = static_cast<UINT>(m_marchingIndices.size());
    const size_t vbSize = m_marchingVertices.size() * sizeof(MarchingVertex);
    const size_t ibSize = m_marchingIndices.size() * sizeof(uint32_t);

    if (vbSize == 0 || ibSize == 0)
    {
        m_marchingVertexBuffer.reset();
        m_marchingIndexBuffer.reset();
        m_marchingVertexCapacity = 0;
        m_marchingIndexCapacity = 0;
        return;
    }

    if (!m_marchingVertexBuffer || vbSize > m_marchingVertexCapacity)
    {
        m_marchingVertexCapacity = vbSize;
        m_marchingVertexBuffer = std::make_unique<VertexBuffer>(vbSize, sizeof(MarchingVertex), m_marchingVertices.data());
        if (!m_marchingVertexBuffer || !m_marchingVertexBuffer->IsValid())
        {
            return;
        }
    }
    else
    {
        void* mapped = nullptr;
        if (auto* resource = m_marchingVertexBuffer->GetResource(); resource && SUCCEEDED(resource->Map(0, nullptr, &mapped)))
        {
            memcpy(mapped, m_marchingVertices.data(), vbSize);
            resource->Unmap(0, nullptr);
        }
    }

    m_marchingIndexCapacity = ibSize;
    m_marchingIndexBuffer = std::make_unique<IndexBuffer>(ibSize, m_marchingIndices.data());
    if (!m_marchingIndexBuffer || !m_marchingIndexBuffer->IsValid())
    {
        return;
    }
}

void FluidSystem::GenerateMarchingCubesMesh()
{
    if (m_renderMode != RenderMode::MarchingCubes)
    {
        return;
    }

    m_marchingVertices.clear();
    m_marchingIndices.clear();
    m_marchingIndexCount = 0;

    if (m_gridResolution.x < 2 || m_gridResolution.y < 2 || m_gridResolution.z < 2)
    {
        UpdateMarchingBuffers();
        return;
    }

    const float isoLevel = std::max(m_restDensity * 0.5f, 0.05f); // 静止密度の半分を閾値にしてノイズを抑制
    const float spacing = std::max(m_gridSpacing, 1e-4f);

    // 立方体を 6 個のテトラへ分割し、簡易テーブルでマーチングキューブ相当の面生成を行う
    static constexpr int tetrahedra[6][4] =
    {
        { 0, 5, 1, 6 },
        { 0, 1, 2, 6 },
        { 0, 2, 3, 6 },
        { 0, 3, 7, 6 },
        { 0, 7, 4, 6 },
        { 0, 4, 5, 6 },
    };

    static constexpr int tetraEdgeCorners[6][2] =
    {
        { 0, 1 },
        { 1, 2 },
        { 2, 0 },
        { 0, 3 },
        { 1, 3 },
        { 2, 3 },
    };

    static constexpr int tetraTriTable[16][7] =
    {
        { -1, -1, -1, -1, -1, -1, -1 },
        { 0, 3, 2, -1, -1, -1, -1 },
        { 0, 1, 4, -1, -1, -1, -1 },
        { 1, 4, 2, 2, 4, 3, -1 },
        { 1, 2, 5, -1, -1, -1, -1 },
        { 0, 3, 5, 0, 5, 1, -1 },
        { 0, 2, 5, 0, 5, 4, -1 },
        { 5, 4, 3, -1, -1, -1, -1 },
        { 5, 4, 3, -1, -1, -1, -1 },
        { 0, 5, 4, 0, 2, 5, -1 },
        { 1, 5, 3, 1, 3, 0, -1 },
        { 5, 2, 1, -1, -1, -1, -1 },
        { 1, 4, 3, 1, 3, 2, -1 },
        { 0, 4, 1, -1, -1, -1, -1 },
        { 0, 3, 2, -1, -1, -1, -1 },
        { -1, -1, -1, -1, -1, -1, -1 },
    };

    const int cellsX = m_gridResolution.x - 1;
    const int cellsY = m_gridResolution.y - 1;
    const int cellsZ = m_gridResolution.z - 1;

    for (int z = 0; z < cellsZ; ++z)
    {
        for (int y = 0; y < cellsY; ++y)
        {
            for (int x = 0; x < cellsX; ++x)
            {
                XMFLOAT3 cornerPos[8];
                float cornerValue[8];

                for (int i = 0; i < 8; ++i)
                {
                    const int dx = (i & 1) ? 1 : 0;
                    const int dy = (i & 2) ? 1 : 0;
                    const int dz = (i & 4) ? 1 : 0;

                    const float px = m_gridOrigin.x + (static_cast<float>(x + dx) * spacing);
                    const float py = m_gridOrigin.y + (static_cast<float>(y + dy) * spacing);
                    const float pz = m_gridOrigin.z + (static_cast<float>(z + dz) * spacing);

                    cornerPos[i] = XMFLOAT3(px, py, pz);
                    cornerValue[i] = m_gridNodes[GridIndex(x + dx, y + dy, z + dz)].mass;
                }

                for (const auto& tetra : tetrahedra)
                {
                    XMFLOAT3 tetraPos[4];
                    float tetraValue[4];
                    int caseIndex = 0;

                    for (int i = 0; i < 4; ++i)
                    {
                        const int cornerIdx = tetra[i];
                        tetraPos[i] = cornerPos[cornerIdx];
                        tetraValue[i] = cornerValue[cornerIdx];
                        if (tetraValue[i] > isoLevel)
                        {
                            caseIndex |= (1 << i);
                        }
                    }

                    const int* edges = tetraTriTable[caseIndex];
                    for (int e = 0; edges[e] != -1; e += 3)
                    {
                        MarchingVertex tri[3];
                        for (int j = 0; j < 3; ++j)
                        {
                            const int edge = edges[e + j];
                            const int ia = tetraEdgeCorners[edge][0];
                            const int ib = tetraEdgeCorners[edge][1];
                            const float va = tetraValue[ia];
                            const float vb = tetraValue[ib];
                            const float denom = vb - va;
                            float t = 0.5f;
                            if (std::fabs(denom) > 1e-6f)
                            {
                                t = std::clamp((isoLevel - va) / denom, 0.0f, 1.0f);
                            }

                            tri[j].position.x = tetraPos[ia].x + (tetraPos[ib].x - tetraPos[ia].x) * t;
                            tri[j].position.y = tetraPos[ia].y + (tetraPos[ib].y - tetraPos[ia].y) * t;
                            tri[j].position.z = tetraPos[ia].z + (tetraPos[ib].z - tetraPos[ia].z) * t;
                            tri[j].normal = SampleGridGradient(tri[j].position);
                        }

                        const uint32_t baseIndex = static_cast<uint32_t>(m_marchingVertices.size());
                        m_marchingVertices.push_back(tri[0]);
                        m_marchingVertices.push_back(tri[1]);
                        m_marchingVertices.push_back(tri[2]);

                        m_marchingIndices.push_back(baseIndex);
                        m_marchingIndices.push_back(baseIndex + 1);
                        m_marchingIndices.push_back(baseIndex + 2);
                    }
                }
            }
        }
    }

    UpdateMarchingBuffers();
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

namespace
{
    // ※シェーダーファイル探索を行い、CSO または HLSL のパスを解決するヘルパー
    std::filesystem::path ResolveShaderPath(const std::wstring& fileName)
    {
        std::vector<std::filesystem::path> candidates;
        candidates.push_back(std::filesystem::current_path());

        wchar_t exePath[MAX_PATH] = {};
        DWORD len = GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        if (len > 0 && len < MAX_PATH)
        {
            auto dir = std::filesystem::path(exePath).parent_path();
            for (int i = 0; i < 6 && !dir.empty(); ++i)
            {
                candidates.push_back(dir);
                dir = dir.parent_path();
            }
        }

        for (const auto& base : candidates)
        {
            auto path = base / fileName;
            if (std::filesystem::exists(path))
            {
                return path;
            }
        }
        return {};
    }

    bool LoadShaderBytecode(const std::wstring& fileName, const char* entryPoint, const char* target, Microsoft::WRL::ComPtr<ID3DBlob>& blob)
    {
        std::filesystem::path csoPath = ResolveShaderPath(fileName);
        HRESULT hr = E_FAIL;
        if (!csoPath.empty())
        {
            hr = D3DReadFileToBlob(csoPath.c_str(), blob.ReleaseAndGetAddressOf());
        }

        if (FAILED(hr))
        {
            std::wstring hlsl = fileName;
            size_t pos = hlsl.find_last_of(L'.');
            if (pos != std::wstring::npos)
            {
                hlsl.replace(pos, std::wstring::npos, L".hlsl");
            }

            std::filesystem::path hlslPath = ResolveShaderPath(hlsl);
            if (hlslPath.empty())
            {
                return false;
            }

            UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
            flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
            Microsoft::WRL::ComPtr<ID3DBlob> error;
            hr = D3DCompileFromFile(hlslPath.c_str(), nullptr, nullptr, entryPoint, target, flags, 0,
                blob.ReleaseAndGetAddressOf(), error.ReleaseAndGetAddressOf());
            if (FAILED(hr))
            {
                if (error)
                {
                    OutputDebugStringA(reinterpret_cast<const char*>(error->GetBufferPointer()));
                }
                return false;
            }
        }

        return SUCCEEDED(hr);
    }
}

bool FluidSystem::CreateSSFROnce()
{
    if (!m_useSSFR)
    {
        return true; // ※球メッシュ描画のみの場合は初期化不要
    }

    auto device = g_Engine->Device();
    if (!device)
    {
        return false;
    }

    if (!m_ssfrParticleRootSig || !m_ssfrCompositeRootSig || !m_ssfrComputeRootSig)
    {
        if (!CreateSSFRRootSignatures())
        {
            return false;
        }
    }

    if (!m_ssfrParticlePSO)
    {
        if (!CreateParticlePSO())
        {
            return false;
        }
    }

    if (!m_ssfrCompositePSO)
    {
        if (!CreateCompositePSO())
        {
            return false;
        }
    }

    if (!m_ssfrBilateralPSO || !m_ssfrNormalPSO)
    {
        if (!CreateComputePSO())
        {
            return false;
        }
    }

    for (auto& cb : m_ssfrConstantBuffers)
    {
        if (!cb)
        {
            cb = std::make_unique<ConstantBuffer>(sizeof(SSFRConstant));
            if (!cb || !cb->IsValid())
            {
                return false;
            }
        }
    }

    if (!m_ssfrCpuDescriptorHeap)
    {
        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        desc.NumDescriptors = 32; // ※深度・厚み・法線など複数 UAV をまとめて扱うため十分な数を確保
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        if (FAILED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(m_ssfrCpuDescriptorHeap.ReleaseAndGetAddressOf()))))
        {
            return false;
        }
        m_ssfrCpuDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_ssfrCpuDescriptorCursor = 0;
    }

    if (!m_ssfrRtvDescriptorHeap)
    {
        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        desc.NumDescriptors = 8; // ※MRT 用に深度・厚みの2枚＋予備を確保
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        if (FAILED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(m_ssfrRtvDescriptorHeap.ReleaseAndGetAddressOf()))))
        {
            return false;
        }
        m_ssfrRtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        m_ssfrRtvDescriptorCursor = 0;
    }

    return true;
}

bool FluidSystem::EnsureSSFRResources()
{
    if (!m_useSSFR)
    {
        return true;
    }

    if (!CreateSSFROnce())
    {
        return false;
    }

    UINT fbWidth = g_Engine->FrameBufferWidth();
    UINT fbHeight = g_Engine->FrameBufferHeight();
    if (fbWidth == 0 || fbHeight == 0)
    {
        return false;
    }

    UINT targetWidth = std::max(1u, static_cast<UINT>(std::ceil(static_cast<float>(fbWidth) * m_ssfrResolutionScale)));
    UINT targetHeight = std::max(1u, static_cast<UINT>(std::ceil(static_cast<float>(fbHeight) * m_ssfrResolutionScale)));

    if (targetWidth != m_ssfrWidth || targetHeight != m_ssfrHeight)
    {
        if (!ResizeSSFRTargets(targetWidth, targetHeight))
        {
            return false;
        }
    }

    return true;
}

D3D12_CPU_DESCRIPTOR_HANDLE FluidSystem::AllocateCpuDescriptor()
{
    D3D12_CPU_DESCRIPTOR_HANDLE handle{};
    if (!m_ssfrCpuDescriptorHeap)
    {
        handle.ptr = 0;
        return handle;
    }

    handle = m_ssfrCpuDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(m_ssfrCpuDescriptorCursor) * m_ssfrCpuDescriptorSize;
    ++m_ssfrCpuDescriptorCursor;
    return handle;
}

D3D12_CPU_DESCRIPTOR_HANDLE FluidSystem::AllocateRtvDescriptor()
{
    D3D12_CPU_DESCRIPTOR_HANDLE handle{};
    if (!m_ssfrRtvDescriptorHeap)
    {
        handle.ptr = 0;
        return handle;
    }

    handle = m_ssfrRtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(m_ssfrRtvDescriptorCursor) * m_ssfrRtvDescriptorSize;
    ++m_ssfrRtvDescriptorCursor;
    return handle;
}

bool FluidSystem::ResizeSSFRTargets(UINT width, UINT height)
{
    auto device = g_Engine->Device();
    auto* heap = g_Engine->CbvSrvUavHeap();
    if (!device || !heap)
    {
        return false;
    }

    CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_DEFAULT);

    auto createTexture = [&](SSFRTarget& target, DXGI_FORMAT format, bool createSrv, bool createUav, bool createRtv, D3D12_RESOURCE_STATES initialState, const float* clearColor)
    {
        if (width == 0 || height == 0)
        {
            return false;
        }

        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;
        if (createUav)
        {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }
        if (createRtv)
        {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        }

        D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Tex2D(
            format,
            width,
            height,
            1, 1, 1, 0,
            flags);

        D3D12_CLEAR_VALUE clearValue{};
        D3D12_CLEAR_VALUE* clearPtr = nullptr;
        if (createRtv && clearColor)
        {
            clearValue.Format = format;
            clearValue.Color[0] = clearColor[0];
            clearValue.Color[1] = clearColor[1];
            clearValue.Color[2] = clearColor[2];
            clearValue.Color[3] = clearColor[3];
            clearPtr = &clearValue;
        }

        if (FAILED(device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            initialState,
            clearPtr,
            IID_PPV_ARGS(target.resource.ReleaseAndGetAddressOf()))))
        {
            return false;
        }
        target.currentState = initialState;

        if (!createUav)
        {
            target.uavHandle = nullptr;
            target.uavCpuHandle.ptr = 0;
        }

        if (createUav)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
            uav.Format = format;
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            uav.Texture2D.MipSlice = 0;
            uav.Texture2D.PlaneSlice = 0;

            if (!target.uavHandle)
            {
                target.uavHandle = heap->RegisterTextureUAV(target.resource.Get(), format);
            }
            else
            {
                device->CreateUnorderedAccessView(target.resource.Get(), nullptr, &uav, target.uavHandle->HandleCPU);
            }

            if (target.uavCpuHandle.ptr == 0)
            {
                target.uavCpuHandle = AllocateCpuDescriptor();
            }
            device->CreateUnorderedAccessView(target.resource.Get(), nullptr, &uav, target.uavCpuHandle);
        }

        if (createRtv)
        {
            if (target.rtvHandle.ptr == 0)
            {
                target.rtvHandle = AllocateRtvDescriptor();
            }
            if (target.rtvHandle.ptr != 0)
            {
                device->CreateRenderTargetView(target.resource.Get(), nullptr, target.rtvHandle);
            }
        }

        if (createSrv)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = format;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Texture2D.MipLevels = 1;
            srv.Texture2D.MostDetailedMip = 0;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

            if (!target.srvHandle)
            {
                target.srvHandle = heap->Register(target.resource.Get(), srv);
            }
            else
            {
                device->CreateShaderResourceView(target.resource.Get(), &srv, target.srvHandle->HandleCPU);
            }
        }

        return true;
    };

    const float depthClear[4] = { 1000000.0f, 0.0f, 0.0f, 0.0f }; // ※MINブレンド用に十分大きな値で初期化し遠方値を表現
    if (!createTexture(m_rawDepth, DXGI_FORMAT_R32_FLOAT, true, false, true, D3D12_RESOURCE_STATE_RENDER_TARGET, depthClear))
    {
        return false;
    }

    if (!createTexture(m_smoothedDepth, DXGI_FORMAT_R32_FLOAT, true, true, false, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr))
    {
        return false;
    }

    if (!createTexture(m_normal, DXGI_FORMAT_R16G16B16A16_FLOAT, true, true, false, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr))
    {
        return false;
    }

    const float thicknessClear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    if (!createTexture(m_thickness, DXGI_FORMAT_R16_FLOAT, true, false, true, D3D12_RESOURCE_STATE_RENDER_TARGET, thicknessClear))
    {
        return false;
    }

    // ※深度バッファ SRV は DSV の実体を参照し直して最新に保つ
    ID3D12Resource* depthResource = g_Engine->DepthStencilBuffer();
    if (depthResource)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC depthSrv{};
        depthSrv.Format = DXGI_FORMAT_R32_FLOAT;
        depthSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        depthSrv.Texture2D.MostDetailedMip = 0;
        depthSrv.Texture2D.MipLevels = 1;
        depthSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

        if (!m_sceneDepthSrv)
        {
            m_sceneDepthSrv = heap->Register(depthResource, depthSrv);
        }
        else
        {
            device->CreateShaderResourceView(depthResource, &depthSrv, m_sceneDepthSrv->HandleCPU);
        }
        m_cachedSceneDepth = depthResource;
    }

    // ※背景カラー退避用テクスチャはフル解像度で確保する
    UINT fbWidth = g_Engine->FrameBufferWidth();
    UINT fbHeight = g_Engine->FrameBufferHeight();
    D3D12_RESOURCE_DESC colorDesc = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_R8G8B8A8_UNORM,
        fbWidth,
        fbHeight,
        1, 1, 1, 0,
        D3D12_RESOURCE_FLAG_NONE);

    if (FAILED(device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &colorDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(m_sceneColorCopy.resource.ReleaseAndGetAddressOf()))))
    {
        return false;
    }
    m_sceneColorCopy.currentState = D3D12_RESOURCE_STATE_COPY_DEST;

    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Texture2D.MostDetailedMip = 0;
        srv.Texture2D.MipLevels = 1;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

        if (!m_sceneColorCopy.srvHandle)
        {
            m_sceneColorCopy.srvHandle = heap->Register(m_sceneColorCopy.resource.Get(), srv);
        }
        else
        {
            device->CreateShaderResourceView(m_sceneColorCopy.resource.Get(), &srv, m_sceneColorCopy.srvHandle->HandleCPU);
        }
    }

    m_ssfrWidth = width;
    m_ssfrHeight = height;

    return true;
}

void FluidSystem::UpdateSSFRConstants(const Camera& camera)
{
    if (!m_useSSFR)
    {
        return;
    }

    UINT frameIndex = g_Engine->CurrentBackBufferIndex();
    if (frameIndex >= m_ssfrConstantBuffers.size())
    {
        return;
    }

    auto& cb = m_ssfrConstantBuffers[frameIndex];
    if (!cb)
    {
        return;
    }

    auto* constant = cb->GetPtr<SSFRConstant>();
    // ※シェーダー側では行列を列優先で解釈するため、ビュー・プロジェクションは転置して書き込んでSSFR描画の姿勢ずれを防ぐ
    constant->view = XMMatrixTranspose(camera.GetViewMatrix());
    constant->proj = XMMatrixTranspose(camera.GetProjMatrix());
    constant->screenSize = XMFLOAT2(static_cast<float>(m_ssfrWidth), static_cast<float>(m_ssfrHeight));
    constant->nearZ = 0.1f;
    constant->farZ = 1000.0f;
    constant->iorF0 = XMFLOAT3(0.02f, 0.02f, 0.02f);
    constant->absorb = 1.5f;
    // フル解像度へ戻した合成パス用にバックバッファ解像度を保持し、UV計算での不一致を防ぐ
    constant->framebufferSize = XMFLOAT2(
        static_cast<float>(g_Engine->FrameBufferWidth()),
        static_cast<float>(g_Engine->FrameBufferHeight()));
    // バイラテラルフィルタ係数も b0 に収納して RS の不一致警告 (#5337) を封じる
    constant->bilateralSigma = XMFLOAT2(m_bilateralSpatialSigma, m_bilateralDepthSigma);
    // 深度フィルタの再設計に合わせ、半径と深度閾値をまとめて渡す
    constant->bilateralKernel = XMFLOAT2(m_bilateralKernelRadius, m_bilateralDepthThreshold);
    constant->_pad = XMFLOAT2(0.0f, 0.0f);
}

void FluidSystem::TransitionSSFRTarget(ID3D12GraphicsCommandList* cmd, SSFRTarget& target, D3D12_RESOURCE_STATES newState)
{
    if (!target.resource)
    {
        return;
    }
    if (target.currentState == newState)
    {
        return;
    }

    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(target.resource.Get(), target.currentState, newState);
    cmd->ResourceBarrier(1, &barrier);
    target.currentState = newState;
}

void FluidSystem::PrepareSSFRTargets(ID3D12GraphicsCommandList* cmd)
{
    if (!cmd)
    {
        return;
    }

    // ※MRT に書き込む深度/厚みは毎フレーム既知の値へ初期化し、古い情報によるチラつきを防ぐ
    constexpr float kDepthClearValue = 1000000.0f;
    const float depthClear[4] = { kDepthClearValue, 0.0f, 0.0f, 0.0f };
    if (m_rawDepth.resource && m_rawDepth.rtvHandle.ptr != 0)
    {
        TransitionSSFRTarget(cmd, m_rawDepth, D3D12_RESOURCE_STATE_RENDER_TARGET);
        cmd->ClearRenderTargetView(m_rawDepth.rtvHandle, depthClear, 0, nullptr);
    }

    const float thicknessClear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    if (m_thickness.resource && m_thickness.rtvHandle.ptr != 0)
    {
        TransitionSSFRTarget(cmd, m_thickness, D3D12_RESOURCE_STATE_RENDER_TARGET);
        cmd->ClearRenderTargetView(m_thickness.rtvHandle, thicknessClear, 0, nullptr);
    }

    // ※コンピュート用の UAV もゼロ初期化し、残像ノイズを抑制する
    const float clearFloat[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    auto clearUav = [&](SSFRTarget& target)
    {
        if (!target.resource || !target.uavHandle || target.uavCpuHandle.ptr == 0)
        {
            return;
        }
        TransitionSSFRTarget(cmd, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd->ClearUnorderedAccessViewFloat(target.uavHandle->HandleGPU, target.uavCpuHandle, target.resource.Get(), clearFloat, 0, nullptr);
    };

    clearUav(m_smoothedDepth);
    clearUav(m_normal);
}

bool FluidSystem::CreateParticlePSO()
{
    auto device = g_Engine->Device();
    if (!device || !m_ssfrParticleRootSig)
    {
        return false;
    }

    Microsoft::WRL::ComPtr<ID3DBlob> vs;
    Microsoft::WRL::ComPtr<ID3DBlob> ps;
    if (!LoadShaderBytecode(L"SSFRParticleVS.cso", "main", "vs_5_0", vs))
    {
        return false;
    }
    if (!LoadShaderBytecode(L"SSFRParticlePS.cso", "main", "ps_5_0", ps))
    {
        return false;
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = m_ssfrParticleRootSig->Get();
    desc.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
    desc.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
    desc.SampleMask = UINT_MAX;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    desc.DepthStencilState.DepthEnable = FALSE;
    desc.DepthStencilState.StencilEnable = FALSE;
    desc.DSVFormat = DXGI_FORMAT_UNKNOWN; // ※深度バッファを使わない宣言で OMSetRenderTargets の整合性を確保
    desc.NumRenderTargets = 2;
    desc.RTVFormats[0] = DXGI_FORMAT_R32_FLOAT; // ※ビュー空間深度を線形で保持
    desc.RTVFormats[1] = DXGI_FORMAT_R16_FLOAT; // ※厚みは半精度で積算

    auto& depthBlend = desc.BlendState.RenderTarget[0];
    depthBlend.BlendEnable = TRUE;
    depthBlend.SrcBlend = D3D12_BLEND_ONE;
    depthBlend.DestBlend = D3D12_BLEND_ONE;
    depthBlend.BlendOp = D3D12_BLEND_OP_MIN; // ※粒子間の深度は最小値ブレンドで最前面を取得
    depthBlend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_RED;

    auto& thicknessBlend = desc.BlendState.RenderTarget[1];
    thicknessBlend.BlendEnable = TRUE;
    thicknessBlend.SrcBlend = D3D12_BLEND_ONE;
    thicknessBlend.DestBlend = D3D12_BLEND_ONE;
    thicknessBlend.BlendOp = D3D12_BLEND_OP_ADD; // ※厚みは加算して蓄積
    thicknessBlend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_RED;

    desc.SampleDesc.Count = 1;

    return SUCCEEDED(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(m_ssfrParticlePSO.ReleaseAndGetAddressOf())));
}

bool FluidSystem::CreateCompositePSO()
{
    auto device = g_Engine->Device();
    if (!device || !m_ssfrCompositeRootSig)
    {
        return false;
    }

    Microsoft::WRL::ComPtr<ID3DBlob> vs;
    Microsoft::WRL::ComPtr<ID3DBlob> ps;
    if (!LoadShaderBytecode(L"FullscreenVS.cso", "main", "vs_5_0", vs))
    {
        return false;
    }
    if (!LoadShaderBytecode(L"SSFRCompositePS.cso", "main", "ps_5_0", ps))
    {
        return false;
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = m_ssfrCompositeRootSig->Get();
    desc.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
    desc.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
    desc.SampleMask = UINT_MAX;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    desc.DepthStencilState.DepthEnable = FALSE;
    desc.DepthStencilState.StencilEnable = FALSE;
    desc.DSVFormat = DXGI_FORMAT_UNKNOWN; // ※合成も深度を使用しないため DSV 無しを宣言
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;

    return SUCCEEDED(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(m_ssfrCompositePSO.ReleaseAndGetAddressOf())));
}

bool FluidSystem::CreateComputePSO()
{
    auto device = g_Engine->Device();
    if (!device || !m_ssfrComputeRootSig)
    {
        return false;
    }

    m_ssfrBilateralPSO = std::make_unique<ComputePipelineState>();
    m_ssfrBilateralPSO->SetDevice(device);
    m_ssfrBilateralPSO->SetRootSignature(m_ssfrComputeRootSig->Get());
    m_ssfrBilateralPSO->SetCS(L"SSFRBilateralCS.cso");
    if (!m_ssfrBilateralPSO->Create())
    {
        return false;
    }

    m_ssfrNormalPSO = std::make_unique<ComputePipelineState>();
    m_ssfrNormalPSO->SetDevice(device);
    m_ssfrNormalPSO->SetRootSignature(m_ssfrComputeRootSig->Get());
    m_ssfrNormalPSO->SetCS(L"SSFRNormalCS.cso");
    if (!m_ssfrNormalPSO->Create())
    {
        return false;
    }

    return true;
}

bool FluidSystem::CreateSSFRRootSignatures()
{
    auto device = g_Engine->Device();
    (void)device;

    // ※粒子描画用のルートシグネチャ（CBV + SRV + UAV）
    {
        CD3DX12_DESCRIPTOR_RANGE srvRange;
        srvRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);

        CD3DX12_ROOT_PARAMETER params[2];
        params[0].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_ALL);
        params[1].InitAsDescriptorTable(1, &srvRange, D3D12_SHADER_VISIBILITY_VERTEX);

        D3D12_ROOT_SIGNATURE_DESC desc{};
        desc.NumParameters = _countof(params);
        desc.pParameters = params;
        desc.NumStaticSamplers = 0;
        desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        if (!m_ssfrParticleRootSig)
        {
            m_ssfrParticleRootSig = std::make_unique<RootSignature>();
        }
        if (!m_ssfrParticleRootSig->Init(desc))
        {
            return false;
        }
    }

    // ※合成用ルートシグネチャ（CBV + 個別SRVテーブル + サンプラ）
    {
        CD3DX12_DESCRIPTOR_RANGE ranges[5];
        for (UINT i = 0; i < 5; ++i)
        {
            ranges[i].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, i);
        }

        CD3DX12_ROOT_PARAMETER params[6];
        params[0].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_PIXEL);
        for (UINT i = 0; i < 5; ++i)
        {
            // SRVを連続テーブルへまとめずに個別指定し、ハンドルの非連続性による誤参照を排除する
            params[1 + i].InitAsDescriptorTable(1, &ranges[i], D3D12_SHADER_VISIBILITY_PIXEL);
        }

        CD3DX12_STATIC_SAMPLER_DESC sampler(0, D3D12_FILTER_MIN_MAG_MIP_LINEAR);
        sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;

        D3D12_ROOT_SIGNATURE_DESC desc{};
        desc.NumParameters = _countof(params);
        desc.pParameters = params;
        desc.NumStaticSamplers = 1;
        desc.pStaticSamplers = &sampler;
        desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        if (!m_ssfrCompositeRootSig)
        {
            m_ssfrCompositeRootSig = std::make_unique<RootSignature>();
        }
        if (!m_ssfrCompositeRootSig->Init(desc))
        {
            return false;
        }
    }

    // ※バイラテラル & 法線計算用ルートシグネチャ（CBV b0 + SRV + UAV）
    //    ┗ フィルタ係数も b0 へ統一し、RS スロット不一致による GPU エラー (#1314) を撲滅
    {
        CD3DX12_DESCRIPTOR_RANGE ranges[2];
        ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
        ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);

        CD3DX12_ROOT_PARAMETER params[3];
        params[0].InitAsConstantBufferView(0);
        params[1].InitAsDescriptorTable(1, &ranges[0]);
        params[2].InitAsDescriptorTable(1, &ranges[1]);

        D3D12_ROOT_SIGNATURE_DESC desc{};
        desc.NumParameters = _countof(params);
        desc.pParameters = params;
        desc.NumStaticSamplers = 0;
        desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

        if (!m_ssfrComputeRootSig)
        {
            m_ssfrComputeRootSig = std::make_unique<RootSignature>();
        }
        if (!m_ssfrComputeRootSig->Init(desc))
        {
            return false;
        }
    }

    return true;
}

void FluidSystem::RenderSSFR(ID3D12GraphicsCommandList* cmd, const Camera& camera)
{
    if (!cmd || !EnsureSSFRResources())
    {
        return;
    }

    // ※UAVクリア前にシェーダ可視ヒープを必ずバインドし、GPUハンドル無効化エラー(#1314)を防止する
    ID3D12DescriptorHeap* heaps[] = { g_Engine->CbvSrvUavHeap()->GetHeap() };
    cmd->SetDescriptorHeaps(1, heaps);

    // ここで半解像度のビューポートとシザーに切り替えて、半解像度テクスチャ全面へ粒子を書き込めるようにする
    D3D12_VIEWPORT vpSSFR = { 0.0f, 0.0f, static_cast<float>(m_ssfrWidth), static_cast<float>(m_ssfrHeight), 0.0f, 1.0f };
    D3D12_RECT scSSFR = { 0, 0, static_cast<LONG>(m_ssfrWidth), static_cast<LONG>(m_ssfrHeight) };
    cmd->RSSetViewports(1, &vpSSFR);
    cmd->RSSetScissorRects(1, &scSSFR);

    UpdateSSFRConstants(camera);
    PrepareSSFRTargets(cmd);

    if (m_rawDepth.rtvHandle.ptr != 0 && m_thickness.rtvHandle.ptr != 0)
    {
        // ※粒子スプラットは半解像度のMRTへ直接出力し、UAV競合を避ける
        D3D12_CPU_DESCRIPTOR_HANDLE rtvs[] = { m_rawDepth.rtvHandle, m_thickness.rtvHandle };
        cmd->OMSetRenderTargets(2, rtvs, FALSE, nullptr);
    }

    UINT frameIndex = g_Engine->CurrentBackBufferIndex();
    auto cbAddress = m_ssfrConstantBuffers[frameIndex]->GetAddress();

    // 粒子ビルボード深度 & 厚み生成
    cmd->SetGraphicsRootSignature(m_ssfrParticleRootSig->Get());
    cmd->SetPipelineState(m_ssfrParticlePSO.Get());
    cmd->SetGraphicsRootConstantBufferView(0, cbAddress);
    if (m_particleBufferSrv)
    {
        cmd->SetGraphicsRootDescriptorTable(1, m_particleBufferSrv->HandleGPU);
    }
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    cmd->DrawInstanced(4, static_cast<UINT>(m_particles.size()), 0, 0);

    TransitionSSFRTarget(cmd, m_rawDepth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    TransitionSSFRTarget(cmd, m_thickness, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    // 深度バイラテラルフィルタ
    if (m_ssfrBilateralPSO)
    {
        TransitionSSFRTarget(cmd, m_smoothedDepth, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd->SetComputeRootSignature(m_ssfrComputeRootSig->Get());
        cmd->SetPipelineState(m_ssfrBilateralPSO->Get());
        cmd->SetComputeRootConstantBufferView(0, cbAddress);
        if (m_rawDepth.srvHandle)
        {
            cmd->SetComputeRootDescriptorTable(1, m_rawDepth.srvHandle->HandleGPU);
        }
        if (m_smoothedDepth.uavHandle)
        {
            cmd->SetComputeRootDescriptorTable(2, m_smoothedDepth.uavHandle->HandleGPU);
        }

        UINT groupX = (m_ssfrWidth + 7) / 8;
        UINT groupY = (m_ssfrHeight + 7) / 8;
        cmd->Dispatch(groupX, groupY, 1);

        {
            auto uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr); // ※UAV 書き込みの完了を保証
            cmd->ResourceBarrier(1, &uavBarrier);
        }

        TransitionSSFRTarget(cmd, m_smoothedDepth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    // 法線再構成
    if (m_ssfrNormalPSO)
    {
        TransitionSSFRTarget(cmd, m_normal, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd->SetComputeRootSignature(m_ssfrComputeRootSig->Get());
        cmd->SetPipelineState(m_ssfrNormalPSO->Get());
        cmd->SetComputeRootConstantBufferView(0, cbAddress);
        if (m_smoothedDepth.srvHandle)
        {
            cmd->SetComputeRootDescriptorTable(1, m_smoothedDepth.srvHandle->HandleGPU);
        }
        if (m_normal.uavHandle)
        {
            cmd->SetComputeRootDescriptorTable(2, m_normal.uavHandle->HandleGPU);
        }

        UINT groupX = (m_ssfrWidth + 7) / 8;
        UINT groupY = (m_ssfrHeight + 7) / 8;
        cmd->Dispatch(groupX, groupY, 1);

        {
            auto uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr); // ※UAV 書き込みの完了を保証
            cmd->ResourceBarrier(1, &uavBarrier);
        }

        TransitionSSFRTarget(cmd, m_normal, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }

    TransitionSSFRTarget(cmd, m_smoothedDepth, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    TransitionSSFRTarget(cmd, m_thickness, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    // 背景カラーを退避
    ID3D12Resource* backBuffer = g_Engine->CurrentRenderTargetResource();
    if (backBuffer && m_sceneColorCopy.resource)
    {
        if (m_sceneColorCopy.currentState != D3D12_RESOURCE_STATE_COPY_DEST)
        {
            auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(m_sceneColorCopy.resource.Get(), m_sceneColorCopy.currentState, D3D12_RESOURCE_STATE_COPY_DEST);
            cmd->ResourceBarrier(1, &barrier);
            m_sceneColorCopy.currentState = D3D12_RESOURCE_STATE_COPY_DEST;
        }

        auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(backBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmd->ResourceBarrier(1, &toCopy);
        cmd->CopyResource(m_sceneColorCopy.resource.Get(), backBuffer);
        auto toRT = CD3DX12_RESOURCE_BARRIER::Transition(backBuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        cmd->ResourceBarrier(1, &toRT);

        auto toSrv = CD3DX12_RESOURCE_BARRIER::Transition(m_sceneColorCopy.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        cmd->ResourceBarrier(1, &toSrv);
        m_sceneColorCopy.currentState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    }

    ID3D12Resource* depthResource = g_Engine->DepthStencilBuffer();
    if (depthResource)
    {
        if (m_cachedSceneDepth != depthResource && m_sceneDepthSrv)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC depthSrv{};
            depthSrv.Format = DXGI_FORMAT_R32_FLOAT;
            depthSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            depthSrv.Texture2D.MostDetailedMip = 0;
            depthSrv.Texture2D.MipLevels = 1;
            depthSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            g_Engine->Device()->CreateShaderResourceView(depthResource, &depthSrv, m_sceneDepthSrv->HandleCPU);
            m_cachedSceneDepth = depthResource;
        }

        auto toSrv = CD3DX12_RESOURCE_BARRIER::Transition(depthResource, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        cmd->ResourceBarrier(1, &toSrv);
    }

    // ここで合成描画へ移る前にフル解像度へ戻す（半解像度のままだとフルスクリーン三角形が左上だけになるため）
    D3D12_VIEWPORT vpFull = { 0.0f, 0.0f,
        static_cast<float>(g_Engine->FrameBufferWidth()),
        static_cast<float>(g_Engine->FrameBufferHeight()), 0.0f, 1.0f };
    D3D12_RECT scFull = { 0, 0,
        static_cast<LONG>(g_Engine->FrameBufferWidth()),
        static_cast<LONG>(g_Engine->FrameBufferHeight()) };
    cmd->RSSetViewports(1, &vpFull);
    cmd->RSSetScissorRects(1, &scFull);

    D3D12_CPU_DESCRIPTOR_HANDLE backBufferRtv = g_Engine->CurrentBackBufferView();
    if (backBufferRtv.ptr != 0)
    {
        // ※合成パスではバックバッファへ描くため、RTV を明示的に結合して不定状態を回避
        cmd->OMSetRenderTargets(1, &backBufferRtv, FALSE, nullptr);
    }

    // 合成
    bool canComposite = m_smoothedDepth.srvHandle && m_normal.srvHandle && m_thickness.srvHandle && m_sceneDepthSrv && m_sceneColorCopy.srvHandle;
    if (canComposite)
    {
        cmd->SetGraphicsRootSignature(m_ssfrCompositeRootSig->Get());
        cmd->SetPipelineState(m_ssfrCompositePSO.Get());
        cmd->SetGraphicsRootConstantBufferView(0, cbAddress);
        // 連続SRVテーブル誤参照を防ぐため、各SRVを個別にバインドする
        cmd->SetGraphicsRootDescriptorTable(1, m_smoothedDepth.srvHandle->HandleGPU);
        cmd->SetGraphicsRootDescriptorTable(2, m_normal.srvHandle->HandleGPU);
        cmd->SetGraphicsRootDescriptorTable(3, m_thickness.srvHandle->HandleGPU);
        cmd->SetGraphicsRootDescriptorTable(4, m_sceneDepthSrv->HandleGPU);
        cmd->SetGraphicsRootDescriptorTable(5, m_sceneColorCopy.srvHandle->HandleGPU);
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd->DrawInstanced(3, 1, 0, 0);
    }

    // 状態を次フレーム向けに戻す
    if (depthResource)
    {
        auto toDepth = CD3DX12_RESOURCE_BARRIER::Transition(depthResource, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE);
        cmd->ResourceBarrier(1, &toDepth);
    }

    if (m_sceneColorCopy.resource)
    {
        auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(m_sceneColorCopy.resource.Get(), m_sceneColorCopy.currentState, D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->ResourceBarrier(1, &toCopy);
        m_sceneColorCopy.currentState = D3D12_RESOURCE_STATE_COPY_DEST;
    }

    {
        D3D12_CPU_DESCRIPTOR_HANDLE rtv = g_Engine->CurrentBackBufferView();
        D3D12_CPU_DESCRIPTOR_HANDLE dsv = g_Engine->DepthStencilView();
        // ※グリッドや通常ジオメトリへ戻る前に DSV を再結合し、#615 の描画不具合を防止
        if (rtv.ptr != 0)
        {
            if (dsv.ptr != 0)
            {
                cmd->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
            }
            else
            {
                cmd->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
            }
        }
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

    if (m_renderMode == RenderMode::SSFR && m_useSSFR)
    {
        RenderSSFR(cmd, camera); // ※SSFR を使う場合はスクリーンスペース流体描画へ切り替え
        cmd->SetGraphicsRootSignature(m_rootSignature->Get());
        cmd->SetGraphicsRootConstantBufferView(0, cb->GetAddress()); // ※SSFR 内でルートシグネチャが変わるため再設定
    }
    else if (m_renderMode == RenderMode::MarchingCubes && m_marchingPipelineState && m_marchingPipelineState->IsValid() &&
        m_marchingVertexBuffer && m_marchingIndexBuffer && m_marchingIndexCount > 0)
    {
        // グリッド密度を元に生成したサーフェスを描画する
        cmd->SetPipelineState(m_marchingPipelineState->Get());
        auto vbView = m_marchingVertexBuffer->View();
        auto ibView = m_marchingIndexBuffer->View();
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd->IASetVertexBuffers(0, 1, &vbView);
        cmd->IASetIndexBuffer(&ibView);
        cmd->DrawIndexedInstanced(m_marchingIndexCount, 1, 0, 0, 0);
    }
    else if (m_sphereVertexBuffer && m_sphereIndexBuffer && m_indexCount > 0)
    {
        cmd->SetPipelineState(m_pipelineState->Get());
        auto vbViews = std::array<D3D12_VERTEX_BUFFER_VIEW, 2>{ m_sphereVertexBuffer->View(), m_instanceBuffer->View() };
        auto ibView = m_sphereIndexBuffer->View();
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd->IASetVertexBuffers(0, static_cast<UINT>(vbViews.size()), vbViews.data());
        cmd->IASetIndexBuffer(&ibView);
        cmd->DrawIndexedInstanced(m_indexCount, static_cast<UINT>(m_particles.size()), 0, 0, 0);
    }

    if (m_drawParticlePoints && m_pointPipelineState && m_pointPipelineState->IsValid() && m_instanceBuffer)
    {
        // SSFR の結果に粒子中心を重ねて表示し、位置整合性を目視で検証できるようにする
        cmd->SetPipelineState(m_pointPipelineState->Get());
        auto pointVb = m_instanceBuffer->View();
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
        cmd->IASetVertexBuffers(0, 1, &pointVb);
        cmd->DrawInstanced(static_cast<UINT>(m_particles.size()), 1, 0, 0);
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
    GenerateMarchingCubesMesh(); // 境界変更時もサーフェスを再構築して破綻を防止する
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
