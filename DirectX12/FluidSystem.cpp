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
}

namespace
{
    constexpr float kSSFRNearZ = 0.1f;   // ニア・ファーを固定しシェーダーと同値にして座標系の不一致を避ける
    constexpr float kSSFRFarZ  = 1000.0f;
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

    if (!EnsureSSFRResources())
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

    if (!m_bilateralParamsBuffer)
    {
        m_bilateralParamsBuffer = std::make_unique<ConstantBuffer>(sizeof(BilateralParams));
        if (!m_bilateralParamsBuffer || !m_bilateralParamsBuffer->IsValid())
        {
            return false;
        }

        auto* params = m_bilateralParamsBuffer->GetPtr<BilateralParams>();
        params->spatialSigma = 2.0f;
        params->depthSigma = 0.05f;
        params->normalSigma = 16.0f;
        params->kernelRadius = 2;
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

    if (!m_ssfrRtvHeap)
    {
        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        desc.NumDescriptors = 8; // ※RTV をまとめて確保し、毎フレームの CreateRenderTargetView 呼び出しを最小化
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        if (FAILED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(m_ssfrRtvHeap.ReleaseAndGetAddressOf()))))
        {
            return false;
        }
        m_ssfrRtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        m_ssfrRtvCursor = 0;
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
    if (!m_ssfrRtvHeap)
    {
        handle.ptr = 0;
        return handle;
    }

    handle = m_ssfrRtvHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(m_ssfrRtvCursor) * m_ssfrRtvDescriptorSize;
    ++m_ssfrRtvCursor;
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

    auto createTexture = [&](SSFRTarget& target, DXGI_FORMAT format, bool createSrv, bool createUav, bool createRtv)
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

        auto initState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        if (createUav)
        {
            initState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        if (createRtv)
        {
            initState = D3D12_RESOURCE_STATE_RENDER_TARGET;
        }

        D3D12_CLEAR_VALUE clearValue{};
        const D3D12_CLEAR_VALUE* clearPtr = nullptr;
        if (createRtv)
        {
            clearValue.Format = format;
            float clearDepth = (format == DXGI_FORMAT_R32_FLOAT) ? kSSFRFarZ : 0.0f;
            clearValue.Color[0] = clearDepth;
            clearValue.Color[1] = clearDepth;
            clearValue.Color[2] = clearDepth;
            clearValue.Color[3] = clearDepth;
            clearPtr = &clearValue;
        }

        if (FAILED(device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            initState,
            clearPtr,
            IID_PPV_ARGS(target.resource.ReleaseAndGetAddressOf()))))
        {
            return false;
        }
        target.currentState = initState;

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
        else
        {
            target.uavHandle = nullptr;
            target.uavCpuHandle = {};
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

        if (createRtv)
        {
            if (target.rtvHandle.ptr == 0)
            {
                target.rtvHandle = AllocateRtvDescriptor();
            }
            device->CreateRenderTargetView(target.resource.Get(), nullptr, target.rtvHandle);
        }
        else
        {
            target.rtvHandle = {};
        }

        return true;
    };

    if (!createTexture(m_rawDepth, DXGI_FORMAT_R32_FLOAT, true, false, true))
    {
        return false;
    }

    if (!createTexture(m_smoothedDepth, DXGI_FORMAT_R32_FLOAT, true, true, false))
    {
        return false;
    }

    if (!createTexture(m_normal, DXGI_FORMAT_R16G16B16A16_FLOAT, true, true, false))
    {
        return false;
    }

    if (!createTexture(m_thickness, DXGI_FORMAT_R16_FLOAT, true, false, true))
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
    constant->view = camera.GetViewMatrix();
    constant->proj = camera.GetProjMatrix();
    constant->screenSize = XMFLOAT2(static_cast<float>(m_ssfrWidth), static_cast<float>(m_ssfrHeight));
    constant->nearZ = kSSFRNearZ;
    constant->farZ = kSSFRFarZ;
    constant->iorF0 = XMFLOAT3(0.02f, 0.02f, 0.02f);
    constant->absorb = 1.5f;
    // フル解像度へ戻した合成パス用にバックバッファ解像度を保持し、UV計算での不一致を防ぐ
    constant->framebufferSize = XMFLOAT2(
        static_cast<float>(g_Engine->FrameBufferWidth()),
        static_cast<float>(g_Engine->FrameBufferHeight()));
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

void FluidSystem::ClearSSFRIntermediateUAVs(ID3D12GraphicsCommandList* cmd)
{
    const float clearZero[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    auto clearTarget = [&](SSFRTarget& target)
    {
        if (!target.resource || !target.uavHandle || target.uavCpuHandle.ptr == 0)
        {
            return;
        }
        TransitionSSFRTarget(cmd, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd->ClearUnorderedAccessViewFloat(target.uavHandle->HandleGPU, target.uavCpuHandle, target.resource.Get(), clearZero, 0, nullptr);
    };

    // RTV 化した深度・厚みは別途クリアするため、ここでは UAV のみを 0 へ初期化する
    clearTarget(m_smoothedDepth);
    clearTarget(m_normal);
}

void FluidSystem::ClearSSFROffscreenRenderTargets(ID3D12GraphicsCommandList* cmd, float farDepth)
{
    const float depthClear[4] = { farDepth, farDepth, farDepth, farDepth };
    const float thicknessClear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    auto clearRtv = [&](SSFRTarget& target, const float color[4])
    {
        if (!target.resource || target.rtvHandle.ptr == 0)
        {
            return;
        }
        TransitionSSFRTarget(cmd, target, D3D12_RESOURCE_STATE_RENDER_TARGET);
        cmd->ClearRenderTargetView(target.rtvHandle, color, 0, nullptr);
    };

    // 最小値ブレンドを成立させるため、深度 RTV は遠方値で初期化して粒子の前面を確保する
    clearRtv(m_rawDepth, depthClear);
    // 厚み RTV は足し算に備えてゼロ初期化する
    clearRtv(m_thickness, thicknessClear);
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
    // RTV へ直接深度/厚みを書き込むため、RT0 は最小値ブレンド、RT1 は加算に設定する
    auto& minBlend = desc.BlendState.RenderTarget[0];
    minBlend.BlendEnable = TRUE;
    minBlend.SrcBlend = D3D12_BLEND_ONE;
    minBlend.DestBlend = D3D12_BLEND_ONE;
    minBlend.BlendOp = D3D12_BLEND_OP_MIN;
    minBlend.SrcBlendAlpha = D3D12_BLEND_ONE;
    minBlend.DestBlendAlpha = D3D12_BLEND_ONE;
    minBlend.BlendOpAlpha = D3D12_BLEND_OP_MIN;
    minBlend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

    auto& addBlend = desc.BlendState.RenderTarget[1];
    addBlend.BlendEnable = TRUE;
    addBlend.SrcBlend = D3D12_BLEND_ONE;
    addBlend.DestBlend = D3D12_BLEND_ONE;
    addBlend.BlendOp = D3D12_BLEND_OP_ADD;
    addBlend.SrcBlendAlpha = D3D12_BLEND_ONE;
    addBlend.DestBlendAlpha = D3D12_BLEND_ONE;
    addBlend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    addBlend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    desc.DepthStencilState.DepthEnable = FALSE;
    desc.DepthStencilState.StencilEnable = FALSE;
    desc.NumRenderTargets = 2;
    desc.RTVFormats[0] = DXGI_FORMAT_R32_FLOAT;
    desc.RTVFormats[1] = DXGI_FORMAT_R16_FLOAT;
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

    // ※粒子描画用のルートシグネチャ（CBV + SRV のみに簡素化し、RTV 書き込みへ移行）
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

    // ※バイラテラル & 法線計算用ルートシグネチャ（CBV2 + SRV + UAV）
    {
        CD3DX12_DESCRIPTOR_RANGE ranges[2];
        ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
        ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);

        CD3DX12_ROOT_PARAMETER params[4];
        params[0].InitAsConstantBufferView(0);
        params[1].InitAsConstantBufferView(1);
        params[2].InitAsDescriptorTable(1, &ranges[0]);
        params[3].InitAsDescriptorTable(1, &ranges[1]);

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
    ClearSSFROffscreenRenderTargets(cmd, kSSFRFarZ);
    ClearSSFRIntermediateUAVs(cmd);

    UINT frameIndex = g_Engine->CurrentBackBufferIndex();
    auto cbAddress = m_ssfrConstantBuffers[frameIndex]->GetAddress();

    if (m_rawDepth.rtvHandle.ptr == 0 || m_thickness.rtvHandle.ptr == 0)
    {
        return; // RTV が生成できていない場合は以降の描画を避ける
    }

    D3D12_CPU_DESCRIPTOR_HANDLE particleRtvs[] = { m_rawDepth.rtvHandle, m_thickness.rtvHandle };
    cmd->OMSetRenderTargets(_countof(particleRtvs), particleRtvs, FALSE, nullptr);

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
        cmd->SetComputeRootConstantBufferView(1, m_bilateralParamsBuffer->GetAddress());
        if (m_rawDepth.srvHandle)
        {
            cmd->SetComputeRootDescriptorTable(2, m_rawDepth.srvHandle->HandleGPU);
        }
        if (m_smoothedDepth.uavHandle)
        {
            cmd->SetComputeRootDescriptorTable(3, m_smoothedDepth.uavHandle->HandleGPU);
        }

        UINT groupX = (m_ssfrWidth + 7) / 8;
        UINT groupY = (m_ssfrHeight + 7) / 8;
        cmd->Dispatch(groupX, groupY, 1);

        TransitionSSFRTarget(cmd, m_smoothedDepth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    // 法線再構成
    if (m_ssfrNormalPSO)
    {
        TransitionSSFRTarget(cmd, m_normal, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd->SetComputeRootSignature(m_ssfrComputeRootSig->Get());
        cmd->SetPipelineState(m_ssfrNormalPSO->Get());
        cmd->SetComputeRootConstantBufferView(0, cbAddress);
        cmd->SetComputeRootConstantBufferView(1, m_bilateralParamsBuffer->GetAddress());
        if (m_smoothedDepth.srvHandle)
        {
            cmd->SetComputeRootDescriptorTable(2, m_smoothedDepth.srvHandle->HandleGPU);
        }
        if (m_normal.uavHandle)
        {
            cmd->SetComputeRootDescriptorTable(3, m_normal.uavHandle->HandleGPU);
        }

        UINT groupX = (m_ssfrWidth + 7) / 8;
        UINT groupY = (m_ssfrHeight + 7) / 8;
        cmd->Dispatch(groupX, groupY, 1);

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

    // 合成
    D3D12_CPU_DESCRIPTOR_HANDLE backBufferRtv = g_Engine->CurrentBackBufferView();
    cmd->OMSetRenderTargets(1, &backBufferRtv, FALSE, nullptr);
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

    if (m_useSSFR)
    {
        RenderSSFR(cmd, camera); // ※SSFR を使う場合はスクリーンスペース流体描画へ切り替え
        cmd->SetGraphicsRootSignature(m_rootSignature->Get());
        cmd->SetGraphicsRootConstantBufferView(0, cb->GetAddress()); // ※SSFR 内でルートシグネチャが変わるため再設定
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
