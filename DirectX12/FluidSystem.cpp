#include "FluidSystem.h"
#include "Camera.h"
#include "Engine.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

using namespace DirectX;

namespace
{
    constexpr float kGravity = -9.8f;       // 重力加速度
    constexpr float kWallThickness = 0.0f;  // AABB扱いなので実体厚みは不要
}

FluidSystem::FluidSystem()
    : m_world(XMMatrixIdentity())
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
    m_pipelineState->SetInputLayout(ParticleVertex::InputLayout);
    m_pipelineState->SetRootSignature(m_rootSignature->Get());
    m_pipelineState->SetVS(L"ParticleVS.cso");
    m_pipelineState->SetPS(L"ParticlePS.cso");
    m_pipelineState->SetDepthStencilFormat(DXGI_FORMAT_D32_FLOAT);
    m_pipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT);
    if (!m_pipelineState->IsValid())
    {
        return false;
    }

    size_t bufferSize = sizeof(ParticleVertex) * m_renderVertices.size();
    m_vertexBuffer = std::make_unique<VertexBuffer>(bufferSize, sizeof(ParticleVertex), m_renderVertices.data());
    if (!m_vertexBuffer || !m_vertexBuffer->IsValid())
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
    m_renderVertices.clear();
    m_renderVertices.resize(particleCount);

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

                m_renderVertices[index].position = particle.position;
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

    for (auto& particle : m_particles)
    {
        particle.velocity.y += gravityStep;
        particle.velocity.x *= dragFactor;
        particle.velocity.y *= dragFactor;
        particle.velocity.z *= dragFactor;

        particle.position.x += particle.velocity.x * dt;
        particle.position.y += particle.velocity.y * dt;
        particle.position.z += particle.velocity.z * dt;

        ResolveCollisions(particle);
    }

    UpdateVertexBuffer();
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

void FluidSystem::UpdateVertexBuffer()
{
    for (size_t i = 0; i < m_particles.size(); ++i)
    {
        m_renderVertices[i].position = m_particles[i].position;
    }

    if (!m_vertexBuffer)
    {
        return;
    }

    void* mapped = nullptr;
    auto resource = m_vertexBuffer->GetResource();
    if (resource && SUCCEEDED(resource->Map(0, nullptr, &mapped)))
    {
        memcpy(mapped, m_renderVertices.data(), sizeof(ParticleVertex) * m_renderVertices.size());
        resource->Unmap(0, nullptr);
    }
}

void FluidSystem::Draw(ID3D12GraphicsCommandList* cmd, const Camera& camera)
{
    if (!cmd || !m_vertexBuffer || !m_pipelineState || !m_rootSignature)
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
    cmd->SetPipelineState(m_pipelineState->Get());
    cmd->SetGraphicsRootConstantBufferView(0, cb->GetAddress());

    auto vbView = m_vertexBuffer->View();
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
    cmd->IASetVertexBuffers(0, 1, &vbView);
    cmd->DrawInstanced(static_cast<UINT>(m_particles.size()), 1, 0, 0);
}

void FluidSystem::SetBounds(const Bounds& bounds)
{
    m_bounds = bounds;
    for (auto& particle : m_particles)
    {
        ResolveCollisions(particle);
    }
    UpdateVertexBuffer();
}

void FluidSystem::MoveBounds(const XMFLOAT3& delta)
{
    m_bounds.min.x += delta.x;
    m_bounds.min.y += delta.y;
    m_bounds.min.z += delta.z;
    m_bounds.max.x += delta.x;
    m_bounds.max.y += delta.y;
    m_bounds.max.z += delta.z;

    for (auto& particle : m_particles)
    {
        particle.position.x += delta.x;
        particle.position.y += delta.y;
        particle.position.z += delta.z;
    }
    UpdateVertexBuffer();
}
