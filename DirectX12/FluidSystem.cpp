#include "FluidSystem.h"
#include "Engine.h"
#include "MetaBallPipelineState.h"
#include "RandomUtil.h"
#include "ComputePipelineState.h"
#include <d3dx12.h>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <random>

using namespace DirectX;

namespace
{
    // GPUバッファ用の粒子構造体
    struct GPUFluidParticle
    {
        XMFLOAT3 position;
        float    pad0 = 0.0f;
        XMFLOAT3 velocity;
        float    pad1 = 0.0f;
    };

    struct ParticleMetaGPU
    {
        XMFLOAT3 position;
        float    radius;
    };

    // PBFカーネル関数（Poly6）
    float Poly6(float r, float h)
    {
        if (r >= h)
        {
            return 0.0f;
        }
        float diff = h * h - r * r;
        float coeff = 315.0f / (64.0f * XM_PI * std::pow(h, 9));
        return coeff * diff * diff * diff;
    }

    // PBFカーネルの勾配（Spiky）
    XMFLOAT3 GradSpiky(const XMFLOAT3& rij, float r, float h)
    {
        if (r <= 1e-6f || r >= h)
        {
            return XMFLOAT3(0.0f, 0.0f, 0.0f);
        }
        float coeff = -45.0f / (XM_PI * std::pow(h, 6));
        float scale = coeff * (h - r) * (h - r) / r;
        return XMFLOAT3(rij.x * scale, rij.y * scale, rij.z * scale);
    }

    inline float Length(const XMFLOAT3& v)
    {
        return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }
}

FluidMaterial CreateFluidMaterial(FluidMaterialPreset preset)
{
    FluidMaterial material{};
    switch (preset)
    {
    case FluidMaterialPreset::Magma:
        material.restDensity = 1500.0f;
        material.particleMass = 1.2f;
        material.smoothingRadius = 0.14f;
        material.viscosity = 0.25f;      // 高い粘性でゆっくり流れる
        material.stiffness = 350.0f;
        material.renderRadius = 0.12f;
        material.lambdaEpsilon = 150.0f;
        material.xsphC = 0.15f;
        material.solverIterations = 6;
        break;
    case FluidMaterialPreset::Water:
    default:
        material = FluidMaterial();
        break;
    }
    return material;
}

FluidSystem::FluidSystem()
    : m_spatialGrid(0.12f)
{
    m_material = CreateFluidMaterial(FluidMaterialPreset::Water);
    m_boundsMin = XMFLOAT3(-2.0f, 0.0f, -2.0f);
    m_boundsMax = XMFLOAT3(2.0f, 4.0f, 2.0f);
    m_gridDim = XMUINT3(1, 1, 1);
}

FluidSystem::~FluidSystem()
{
    // GPU用フェンスイベントを確実にクローズしてリークを防ぐ
    if (m_computeFenceEvent)
    {
        CloseHandle(m_computeFenceEvent);
        m_computeFenceEvent = nullptr;
    }
}

void FluidSystem::Init(ID3D12Device* device, DXGI_FORMAT rtvFormat, UINT maxParticles, UINT threadGroupCount)
{
    (void)threadGroupCount;
    m_device = device;
    m_rtvFormat = rtvFormat;
    m_maxParticles = std::max<UINT>(1u, maxParticles);
    m_cpuParticles.clear();
    m_cpuParticles.reserve(m_maxParticles);
    m_particleCount = 0;

    UpdateGridSettings();
    CreateMetaPipeline(device, rtvFormat);
    CreateGPUResources(device);

    // GPU・CPUリソースの生成が完了したタイミングで初期化済みフラグを立てる
    m_initialized = true;

    // ひとまず初期状態として軽く粒子を生成しておく
    SpawnParticlesSphere(XMFLOAT3(0.0f, 1.0f, 0.0f), 0.6f, m_maxParticles / 2);

    UpdateParticleBuffer();
}

void FluidSystem::UseGPU(bool enable)
{
    if (!m_initialized)
    {
        return;
    }

    if (enable && !m_gpuAvailable)
    {
        CreateGPUResources(m_device);
    }

    m_useGPU = enable && m_gpuAvailable;
}

FluidSimulationMode FluidSystem::Mode() const
{
    return (m_useGPU && m_gpuAvailable) ? FluidSimulationMode::GPU : FluidSimulationMode::CPU;
}

void FluidSystem::SetMaterialPreset(FluidMaterialPreset preset)
{
    SetMaterial(CreateFluidMaterial(preset));
}

void FluidSystem::SetMaterial(const FluidMaterial& material)
{
    m_material = material;
    m_spatialGrid.SetCellSize(m_material.smoothingRadius);
    UpdateGridSettings();

    m_particleCount = static_cast<UINT>(std::min<size_t>(m_cpuParticles.size(), m_maxParticles));

    m_cpuDirty = true;
    m_gpuDirty = true;

    // マテリアル変更時はGPUリソースも再作成して整合を取る
    if (m_device)
    {
        CreateGPUResources(m_device);
    }
}

// ============================
// 流体生成
// ============================
void FluidSystem::SpawnParticlesSphere(const XMFLOAT3& center, float radius, UINT count)
{
    if (!m_initialized || count == 0)
    {
        return;
    }

    std::mt19937 rng{ std::random_device{}() };
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (UINT i = 0; i < count && m_particleCount < m_maxParticles; ++i)
    {
        float u = dist(rng);
        float v = dist(rng);
        float theta = 2.0f * XM_PI * u;
        float phi = std::acos(2.0f * v - 1.0f);
        float r = radius * std::cbrt(dist(rng));

        XMFLOAT3 offset{
            r * std::sin(phi) * std::cos(theta),
            r * std::cos(phi),
            r * std::sin(phi) * std::sin(theta)
        };

        FluidParticle particle{};
        particle.position = XMFLOAT3(center.x + offset.x, center.y + offset.y, center.z + offset.z);
        particle.velocity = XMFLOAT3(0.0f, 0.0f, 0.0f);
        particle.predicted = particle.position;
        particle.density = m_material.restDensity;
        particle.lambda = 0.0f;

        m_cpuParticles.push_back(particle);
        ++m_particleCount;
    }

    m_cpuDirty = true;
    m_gpuDirty = true;
}

// ============================
// 流体削除
// ============================
void FluidSystem::RemoveParticlesSphere(const XMFLOAT3& center, float radius)
{
    if (!m_initialized || m_particleCount == 0)
    {
        return;
    }

    float r2 = radius * radius;
    auto it = std::remove_if(m_cpuParticles.begin(), m_cpuParticles.end(), [&](const FluidParticle& p)
        {
            XMFLOAT3 diff{ p.position.x - center.x, p.position.y - center.y, p.position.z - center.z };
            float len2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
            return len2 <= r2;
        });
    m_cpuParticles.erase(it, m_cpuParticles.end());
    m_particleCount = static_cast<UINT>(std::min<size_t>(m_cpuParticles.size(), m_maxParticles));

    m_cpuDirty = true;
    m_gpuDirty = true;
}


void FluidSystem::QueueGather(const XMFLOAT3& target, float radius, float strength)
{
    m_gatherOps.push_back({ target, radius, strength });
}

void FluidSystem::QueueSplash(const XMFLOAT3& position, float radius, float impulse)
{
    m_splashOps.push_back({ position, radius, impulse });
}

void FluidSystem::ClearDynamicOperations()
{
    m_gatherOps.clear();
    m_splashOps.clear();
}

float FluidSystem::EffectiveTimeStep(float dt) const
{
    const float minStep = 1.0f / 240.0f;
    const float maxStep = 1.0f / 30.0f;
    return std::clamp(dt, minStep, maxStep);
}

void FluidSystem::Simulate(ID3D12GraphicsCommandList* cmd, float dt)
{
    if (!m_initialized)
    {
        return;
    }

    m_particleCount = static_cast<UINT>(std::min<size_t>(m_cpuParticles.size(), m_maxParticles));
    if (m_particleCount == 0)
    {
        return;
    }

    float step = EffectiveTimeStep(dt);

    if (Mode() == FluidSimulationMode::GPU)
    {
        // 前フレームの結果を読み戻してCPU側と同期
        if (m_pendingReadback)
        {
            ReadbackGPUToCPU();
        }

        ApplyExternalOperationsCPU(step);
        UpdateComputeParams(step);

        // GPU用コマンドリストをリセットして記録を開始
        ID3D12GraphicsCommandList* computeCmd = BeginComputeCommandList();
        if (!computeCmd)
        {
            // 取得に失敗した場合はGPUモードを諦めてCPUシミュレーションにフォールバック
            StepCPU(step);
            UpdateParticleBuffer();
            m_activeMetaSRV = m_cpuMetaSRV;
            return;
        }

        UploadCPUToGPU(computeCmd);
        StepGPU(computeCmd, step);
        SubmitComputeCommandList();
        m_activeMetaSRV = m_gpuMetaSRV;
    }
    else
    {
        ApplyExternalOperationsCPU(step);
        StepCPU(step);
        UpdateParticleBuffer();
        m_activeMetaSRV = m_cpuMetaSRV;
    }
}

void FluidSystem::Render(ID3D12GraphicsCommandList* cmd, const XMFLOAT4X4& invViewProj, const XMFLOAT3& camPos, float isoLevel)
{
    if (!m_initialized || !cmd || m_particleCount == 0 || !m_activeMetaSRV ||
        !m_metaRootSignature || !m_metaPipelineState)
    {
        return;
    }

    UINT frameIndex = g_Engine->CurrentBackBufferIndex();
    MetaConstants* cb = m_metaCB[frameIndex]->GetPtr<MetaConstants>();

    XMMATRIX invVP = XMLoadFloat4x4(&invViewProj);
    invVP = XMMatrixTranspose(invVP);
    XMStoreFloat4x4(&cb->InvViewProj, invVP);

    cb->CamRadius = XMFLOAT4(camPos.x, camPos.y, camPos.z, m_material.renderRadius);
    cb->IsoCount = XMFLOAT4(isoLevel * 0.6f, static_cast<float>(m_particleCount), m_rayStepScale, 0.0f);

    ID3D12DescriptorHeap* heaps[] = { g_Engine->CbvSrvUavHeap()->GetHeap() };
    cmd->SetDescriptorHeaps(1, heaps);
    cmd->SetGraphicsRootSignature(m_metaRootSignature.Get());
    cmd->SetPipelineState(m_metaPipelineState.Get());
    cmd->SetGraphicsRootDescriptorTable(0, m_activeMetaSRV->HandleGPU);
    cmd->SetGraphicsRootConstantBufferView(1, m_metaCB[frameIndex]->GetAddress());
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd->DrawInstanced(3, 1, 0, 0);
}

void FluidSystem::ApplyExternalOperationsCPU(float dt)
{
    if (m_cpuParticles.empty())
    {
        return;
    }

    bool modified = false;

    // 集束処理
    for (const auto& op : m_gatherOps)
    {
        XMVECTOR target = XMLoadFloat3(&op.target);
        for (auto& particle : m_cpuParticles)
        {
            XMVECTOR pos = XMLoadFloat3(&particle.position);
            XMVECTOR diff = XMVectorSubtract(target, pos);
            float dist = XMVectorGetX(XMVector3Length(diff));
            if (dist < op.radius && dist > 1e-5f)
            {
                float weight = 1.0f - (dist / op.radius);
                float accel = op.strength * weight;
                XMVECTOR dir = XMVector3Normalize(diff);
                XMVECTOR vel = XMLoadFloat3(&particle.velocity);
                vel = XMVectorAdd(vel, XMVectorScale(dir, accel * dt));
                XMStoreFloat3(&particle.velocity, vel);
                modified = true;
            }
        }
    }

    // 発散処理は1回で取り除く
    if (!m_splashOps.empty())
    {
        for (const auto& op : m_splashOps)
        {
            XMVECTOR origin = XMLoadFloat3(&op.origin);
            for (auto& particle : m_cpuParticles)
            {
                XMVECTOR pos = XMLoadFloat3(&particle.position);
                XMVECTOR diff = XMVectorSubtract(pos, origin);
                float dist = XMVectorGetX(XMVector3Length(diff));
                if (dist < op.radius && dist > 1e-5f)
                {
                    float weight = 1.0f - (dist / op.radius);
                    XMVECTOR dir = XMVector3Normalize(diff);
                    XMVECTOR vel = XMLoadFloat3(&particle.velocity);
                    vel = XMVectorAdd(vel, XMVectorScale(dir, op.impulse * weight));
                    XMStoreFloat3(&particle.velocity, vel);
                    modified = true;
                }
            }
        }
        m_splashOps.clear();
    }

    if (modified)
    {
        m_cpuDirty = true;
    }
}

void FluidSystem::StepCPU(float dt)
{
    if (m_cpuParticles.empty())
    {
        return;
    }

    const float h = m_material.smoothingRadius;
    const float mass = m_material.particleMass;
    const float restDensity = m_material.restDensity;
    const float epsilon = m_material.lambdaEpsilon;
    const float sCorrK = -0.001f;
    const float sCorrN = 4.0f;
    const float deltaQ = 0.3f * h;
    const float invRestDensity = 1.0f / restDensity;
    const XMFLOAT3 gravity = XMFLOAT3(0.0f, -9.8f, 0.0f);

    m_spatialGrid.Clear();

    // 外力適用と予測位置計算
    for (size_t i = 0; i < m_cpuParticles.size(); ++i)
    {
        auto& particle = m_cpuParticles[i];
        XMVECTOR vel = XMLoadFloat3(&particle.velocity);
        XMVECTOR grav = XMLoadFloat3(&gravity);
        vel = XMVectorAdd(vel, XMVectorScale(grav, dt));
        XMStoreFloat3(&particle.velocity, vel);

        XMVECTOR pos = XMLoadFloat3(&particle.position);
        XMVECTOR pred = XMVectorAdd(pos, XMVectorScale(vel, dt));
        XMStoreFloat3(&particle.predicted, pred);
        particle.lambda = 0.0f;

        m_spatialGrid.Insert(i, particle.predicted);
    }

    std::vector<size_t> neighbors;
    neighbors.reserve(64);
    std::vector<size_t> xsphNeighbors;
    xsphNeighbors.reserve(64);

    for (int iteration = 0; iteration < m_material.solverIterations; ++iteration)
    {
        // λ計算
        for (size_t i = 0; i < m_cpuParticles.size(); ++i)
        {
            FluidParticle& pi = m_cpuParticles[i];
            neighbors.clear();
            m_spatialGrid.Query(pi.predicted, h, neighbors);

            float density = 0.0f;
            for (size_t idx : neighbors)
            {
                const FluidParticle& pj = m_cpuParticles[idx];
                XMFLOAT3 rij{ pi.predicted.x - pj.predicted.x, pi.predicted.y - pj.predicted.y, pi.predicted.z - pj.predicted.z };
                float r = Length(rij);
                density += mass * Poly6(r, h);
            }
            pi.density = std::max(density, restDensity * 0.1f);
            float Ci = pi.density * invRestDensity - 1.0f;

            XMFLOAT3 gradSum{ 0.0f, 0.0f, 0.0f };
            float sumGrad2 = 0.0f;

            for (size_t idx : neighbors)
            {
                if (idx == i)
                {
                    continue;
                }
                const FluidParticle& pj = m_cpuParticles[idx];
                XMFLOAT3 rij{ pi.predicted.x - pj.predicted.x, pi.predicted.y - pj.predicted.y, pi.predicted.z - pj.predicted.z };
                float r = Length(rij);
                XMFLOAT3 grad = GradSpiky(rij, r, h);
                grad.x *= mass * invRestDensity;
                grad.y *= mass * invRestDensity;
                grad.z *= mass * invRestDensity;
                gradSum.x += grad.x;
                gradSum.y += grad.y;
                gradSum.z += grad.z;
                sumGrad2 += grad.x * grad.x + grad.y * grad.y + grad.z * grad.z;
            }

            sumGrad2 += gradSum.x * gradSum.x + gradSum.y * gradSum.y + gradSum.z * gradSum.z;
            pi.lambda = -Ci / (sumGrad2 + epsilon);
        }

        // Δp計算
        for (size_t i = 0; i < m_cpuParticles.size(); ++i)
        {
            FluidParticle& pi = m_cpuParticles[i];
            neighbors.clear();
            m_spatialGrid.Query(pi.predicted, h, neighbors);

            XMFLOAT3 delta{ 0.0f, 0.0f, 0.0f };
            for (size_t idx : neighbors)
            {
                if (idx == i)
                {
                    continue;
                }
                const FluidParticle& pj = m_cpuParticles[idx];
                XMFLOAT3 rij{ pi.predicted.x - pj.predicted.x, pi.predicted.y - pj.predicted.y, pi.predicted.z - pj.predicted.z };
                float r = Length(rij);
                if (r >= h)
                {
                    continue;
                }
                float w = Poly6(r, h);
                float corr = 0.0f;
                float wq = Poly6(deltaQ, h);
                if (wq > 0.0f)
                {
                    corr = sCorrK * std::pow(w / wq, sCorrN);
                }
                XMFLOAT3 grad = GradSpiky(rij, r, h);
                float factor = (pi.lambda + pj.lambda + corr) * mass * invRestDensity;
                delta.x += grad.x * factor;
                delta.y += grad.y * factor;
                delta.z += grad.z * factor;
            }

            pi.predicted.x += delta.x;
            pi.predicted.y += delta.y;
            pi.predicted.z += delta.z;

            ResolveBounds(pi);
        }
    }

    // 速度と位置の更新（XSPHによる安定化込み）
    for (auto& particle : m_cpuParticles)
    {
        XMFLOAT3 delta{ particle.predicted.x - particle.position.x,
            particle.predicted.y - particle.position.y,
            particle.predicted.z - particle.position.z };
        particle.velocity = XMFLOAT3(delta.x / dt, delta.y / dt, delta.z / dt);

        // XSPH粘性
        xsphNeighbors.clear();
        m_spatialGrid.Query(particle.predicted, h, xsphNeighbors);
        XMFLOAT3 xsph{ 0.0f, 0.0f, 0.0f };
        for (size_t idx : xsphNeighbors)
        {
            if (&particle == &m_cpuParticles[idx])
            {
                continue;
            }
            const FluidParticle& pj = m_cpuParticles[idx];
            XMFLOAT3 vij{ pj.velocity.x - particle.velocity.x,
                pj.velocity.y - particle.velocity.y,
                pj.velocity.z - particle.velocity.z };
            XMFLOAT3 rij{ particle.predicted.x - pj.predicted.x,
                particle.predicted.y - pj.predicted.y,
                particle.predicted.z - pj.predicted.z };
            float r = Length(rij);
            xsph.x += Poly6(r, h) * vij.x;
            xsph.y += Poly6(r, h) * vij.y;
            xsph.z += Poly6(r, h) * vij.z;
        }
        particle.velocity.x += m_material.xsphC * xsph.x;
        particle.velocity.y += m_material.xsphC * xsph.y;
        particle.velocity.z += m_material.xsphC * xsph.z;

        particle.position = particle.predicted;
    }

    m_cpuDirty = true;
}

void FluidSystem::ResolveBounds(FluidParticle& p) const
{
    p.predicted.x = std::clamp(p.predicted.x, m_boundsMin.x, m_boundsMax.x);
    p.predicted.y = std::clamp(p.predicted.y, m_boundsMin.y, m_boundsMax.y);
    p.predicted.z = std::clamp(p.predicted.z, m_boundsMin.z, m_boundsMax.z);
}

void FluidSystem::UploadCPUToGPU(ID3D12GraphicsCommandList* cmd)
{
    if (!m_gpuAvailable || !cmd || !m_gpuUpload)
    {
        return;
    }

    if (!m_cpuDirty && !m_gpuDirty)
    {
        return;
    }

    const UINT64 bufferSize = sizeof(GPUFluidParticle) * m_particleCount;
    if (bufferSize == 0)
    {
        return;
    }

    GPUFluidParticle* mapped = nullptr;
    if (SUCCEEDED(m_gpuUpload->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) && mapped)
    {
        for (UINT i = 0; i < m_particleCount; ++i)
        {
            mapped[i].position = m_cpuParticles[i].position;
            mapped[i].velocity = m_cpuParticles[i].velocity;
        }
        m_gpuUpload->Unmap(0, nullptr);
    }

    for (int i = 0; i < 2; ++i)
    {
        auto& buffer = m_gpuParticleBuffers[i];
        if (!buffer.resource)
        {
            continue;
        }
        auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(buffer.resource.Get(), buffer.state, D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->ResourceBarrier(1, &toCopy);
        buffer.state = D3D12_RESOURCE_STATE_COPY_DEST;
        cmd->CopyBufferRegion(buffer.resource.Get(), 0, m_gpuUpload.Get(), 0, sizeof(GPUFluidParticle) * m_particleCount);
        auto toSRV = CD3DX12_RESOURCE_BARRIER::Transition(buffer.resource.Get(), buffer.state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        cmd->ResourceBarrier(1, &toSRV);
        buffer.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }

    m_cpuDirty = false;
    m_gpuDirty = false;
}

void FluidSystem::UpdateComputeParams(float dt)
{
    if (!m_computeParamsCB)
    {
        m_computeParamsCB = std::make_unique<ConstantBuffer>(sizeof(GPUParams));
    }

    GPUParams* params = m_computeParamsCB->GetPtr<GPUParams>();
    params->restDensity = m_material.restDensity;
    params->particleMass = m_material.particleMass;
    params->viscosity = m_material.viscosity;
    params->stiffness = m_material.stiffness;
    params->radius = m_material.smoothingRadius;
    params->timeStep = dt;
    params->particleCount = m_particleCount;
    params->pad0 = 0;
    params->gridMin = m_boundsMin;
    params->pad1 = 0.0f;
    params->gridDim = m_gridDim;
    params->pad2 = 0;

    if (!m_dummyViewCB)
    {
        m_dummyViewCB = std::make_unique<ConstantBuffer>(sizeof(XMFLOAT4X4));
        XMFLOAT4X4 identity;
        XMStoreFloat4x4(&identity, XMMatrixIdentity());
        *m_dummyViewCB->GetPtr<XMFLOAT4X4>() = identity;
    }
}

void FluidSystem::StepGPU(ID3D12GraphicsCommandList* cmd, float dt)
{
    if (!m_gpuAvailable || !cmd || !m_buildGridPipeline || !m_particlePipeline)
    {
        return;
    }

    const UINT readIndex = m_gpuReadIndex;
    const UINT writeIndex = 1 - m_gpuReadIndex;

    auto& readBuffer = m_gpuParticleBuffers[readIndex];
    auto& writeBuffer = m_gpuParticleBuffers[writeIndex];

    auto toSRV = CD3DX12_RESOURCE_BARRIER::Transition(readBuffer.resource.Get(), readBuffer.state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &toSRV);
    readBuffer.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    auto toUAV = CD3DX12_RESOURCE_BARRIER::Transition(writeBuffer.resource.Get(), writeBuffer.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->ResourceBarrier(1, &toUAV);
    writeBuffer.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    auto metaToUAV = CD3DX12_RESOURCE_BARRIER::Transition(m_gpuMetaBuffer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->ResourceBarrier(1, &metaToUAV);

    auto gridCountToUAV = CD3DX12_RESOURCE_BARRIER::Transition(m_gpuGridCount.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->ResourceBarrier(1, &gridCountToUAV);

    auto gridTableToUAV = CD3DX12_RESOURCE_BARRIER::Transition(m_gpuGridTable.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->ResourceBarrier(1, &gridTableToUAV);

    // グリッドの粒子数カウントを毎フレーム確実にリセットしておく（クリア漏れによる異常なループを防ぐ）
    if (m_gpuGridCountUAV)
    {
        const UINT clearValues[4] = { 0, 0, 0, 0 };
        cmd->ClearUnorderedAccessViewUint(m_gpuGridCountUAV->HandleGPU, m_gpuGridCountUAV->HandleCPU, m_gpuGridCount.Get(), clearValues, 0, nullptr);
    }

    ID3D12DescriptorHeap* heaps[] = { g_Engine->CbvSrvUavHeap()->GetHeap() };
    cmd->SetDescriptorHeaps(1, heaps);
    cmd->SetComputeRootSignature(m_computeRootSignature.Get());

    // グリッド構築
    cmd->SetPipelineState(m_buildGridPipeline->Get());
    cmd->SetComputeRootConstantBufferView(0, m_computeParamsCB->GetAddress());
    cmd->SetComputeRootConstantBufferView(1, m_dummyViewCB->GetAddress());
    cmd->SetComputeRootDescriptorTable(2, readBuffer.srv->HandleGPU);
    cmd->SetComputeRootDescriptorTable(3, writeBuffer.uav->HandleGPU);
    cmd->SetComputeRootDescriptorTable(4, m_gpuMetaUAV->HandleGPU);
    cmd->SetComputeRootDescriptorTable(5, m_gpuGridCountUAV->HandleGPU);
    cmd->SetComputeRootDescriptorTable(6, m_gpuGridTableUAV->HandleGPU);

    UINT cellCount = m_gridDim.x * m_gridDim.y * m_gridDim.z;
    UINT totalThreads = std::max(m_particleCount, cellCount);
    UINT groups = (totalThreads + 255) / 256;
    cmd->Dispatch(groups, 1, 1);

    // 粒子更新
    cmd->SetPipelineState(m_particlePipeline->Get());
    cmd->SetComputeRootConstantBufferView(0, m_computeParamsCB->GetAddress());
    cmd->SetComputeRootConstantBufferView(1, m_dummyViewCB->GetAddress());
    cmd->SetComputeRootDescriptorTable(2, readBuffer.srv->HandleGPU);
    cmd->SetComputeRootDescriptorTable(3, writeBuffer.uav->HandleGPU);
    cmd->SetComputeRootDescriptorTable(4, m_gpuMetaUAV->HandleGPU);
    cmd->SetComputeRootDescriptorTable(5, m_gpuGridCountUAV->HandleGPU);
    cmd->SetComputeRootDescriptorTable(6, m_gpuGridTableUAV->HandleGPU);

    UINT groupsParticle = (m_particleCount + 255) / 256;
    cmd->Dispatch(groupsParticle, 1, 1);

    // 書き込み完了後の状態遷移
    auto metaToSRV = CD3DX12_RESOURCE_BARRIER::Transition(m_gpuMetaBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &metaToSRV);
    auto gridCountToSRV = CD3DX12_RESOURCE_BARRIER::Transition(m_gpuGridCount.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &gridCountToSRV);
    auto gridTableToSRV = CD3DX12_RESOURCE_BARRIER::Transition(m_gpuGridTable.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &gridTableToSRV);

    // 新しい結果を読み戻し用にコピー
    auto toCopySrc = CD3DX12_RESOURCE_BARRIER::Transition(writeBuffer.resource.Get(), writeBuffer.state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cmd->ResourceBarrier(1, &toCopySrc);
    writeBuffer.state = D3D12_RESOURCE_STATE_COPY_SOURCE;
    cmd->CopyBufferRegion(m_gpuReadback.Get(), 0, writeBuffer.resource.Get(), 0, sizeof(GPUFluidParticle) * m_particleCount);
    auto backToSRV = CD3DX12_RESOURCE_BARRIER::Transition(writeBuffer.resource.Get(), writeBuffer.state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &backToSRV);
    writeBuffer.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    m_gpuReadIndex = writeIndex;
    m_pendingReadback = true;
}

void FluidSystem::ReadbackGPUToCPU()
{
    if (!m_gpuReadback || !m_pendingReadback)
    {
        return;
    }

    // フェンスが完了していなければイベントで待機して安全に読み出す
    if (m_computeFence && m_lastSubmittedComputeFence != 0)
    {
        UINT64 completed = m_computeFence->GetCompletedValue();
        if (completed < m_lastSubmittedComputeFence && m_computeFenceEvent)
        {
            m_computeFence->SetEventOnCompletion(m_lastSubmittedComputeFence, m_computeFenceEvent);
            WaitForSingleObject(m_computeFenceEvent, INFINITE);
        }
    }

    GPUFluidParticle* mapped = nullptr;
    if (SUCCEEDED(m_gpuReadback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) && mapped)
    {
        for (UINT i = 0; i < m_particleCount; ++i)
        {
            m_cpuParticles[i].position = mapped[i].position;
            m_cpuParticles[i].velocity = mapped[i].velocity;
            m_cpuParticles[i].predicted = mapped[i].position;
        }
        m_gpuReadback->Unmap(0, nullptr);
    }

    m_pendingReadback = false;
}

void FluidSystem::UpdateParticleBuffer()
{
    if (!m_cpuMetaBuffer)
    {
        return;
    }

    ParticleMetaGPU* mapped = nullptr;
    if (FAILED(m_cpuMetaBuffer->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) || !mapped)
    {
        return;
    }

    for (UINT i = 0; i < m_particleCount; ++i)
    {
        mapped[i].position = m_cpuParticles[i].position;
        mapped[i].radius = m_material.renderRadius;
    }

    for (UINT i = m_particleCount; i < m_maxParticles; ++i)
    {
        mapped[i].position = XMFLOAT3(0.0f, 0.0f, 0.0f);
        mapped[i].radius = 0.0f;
    }

    m_cpuMetaBuffer->Unmap(0, nullptr);
    m_activeMetaSRV = m_cpuMetaSRV;
}

void FluidSystem::CreateMetaPipeline(ID3D12Device* device, DXGI_FORMAT rtvFormat)
{
    if (!graphics::MetaBallPipeline::CreateRootSignature(device, m_metaRootSignature))
    {
        printf("FluidSystem: メタボール描画用ルートシグネチャの初期化に失敗しました\n");
        return;
    }

    if (!graphics::MetaBallPipeline::CreatePipelineState(device, m_metaRootSignature.Get(), rtvFormat, m_metaPipelineState))
    {
        printf("FluidSystem: メタボール描画用パイプラインの初期化に失敗しました\n");
        m_metaRootSignature.Reset();
        return;
    }

    for (UINT i = 0; i < kFrameCount; ++i)
    {
        m_metaCB[i] = std::make_unique<ConstantBuffer>(sizeof(MetaConstants));
    }

    UINT64 bufferSize = sizeof(ParticleMetaGPU) * m_maxParticles;

    // CPU更新用アップロードバッファ
    auto cpuHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto cpuDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
    if (FAILED(device->CreateCommittedResource(&cpuHeap, D3D12_HEAP_FLAG_NONE, &cpuDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(m_cpuMetaBuffer.ReleaseAndGetAddressOf()))))
    {
        printf("FluidSystem: CPUメタデータバッファ生成に失敗しました\n");
        return;
    }

    m_cpuMetaSRV = g_Engine->CbvSrvUavHeap()->RegisterBuffer(m_cpuMetaBuffer.Get(), m_maxParticles, sizeof(ParticleMetaGPU));
    if (!m_cpuMetaSRV)
    {
        printf("FluidSystem: CPUメタデータSRVの登録に失敗しました\n");
        return;
    }
    m_activeMetaSRV = m_cpuMetaSRV;

    // GPU書き込み用バッファ
    auto gpuHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto gpuDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (FAILED(device->CreateCommittedResource(&gpuHeap, D3D12_HEAP_FLAG_NONE, &gpuDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
        IID_PPV_ARGS(m_gpuMetaBuffer.ReleaseAndGetAddressOf()))))
    {
        printf("FluidSystem: GPUメタデータバッファ生成に失敗しました\n");
        return;
    }

    if (!m_gpuMetaSRV)
    {
        m_gpuMetaSRV = g_Engine->CbvSrvUavHeap()->RegisterBuffer(m_gpuMetaBuffer.Get(), m_maxParticles, sizeof(ParticleMetaGPU));
    }
    else
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc.Buffer.NumElements = m_maxParticles;
        desc.Buffer.StructureByteStride = sizeof(ParticleMetaGPU);
        g_Engine->Device()->CreateShaderResourceView(m_gpuMetaBuffer.Get(), &desc, m_gpuMetaSRV->HandleCPU);
    }
    if (!m_gpuMetaSRV)
    {
        printf("FluidSystem: GPUメタデータSRVの登録に失敗しました\n");
        return;
    }

    if (!m_gpuMetaUAV)
    {
        m_gpuMetaUAV = g_Engine->CbvSrvUavHeap()->RegisterBufferUAV(m_gpuMetaBuffer.Get(), m_maxParticles, sizeof(ParticleMetaGPU));
    }
    else
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        desc.Buffer.NumElements = m_maxParticles;
        desc.Buffer.StructureByteStride = sizeof(ParticleMetaGPU);
        g_Engine->Device()->CreateUnorderedAccessView(m_gpuMetaBuffer.Get(), nullptr, &desc, m_gpuMetaUAV->HandleCPU);
    }
    if (!m_gpuMetaUAV)
    {
        printf("FluidSystem: GPUメタデータUAVの登録に失敗しました\n");
        return;
    }
}

void FluidSystem::CreateGPUResources(ID3D12Device* device)
{
    if (!device)
    {
        return;
    }

    UpdateGridSettings();

    UINT particleStride = sizeof(GPUFluidParticle);
    UINT metaStride = sizeof(ParticleMetaGPU);
    UINT particleBufferSize = particleStride * m_maxParticles;
    UINT metaBufferSize = metaStride * m_maxParticles;
    UINT cellCount = std::max<UINT>(1u, m_gridDim.x * m_gridDim.y * m_gridDim.z);

    auto defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    for (int i = 0; i < 2; ++i)
    {
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(particleBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        HRESULT hr = device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
            IID_PPV_ARGS(m_gpuParticleBuffers[i].resource.ReleaseAndGetAddressOf()));
        if (FAILED(hr))
        {
            printf("FluidSystem: GPU粒子バッファ生成に失敗しました (%d)\n", i);
            m_gpuAvailable = false;
            return;
        }
        m_gpuParticleBuffers[i].state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        if (!m_gpuParticleBuffers[i].srv)
        {
            m_gpuParticleBuffers[i].srv = g_Engine->CbvSrvUavHeap()->RegisterBuffer(m_gpuParticleBuffers[i].resource.Get(), m_maxParticles, particleStride);
        }
        else
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC descSrv{};
            descSrv.Format = DXGI_FORMAT_UNKNOWN;
            descSrv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            descSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            descSrv.Buffer.NumElements = m_maxParticles;
            descSrv.Buffer.StructureByteStride = particleStride;
            g_Engine->Device()->CreateShaderResourceView(m_gpuParticleBuffers[i].resource.Get(), &descSrv, m_gpuParticleBuffers[i].srv->HandleCPU);
        }
        if (!m_gpuParticleBuffers[i].srv)
        {
            printf("FluidSystem: GPU粒子SRVの登録に失敗しました (%d)\n", i);
            m_gpuAvailable = false;
            return;
        }

        if (!m_gpuParticleBuffers[i].uav)
        {
            m_gpuParticleBuffers[i].uav = g_Engine->CbvSrvUavHeap()->RegisterBufferUAV(m_gpuParticleBuffers[i].resource.Get(), m_maxParticles, particleStride);
        }
        else
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC descUAV{};
            descUAV.Format = DXGI_FORMAT_UNKNOWN;
            descUAV.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            descUAV.Buffer.NumElements = m_maxParticles;
            descUAV.Buffer.StructureByteStride = particleStride;
            g_Engine->Device()->CreateUnorderedAccessView(m_gpuParticleBuffers[i].resource.Get(), nullptr, &descUAV, m_gpuParticleBuffers[i].uav->HandleCPU);
        }
        if (!m_gpuParticleBuffers[i].uav)
        {
            printf("FluidSystem: GPU粒子UAVの登録に失敗しました (%d)\n", i);
            m_gpuAvailable = false;
            return;
        }
    }

    // グリッドバッファ
    auto gridDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(UINT) * cellCount, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    HRESULT hrGrid = device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &gridDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
        IID_PPV_ARGS(m_gpuGridCount.ReleaseAndGetAddressOf()));
    if (FAILED(hrGrid))
    {
        printf("FluidSystem: グリッドカウントバッファ生成に失敗しました\n");
        m_gpuAvailable = false;
        return;
    }
    if (!m_gpuGridCountUAV)
    {
        m_gpuGridCountUAV = g_Engine->CbvSrvUavHeap()->RegisterBufferUAV(m_gpuGridCount.Get(), cellCount, sizeof(UINT));
    }
    else
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        desc.Buffer.NumElements = cellCount;
        desc.Buffer.StructureByteStride = sizeof(UINT);
        g_Engine->Device()->CreateUnorderedAccessView(m_gpuGridCount.Get(), nullptr, &desc, m_gpuGridCountUAV->HandleCPU);
    }
    if (!m_gpuGridCountUAV)
    {
        printf("FluidSystem: グリッドカウントUAVの登録に失敗しました\n");
        m_gpuAvailable = false;
        return;
    }

    auto tableDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(UINT) * cellCount * 64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    HRESULT hrTable = device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &tableDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
        IID_PPV_ARGS(m_gpuGridTable.ReleaseAndGetAddressOf()));
    if (FAILED(hrTable))
    {
        printf("FluidSystem: グリッドテーブル生成に失敗しました\n");
        m_gpuAvailable = false;
        return;
    }
    if (!m_gpuGridTableUAV)
    {
        m_gpuGridTableUAV = g_Engine->CbvSrvUavHeap()->RegisterBufferUAV(m_gpuGridTable.Get(), cellCount * 64, sizeof(UINT));
    }
    else
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        desc.Buffer.NumElements = cellCount * 64;
        desc.Buffer.StructureByteStride = sizeof(UINT);
        g_Engine->Device()->CreateUnorderedAccessView(m_gpuGridTable.Get(), nullptr, &desc, m_gpuGridTableUAV->HandleCPU);
    }
    if (!m_gpuGridTableUAV)
    {
        printf("FluidSystem: グリッドテーブルUAVの登録に失敗しました\n");
        m_gpuAvailable = false;
        return;
    }

    // アップロード・リードバック
    auto uploadHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto uploadDesc = CD3DX12_RESOURCE_DESC::Buffer(particleBufferSize);
    HRESULT hrUpload = device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &uploadDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(m_gpuUpload.ReleaseAndGetAddressOf()));
    if (FAILED(hrUpload))
    {
        printf("FluidSystem: GPUアップロードバッファ生成に失敗しました\n");
        m_gpuAvailable = false;
        return;
    }

    auto readbackHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    auto readbackDesc = CD3DX12_RESOURCE_DESC::Buffer(particleBufferSize);
    HRESULT hrReadback = device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE, &readbackDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(m_gpuReadback.ReleaseAndGetAddressOf()));
    if (FAILED(hrReadback))
    {
        printf("FluidSystem: GPUリードバックバッファ生成に失敗しました\n");
        m_gpuAvailable = false;
        return;
    }

    // コンピュート用のルートシグネチャとPSO
    CD3DX12_DESCRIPTOR_RANGE srvRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
    CD3DX12_DESCRIPTOR_RANGE uavRange0(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
    CD3DX12_DESCRIPTOR_RANGE uavRange1(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1);
    CD3DX12_DESCRIPTOR_RANGE uavRange2(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2);
    CD3DX12_DESCRIPTOR_RANGE uavRange3(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 3);

    CD3DX12_ROOT_PARAMETER params[7];
    params[0].InitAsConstantBufferView(0);
    params[1].InitAsConstantBufferView(1);
    params[2].InitAsDescriptorTable(1, &srvRange);
    params[3].InitAsDescriptorTable(1, &uavRange0);
    params[4].InitAsDescriptorTable(1, &uavRange1);
    params[5].InitAsDescriptorTable(1, &uavRange2);
    params[6].InitAsDescriptorTable(1, &uavRange3);

    CD3DX12_ROOT_SIGNATURE_DESC rootDesc;
    rootDesc.Init(_countof(params), params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> serialized;
    ComPtr<ID3DBlob> errors;
    if (FAILED(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, serialized.GetAddressOf(), errors.GetAddressOf())))
    {
        if (errors)
        {
            printf("Compute root signature error: %s\n", (char*)errors->GetBufferPointer());
        }
        return;
    }
    HRESULT hrRoot = device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(m_computeRootSignature.ReleaseAndGetAddressOf()));
    if (FAILED(hrRoot))
    {
        printf("FluidSystem: コンピュート用ルートシグネチャ生成に失敗しました\n");
        m_gpuAvailable = false;
        return;
    }

    m_buildGridPipeline = std::make_unique<ComputePipelineState>();
    m_buildGridPipeline->SetDevice(device);
    m_buildGridPipeline->SetRootSignature(m_computeRootSignature.Get());
    m_buildGridPipeline->SetCS(L"BuildGridCS.cso");
    if (!m_buildGridPipeline->Create())
    {
        m_buildGridPipeline.reset();
        m_particlePipeline.reset();
        m_gpuAvailable = false;
        return;
    }

    m_particlePipeline = std::make_unique<ComputePipelineState>();
    m_particlePipeline->SetDevice(device);
    m_particlePipeline->SetRootSignature(m_computeRootSignature.Get());
    m_particlePipeline->SetCS(L"ParticleCS.cso");
    if (!m_particlePipeline->Create())
    {
        m_particlePipeline.reset();
        m_gpuAvailable = false;
        return;
    }

    // コンピュート用のコマンドアロケーター／リスト／フェンスを準備
    if (!m_computeAllocator)
    {
        HRESULT hrAlloc = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(m_computeAllocator.ReleaseAndGetAddressOf()));
        if (FAILED(hrAlloc))
        {
            printf("FluidSystem: コンピュート用アロケーター生成に失敗しました\n");
            m_gpuAvailable = false;
            return;
        }
    }

    if (!m_computeCommandList)
    {
        HRESULT hrList = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_computeAllocator.Get(), nullptr, IID_PPV_ARGS(m_computeCommandList.ReleaseAndGetAddressOf()));
        if (FAILED(hrList))
        {
            printf("FluidSystem: コンピュート用コマンドリスト生成に失敗しました\n");
            m_gpuAvailable = false;
            return;
        }
        // 生成直後は開いているので一度閉じておく
        m_computeCommandList->Close();
    }

    if (!m_computeFence)
    {
        HRESULT hrFence = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(m_computeFence.ReleaseAndGetAddressOf()));
        if (FAILED(hrFence))
        {
            printf("FluidSystem: コンピュート用フェンス生成に失敗しました\n");
            m_gpuAvailable = false;
            return;
        }
        m_computeFenceValue = 0;
        m_lastSubmittedComputeFence = 0;
    }

    if (!m_computeFenceEvent)
    {
        m_computeFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (!m_computeFenceEvent)
        {
            printf("FluidSystem: フェンスイベント生成に失敗しました\n");
            m_gpuAvailable = false;
            return;
        }
    }

    m_gpuReadIndex = 0;
    m_pendingReadback = false;
    m_gpuAvailable = true;
    m_gpuDirty = true;
}

ID3D12GraphicsCommandList* FluidSystem::BeginComputeCommandList()
{
    if (!m_computeAllocator || !m_computeCommandList)
    {
        return nullptr;
    }

    // フレーム毎にアロケーターとコマンドリストをリセットして記録を開始
    HRESULT hrAlloc = m_computeAllocator->Reset();
    if (FAILED(hrAlloc))
    {
        printf("FluidSystem: コンピュートアロケーターのリセットに失敗しました\n");
        return nullptr;
    }

    HRESULT hrCmd = m_computeCommandList->Reset(m_computeAllocator.Get(), nullptr);
    if (FAILED(hrCmd))
    {
        printf("FluidSystem: コンピュートコマンドリストのリセットに失敗しました\n");
        return nullptr;
    }

    return m_computeCommandList.Get();
}

void FluidSystem::SubmitComputeCommandList()
{
    if (!m_computeCommandList)
    {
        return;
    }

    HRESULT hrClose = m_computeCommandList->Close();
    if (FAILED(hrClose))
    {
        printf("FluidSystem: コンピュートコマンドリストのクローズに失敗しました\n");
        return;
    }

    ID3D12CommandList* lists[] = { m_computeCommandList.Get() };
    ID3D12CommandQueue* queue = g_Engine->ComputeCommandQueue();
    if (!queue)
    {
        // コンピュートキューが無ければ描画キューで代用
        queue = g_Engine->CommandQueue();
    }

    if (!queue)
    {
        return;
    }

    queue->ExecuteCommandLists(1, lists);

    if (m_computeFence)
    {
        ++m_computeFenceValue;
        if (SUCCEEDED(queue->Signal(m_computeFence.Get(), m_computeFenceValue)))
        {
            m_lastSubmittedComputeFence = m_computeFenceValue;

            // グラフィックスキューはコンピュート結果を待ってから描画を継続
            if (ID3D12CommandQueue* graphicsQueue = g_Engine->CommandQueue())
            {
                if (graphicsQueue != queue)
                {
                    graphicsQueue->Wait(m_computeFence.Get(), m_computeFenceValue);
                }
            }
        }
    }
}

void FluidSystem::UpdateGridSettings()
{
    float cellSize = std::max(0.02f, m_material.smoothingRadius);
    m_spatialGrid.SetCellSize(cellSize);

    float width = m_boundsMax.x - m_boundsMin.x;
    float height = m_boundsMax.y - m_boundsMin.y;
    float depth = m_boundsMax.z - m_boundsMin.z;

    m_gridDim.x = std::max<UINT>(1u, static_cast<UINT>(std::ceil(width / cellSize)));
    m_gridDim.y = std::max<UINT>(1u, static_cast<UINT>(std::ceil(height / cellSize)));
    m_gridDim.z = std::max<UINT>(1u, static_cast<UINT>(std::ceil(depth / cellSize)));
}
