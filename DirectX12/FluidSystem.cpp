#include "FluidSystem.h"
#include "Engine.h"
#include "RandomUtil.h"
#include <algorithm>
#include <d3dx12.h>

using namespace DirectX;

namespace
{
    // 指定された大きさのアップロードバッファを生成してマップするヘルパー関数
    Microsoft::WRL::ComPtr<ID3D12Resource> CreateMappedUploadBuffer(
        ID3D12Device* device,
        size_t size,
        void** mapped)
    {
        if (!device)
        {
            return nullptr;
        }

        CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_UPLOAD);
        CD3DX12_RESOURCE_DESC   desc = CD3DX12_RESOURCE_DESC::Buffer(size);

        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        HRESULT hr = device->CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(resource.GetAddressOf()));
        if (FAILED(hr))
        {
            return nullptr;
        }

        if (mapped)
        {
            hr = resource->Map(0, nullptr, mapped);
            if (FAILED(hr))
            {
                resource.Reset();
                return nullptr;
            }
        }

        return resource;
    }
}

FluidMaterial CreateFluidMaterial(FluidMaterialPreset preset)
{
    FluidMaterial material{};
    switch (preset)
    {
    case FluidMaterialPreset::Magma:
        material.renderRadius = 0.15f; // マグマらしく少し大きめにする
        break;
    case FluidMaterialPreset::Water:
    default:
        material.renderRadius = 0.10f;
        break;
    }
    return material;
}

FluidSystem::FluidSystem()
{
    // シミュレーション空間をある程度の箱に制限する
    m_boundsMin = XMFLOAT3(-2.0f, 0.0f, -2.0f);
    m_boundsMax = XMFLOAT3(2.0f, 4.0f, 2.0f);
    m_material = CreateFluidMaterial(FluidMaterialPreset::Water);

    m_waterColorShallow = XMFLOAT3(0.25f, 0.55f, 0.95f);
    m_waterColorDeep = XMFLOAT3(0.07f, 0.22f, 0.38f);
}

FluidSystem::~FluidSystem()
{
    // マップ済みバッファは開放前にアンマップする
    if (m_particleBuffer && m_particleMapped)
    {
        m_particleBuffer->Unmap(0, nullptr);
        m_particleMapped = nullptr;
    }
}

void FluidSystem::Init(ID3D12Device* device, DXGI_FORMAT rtvFormat, UINT maxParticles, UINT /*threadGroupCount*/)
{
    // 粒子数の上限を覚えておく
    m_maxParticles = std::max<UINT>(1u, maxParticles);
    m_particles.reserve(m_maxParticles);

    // メタボール描画用の定数バッファを生成
    m_constantBuffer = std::make_unique<ConstantBuffer>(sizeof(MetaConstants));

    // 粒子メタ情報用のアップロードバッファを生成
    size_t bufferSize = sizeof(MetaBallInstance) * static_cast<size_t>(m_maxParticles);
    m_particleBuffer = CreateMappedUploadBuffer(device, bufferSize, &m_particleMapped);
    if (m_particleBuffer)
    {
        std::fill_n(reinterpret_cast<MetaBallInstance*>(m_particleMapped), m_maxParticles, MetaBallInstance{ XMFLOAT3(0,0,0), 0.0f });
        m_particleSRV = g_Engine->CbvSrvUavHeap()->RegisterBuffer(m_particleBuffer.Get(), m_maxParticles, sizeof(MetaBallInstance));
    }

    // グリッド情報は未使用なので 1 要素のダミーバッファを用意する
    UINT zeroValue = 0;
    void* mappedTable = nullptr;
    void* mappedCount = nullptr;
    m_dummyGridTable = CreateMappedUploadBuffer(device, sizeof(UINT), &mappedTable);
    m_dummyGridCount = CreateMappedUploadBuffer(device, sizeof(UINT), &mappedCount);
    if (mappedTable)
    {
        *reinterpret_cast<UINT*>(mappedTable) = zeroValue;
        m_dummyGridTableSRV = g_Engine->CbvSrvUavHeap()->RegisterBuffer(m_dummyGridTable.Get(), 1, sizeof(UINT));
        m_dummyGridTable->Unmap(0, nullptr);
    }
    if (mappedCount)
    {
        *reinterpret_cast<UINT*>(mappedCount) = zeroValue;
        m_dummyGridCountSRV = g_Engine->CbvSrvUavHeap()->RegisterBuffer(m_dummyGridCount.Get(), 1, sizeof(UINT));
        m_dummyGridCount->Unmap(0, nullptr);
    }

    // メタボール描画用 PSO とルートシグネチャを作成
    CreateMetaResources(device, rtvFormat);
}

void FluidSystem::SetMaterialPreset(FluidMaterialPreset preset)
{
    SetMaterial(CreateFluidMaterial(preset));
}

void FluidSystem::SetMaterial(const FluidMaterial& material)
{
    m_material = material;
}

void FluidSystem::SpawnParticlesSphere(const XMFLOAT3& center, float radius, UINT count)
{
    if (count == 0)
    {
        return;
    }

    for (UINT i = 0; i < count && m_particles.size() < m_maxParticles; ++i)
    {
        XMFLOAT3 offset;
        float lengthSquared;
        do
        {
            offset = XMFLOAT3(RandFloat(-1.0f, 1.0f), RandFloat(-1.0f, 1.0f), RandFloat(-1.0f, 1.0f));
            lengthSquared = offset.x * offset.x + offset.y * offset.y + offset.z * offset.z;
        } while (lengthSquared > 1.0f);

        offset.x *= radius;
        offset.y *= radius;
        offset.z *= radius;

        Particle particle{};
        particle.position = XMFLOAT3(center.x + offset.x, center.y + offset.y, center.z + offset.z);
        particle.velocity = XMFLOAT3(0.0f, 0.0f, 0.0f);
        m_particles.push_back(particle);
    }
}

void FluidSystem::RemoveParticlesSphere(const XMFLOAT3& center, float radius)
{
    float radiusSq = radius * radius;
    auto endIt = std::remove_if(m_particles.begin(), m_particles.end(), [&](const Particle& p)
    {
        XMFLOAT3 d(
            p.position.x - center.x,
            p.position.y - center.y,
            p.position.z - center.z);
        float distSq = d.x * d.x + d.y * d.y + d.z * d.z;
        return distSq <= radiusSq;
    });
    m_particles.erase(endIt, m_particles.end());
}

void FluidSystem::QueueGather(const XMFLOAT3& target, float radius, float strength)
{
    m_gathers.push_back({ target, radius, strength });
}

void FluidSystem::QueueSplash(const XMFLOAT3& position, float radius, float impulse)
{
    m_splashes.push_back({ position, radius, impulse });
}

void FluidSystem::ClearDynamicOperations()
{
    m_gathers.clear();
    m_splashes.clear();
}

void FluidSystem::Simulate(ID3D12GraphicsCommandList* /*cmd*/, float dt)
{
    if (m_particles.empty())
    {
        return;
    }

    const float damping = 0.98f;
    const XMFLOAT3 gravity(0.0f, -9.8f, 0.0f);

    for (auto& particle : m_particles)
    {
        XMVECTOR pos = XMLoadFloat3(&particle.position);
        XMVECTOR vel = XMLoadFloat3(&particle.velocity);

        // 重力を適用
        vel += XMLoadFloat3(&gravity) * dt;

        // Gather 操作：指定位置へ引き寄せる
        for (const auto& gather : m_gathers)
        {
            XMVECTOR target = XMLoadFloat3(&gather.target);
            XMVECTOR dir = target - pos;
            float distance = XMVectorGetX(XMVector3Length(dir));
            if (distance < gather.radius && distance > 1e-4f)
            {
                XMVECTOR dirNorm = dir / distance;
                float influence = 1.0f - (distance / gather.radius);
                vel += dirNorm * (gather.strength * influence * dt);
            }
        }

        // Splash 操作：外向きに吹き飛ばす
        for (const auto& splash : m_splashes)
        {
            XMVECTOR origin = XMLoadFloat3(&splash.origin);
            XMVECTOR dir = pos - origin;
            float distance = XMVectorGetX(XMVector3Length(dir));
            if (distance < splash.radius && distance > 1e-4f)
            {
                XMVECTOR dirNorm = dir / distance;
                float influence = 1.0f - (distance / splash.radius);
                vel += dirNorm * (splash.impulse * influence * dt);
            }
        }

        // 速度の減衰を適用
        vel *= damping;

        // 位置を更新
        pos += vel * dt;

        XMStoreFloat3(&particle.velocity, vel);
        XMStoreFloat3(&particle.position, pos);

        // 領域外に出た場合は反射させる
        if (particle.position.x < m_boundsMin.x)
        {
            particle.position.x = m_boundsMin.x;
            particle.velocity.x *= -0.5f;
        }
        if (particle.position.x > m_boundsMax.x)
        {
            particle.position.x = m_boundsMax.x;
            particle.velocity.x *= -0.5f;
        }
        if (particle.position.y < m_boundsMin.y)
        {
            particle.position.y = m_boundsMin.y;
            particle.velocity.y *= -0.5f;
        }
        if (particle.position.y > m_boundsMax.y)
        {
            particle.position.y = m_boundsMax.y;
            particle.velocity.y *= -0.5f;
        }
        if (particle.position.z < m_boundsMin.z)
        {
            particle.position.z = m_boundsMin.z;
            particle.velocity.z *= -0.5f;
        }
        if (particle.position.z > m_boundsMax.z)
        {
            particle.position.z = m_boundsMax.z;
            particle.velocity.z *= -0.5f;
        }
    }

    m_elapsedTime += dt;
}

void FluidSystem::Render(
    ID3D12GraphicsCommandList* /*cmd*/,
    const XMFLOAT4X4& view,
    const XMFLOAT4X4& proj,
    const XMFLOAT4X4& viewProj,
    const XMFLOAT3& camPos,
    float isoLevel)
{
    m_isoLevel = isoLevel;

    // 粒子情報を GPU バッファへ転送
    UpdateParticleBuffer();

    if (!m_constantBuffer)
    {
        return;
    }

    MetaConstants* constants = m_constantBuffer->GetPtr<MetaConstants>();
    if (!constants)
    {
        return;
    }

    XMMATRIX viewMat = XMLoadFloat4x4(&view);
    XMMATRIX projMat = XMLoadFloat4x4(&proj);
    XMMATRIX viewProjMat = XMLoadFloat4x4(&viewProj);
    XMMATRIX invViewProj = XMMatrixInverse(nullptr, viewProjMat);

    XMStoreFloat4x4(&constants->InvViewProj, XMMatrixTranspose(invViewProj));
    XMStoreFloat4x4(&constants->ViewProj, XMMatrixTranspose(viewProjMat));
    constants->CamRadius = XMFLOAT4(camPos.x, camPos.y, camPos.z, m_material.renderRadius);
    constants->IsoCount = XMFLOAT4(m_isoLevel, static_cast<float>(m_particleCount), 1.0f, 0.0f);
    constants->GridMinCell = XMFLOAT4(0.0f, 0.0f, 0.0f, 0.0f); // グリッドは使用しない
    constants->GridDimInfo = XMUINT4(0, 0, 0, 0);
    constants->WaterDeep = XMFLOAT4(m_waterColorDeep.x, m_waterColorDeep.y, m_waterColorDeep.z, m_waterAbsorption);
    constants->WaterShallow = XMFLOAT4(m_waterColorShallow.x, m_waterColorShallow.y, m_waterColorShallow.z, m_foamThreshold);
    constants->ShadingParams = XMFLOAT4(m_foamStrength, m_reflectionStrength, m_specularPower, m_elapsedTime);
}

void FluidSystem::Composite(
    ID3D12GraphicsCommandList* cmd,
    ID3D12Resource* /*sceneColor*/,
    ID3D12Resource* /*sceneDepth*/,
    D3D12_CPU_DESCRIPTOR_HANDLE sceneRTV)
{
    if (!cmd || m_particleCount == 0 || !m_pipelineState || !m_rootSignature)
    {
        return;
    }

    // 描画対象の RTV をセット
    cmd->OMSetRenderTargets(1, &sceneRTV, FALSE, nullptr);

    ID3D12DescriptorHeap* heaps[] = { g_Engine->CbvSrvUavHeap()->GetHeap() };
    cmd->SetDescriptorHeaps(1, heaps);

    cmd->SetGraphicsRootSignature(m_rootSignature.Get());
    cmd->SetPipelineState(m_pipelineState.Get());

    if (m_particleSRV)
    {
        cmd->SetGraphicsRootDescriptorTable(0, m_particleSRV->HandleGPU);
    }
    if (m_constantBuffer)
    {
        cmd->SetGraphicsRootConstantBufferView(1, m_constantBuffer->GetAddress());
    }
    if (m_dummyGridTableSRV)
    {
        cmd->SetGraphicsRootDescriptorTable(2, m_dummyGridTableSRV->HandleGPU);
    }
    if (m_dummyGridCountSRV)
    {
        cmd->SetGraphicsRootDescriptorTable(3, m_dummyGridCountSRV->HandleGPU);
    }

    // フルスクリーン三角形でメタボールを描画
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd->DrawInstanced(3, 1, 0, 0);
}

void FluidSystem::SetWaterAppearance(
    const XMFLOAT3& shallowColor,
    const XMFLOAT3& deepColor,
    float absorption,
    float foamThreshold,
    float foamStrength,
    float reflectionStrength,
    float specularPower)
{
    m_waterColorShallow = shallowColor;
    m_waterColorDeep = deepColor;
    m_waterAbsorption = absorption;
    m_foamThreshold = foamThreshold;
    m_foamStrength = foamStrength;
    m_reflectionStrength = reflectionStrength;
    m_specularPower = specularPower;
}

bool FluidSystem::CreateMetaResources(ID3D12Device* device, DXGI_FORMAT rtvFormat)
{
    if (!device)
    {
        return false;
    }

    if (!graphics::MetaBallPipeline::CreateRootSignature(device, m_rootSignature))
    {
        return false;
    }

    if (!graphics::MetaBallPipeline::CreatePipelineState(device, m_rootSignature.Get(), rtvFormat, DXGI_FORMAT_UNKNOWN, m_pipelineState))
    {
        m_rootSignature.Reset();
        return false;
    }

    return true;
}

void FluidSystem::UpdateParticleBuffer()
{
    if (!m_particleMapped)
    {
        m_particleCount = 0;
        return;
    }

    MetaBallInstance* instances = reinterpret_cast<MetaBallInstance*>(m_particleMapped);
    size_t count = std::min(static_cast<size_t>(m_maxParticles), m_particles.size());
    for (size_t i = 0; i < count; ++i)
    {
        instances[i].position = m_particles[i].position;
        instances[i].radius = m_material.renderRadius;
    }
    for (size_t i = count; i < m_maxParticles; ++i)
    {
        instances[i].position = XMFLOAT3(0.0f, -1000.0f, 0.0f); // 画面に映らない位置へ退避
        instances[i].radius = 0.0f;
    }

    m_particleCount = static_cast<UINT>(count);
}
