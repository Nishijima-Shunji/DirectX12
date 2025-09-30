#include "MetaballRenderer.h"

#include <cmath>
#include <random>
#include "ConstantBuffer.h"
#include "DescriptorHeap.h"
#include "Engine.h"
#include "MetaBallPipelineState.h"
#include <d3dx12.h>

using namespace DirectX;

namespace
{
    constexpr UINT kParticleCount = 100;          // 描画する粒子数
    constexpr float kDefaultRadius = 0.35f;      // 各粒子の半径
    constexpr float kIsoThreshold = 0.85f;       // メタボールのアイソ値
    constexpr float kStepScale = 0.45f;          // レイマーチングのステップ係数
    constexpr float kMaxMarchDistance = 20.0f;   // レイが進む最大距離
}

bool MetaballRenderer::Initialize()
{
    // 粒子のシードを決定
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> angleDist(0.0f, XM_2PI);
    std::uniform_real_distribution<float> radialDist(0.8f, 1.8f);
    std::uniform_real_distribution<float> phaseDist(0.0f, XM_2PI);

    m_particles.clear();
    m_particles.reserve(kParticleCount);

    for (UINT i = 0; i < kParticleCount; ++i)
    {
        ParticleState state{};
        state.position = XMFLOAT3(0.0f, 0.0f, 0.0f);
        state.radius = kDefaultRadius;
        state.angleSeed = angleDist(rng);
        state.radialSeed = radialDist(rng);
        state.heightSeed = phaseDist(rng);
        m_particles.push_back(state);
    }

    if (!CreatePipeline())
    {
        return false;
    }

    if (!CreateBuffers())
    {
        return false;
    }

    UpdateParticleBuffer();
    UpdateConstantBuffer();
    return true;
}

void MetaballRenderer::Update(float deltaTime)
{
    m_time += deltaTime;

    // 粒子の配置を簡単な周期運動で更新する
    for (size_t i = 0; i < m_particles.size(); ++i)
    {
        auto& p = m_particles[i];
        float angle = p.angleSeed + m_time * (0.6f + 0.2f * std::sin(p.radialSeed));
        float ringRadius = p.radialSeed + 0.3f * std::sin(m_time * 0.7f + p.angleSeed);
        float height = 0.7f * std::sin(m_time * 0.9f + p.heightSeed);

        p.position.x = std::cos(angle) * ringRadius;
        p.position.y = height;
        p.position.z = std::sin(angle) * ringRadius;
    }

    UpdateParticleBuffer();
    UpdateConstantBuffer();
}

void MetaballRenderer::Render()
{
    auto commandList = g_Engine->CommandList();
    if (!commandList || !m_pipelineState || !m_rootSignature)
    {
        return;
    }

    ID3D12DescriptorHeap* heaps[] = { g_Engine->CbvSrvUavHeap()->GetHeap() };
    commandList->SetDescriptorHeaps(1, heaps);

    commandList->SetGraphicsRootSignature(m_rootSignature.Get());
    commandList->SetPipelineState(m_pipelineState.Get());

    commandList->SetGraphicsRootDescriptorTable(0, m_particleSrv->HandleGPU);
    commandList->SetGraphicsRootConstantBufferView(1, m_constantBuffer->GetAddress());

    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    commandList->DrawInstanced(3, 1, 0, 0);
}

bool MetaballRenderer::CreatePipeline()
{
    auto device = g_Engine->Device();
    if (!device)
    {
        return false;
    }

    if (!graphics::MetaBallPipeline::CreateRootSignature(device, m_rootSignature))
    {
        return false;
    }

    if (!graphics::MetaBallPipeline::CreatePipelineState(
            device,
            m_rootSignature.Get(),
            DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_D32_FLOAT,
            m_pipelineState))
    {
        return false;
    }

    return true;
}

bool MetaballRenderer::CreateBuffers()
{
    auto device = g_Engine->Device();
    if (!device)
    {
        return false;
    }

    m_constantBuffer = std::make_unique<ConstantBuffer>(sizeof(MetaConstants));
    if (!m_constantBuffer || !m_constantBuffer->IsValid())
    {
        return false;
    }

    auto particleBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(ParticleGPU) * kParticleCount);
    auto uploadHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

    if (FAILED(device->CreateCommittedResource(
            &uploadHeap,
            D3D12_HEAP_FLAG_NONE,
            &particleBufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(m_particleBuffer.ReleaseAndGetAddressOf()))))
    {
        return false;
    }

    if (FAILED(m_particleBuffer->Map(0, nullptr, reinterpret_cast<void**>(&m_mappedParticles))))
    {
        return false;
    }

    m_particleSrv = g_Engine->CbvSrvUavHeap()->RegisterBuffer(
        m_particleBuffer.Get(),
        kParticleCount,
        sizeof(ParticleGPU));

    return m_particleSrv != nullptr;
}

void MetaballRenderer::UpdateConstantBuffer()
{
    if (!m_constantBuffer)
    {
        return;
    }

    auto constants = m_constantBuffer->GetPtr<MetaConstants>();

    const float aspect = static_cast<float>(g_Engine->FrameBufferWidth()) /
                         static_cast<float>(g_Engine->FrameBufferHeight());
    const float cameraRadius = 6.0f;
    XMVECTOR eye = XMVectorSet(std::sin(m_time * 0.25f) * cameraRadius, 3.0f, std::cos(m_time * 0.25f) * cameraRadius, 0.0f);
    XMVECTOR target = XMVectorZero();
    XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

    XMMATRIX view = XMMatrixLookAtRH(eye, target, up);
    XMMATRIX proj = XMMatrixPerspectiveFovRH(XMConvertToRadians(55.0f), aspect, 0.1f, 100.0f);
    XMMATRIX viewProj = XMMatrixMultiply(view, proj);
    XMMATRIX invViewProj = XMMatrixInverse(nullptr, viewProj);

    XMStoreFloat4x4(&constants->invViewProj, XMMatrixTranspose(invViewProj));
    XMStoreFloat4x4(&constants->viewProj, XMMatrixTranspose(viewProj));

    XMFLOAT3 cameraPos{};
    XMStoreFloat3(&cameraPos, eye);
    constants->cameraIso = XMFLOAT4(cameraPos.x, cameraPos.y, cameraPos.z, kIsoThreshold);
    constants->params = XMFLOAT4(kStepScale, kMaxMarchDistance, static_cast<float>(m_particles.size()), m_time);
}

void MetaballRenderer::UpdateParticleBuffer()
{
    if (!m_mappedParticles)
    {
        return;
    }

    for (size_t i = 0; i < m_particles.size(); ++i)
    {
        const auto& p = m_particles[i];
        m_mappedParticles[i].data = XMFLOAT4(p.position.x, p.position.y, p.position.z, p.radius);
    }
}
