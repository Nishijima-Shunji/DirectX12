#include "GameScene.h"
#include "Game.h"
#include "Engine.h" // エンジンの各種リソースへアクセスするためのヘッダー
#include <random>
#include <cwchar>
#include <cmath>
#include <cstring>
#include <windows.h>

using namespace DirectX;
using Microsoft::WRL::ComPtr;

GameScene* GameScene::g_pCurrentScene = nullptr;

GameScene::GameScene(Game* game) : BaseScene(game)
{
    if (g_pCurrentScene == this)
    {
        g_pCurrentScene = nullptr;
    }

    Init();
}

GameScene::~GameScene()
{
    if (m_particleBuffer && m_particleMapped)
    {
        m_particleBuffer->Unmap(0, nullptr);
        m_particleMapped = nullptr;
    }

    if (g_pCurrentScene == this)
    {
        g_pCurrentScene = nullptr;
    }
}

bool GameScene::Init()
{
    g_pCurrentScene = this;

    // ===== カメラの初期化 =====
    Camera* camera = new Camera();
    camera->Init();
    g_Engine->RegisterObj<Camera>("Camera", camera);

    // ===== メタボール描画用のリソース生成 =====
    if (!CreateMetaPipeline())
    {
        return false;
    }

    m_metaConstantBuffer = std::make_unique<ConstantBuffer>(sizeof(MetaCB_CPU));
    if (!m_metaConstantBuffer || !m_metaConstantBuffer->IsValid())
    {
        return false;
    }

    // 粒子の初期値を設定（可視性を優先したプリセット）
    m_radius = 0.12f;
    m_iso = 0.18f;
    m_step = 2.5f;

    const UINT particleCount = 50;
    m_particles.resize(particleCount);
    m_velocities.resize(particleCount, XMFLOAT3(0.0f, 0.0f, 0.0f));

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);

    for (UINT i = 0; i < particleCount; ++i)
    {
        XMFLOAT3 pos(dist(rng), 0.25f + dist(rng), dist(rng));
        m_particles[i].pos = pos;
        m_particles[i].r = m_radius;
    }

    if (!CreateParticleBuffer(particleCount))
    {
        return false;
    }

    UpdateParticleBufferGPU();
    WriteMetaCB();

    return true;
}

void GameScene::Update(float deltaTime)
{
    if (auto camera = g_Engine->GetObj<Camera>("Camera"))
    {
        camera->Update(deltaTime);
    }

    // ===== 粒子の簡易物理 =====
    const float gravity = -3.0f;       // 重力加速度（やや強め）
    const float damping = 0.65f;       // 壁反射時の減衰
    const float boundMin = -1.2f;
    const float boundMax = 1.2f;

    for (size_t i = 0; i < m_particles.size(); ++i)
    {
        XMFLOAT3& pos = m_particles[i].pos;
        XMFLOAT3& vel = m_velocities[i];

        // 重力加速度を加算
        vel.y += gravity * deltaTime;

        // 位置更新
        pos.x += vel.x * deltaTime;
        pos.y += vel.y * deltaTime;
        pos.z += vel.z * deltaTime;

        // サイン波による緩やかな横揺れを付与して水っぽさを演出
        float swirl = 0.25f * std::sin(m_time + static_cast<float>(i) * 0.1f);
        float ripple = 0.18f * std::cos(m_time * 0.7f + static_cast<float>(i) * 0.17f);
        vel.x += swirl * 0.2f * deltaTime;
        vel.z += ripple * 0.15f * deltaTime;
        pos.x += swirl * deltaTime * 0.5f;
        pos.z += ripple * deltaTime * 0.3f;

        // 速度に軽い減衰をかけて暴れすぎないように制御
        vel.x *= 0.995f;
        vel.y *= 0.999f;
        vel.z *= 0.995f;

        // 境界チェック（跳ね返り＋減衰）
        auto clampAxis = [&](float& value, float& velocity)
        {
            if (value < boundMin)
            {
                value = boundMin;
                velocity = std::abs(velocity) * damping;
            }
            else if (value > boundMax)
            {
                value = boundMax;
                velocity = -std::abs(velocity) * damping;
            }
        };

        clampAxis(pos.x, vel.x);
        clampAxis(pos.y, vel.y);
        clampAxis(pos.z, vel.z);
    }

    m_time += deltaTime;

    UpdateParticleBufferGPU();
    WriteMetaCB();

    // ===== FPS 表示更新 =====
    ++m_fpsFrameCount;
    m_fpsTimer += deltaTime;
    if (m_fpsTimer >= 1.0f)
    {
        float fps = static_cast<float>(m_fpsFrameCount) / m_fpsTimer;
        wchar_t title[128] = {};
        swprintf_s(title, L"DirectX12 | FPS: %.1f", fps);
        if (HWND hwnd = GetActiveWindow())
        {
            SetWindowTextW(hwnd, title);
        }
        m_fpsFrameCount = 0;
        m_fpsTimer = 0.0f;
    }
}

void GameScene::Draw()
{
    ID3D12GraphicsCommandList* cmd = g_Engine->CommandList();
    if (!cmd || !m_metaPipelineState || !m_metaRootSignature)
    {
        return;
    }

    ID3D12DescriptorHeap* heaps[] = { g_Engine->CbvSrvUavHeap()->GetHeap() };
    cmd->SetDescriptorHeaps(1, heaps);
    cmd->SetGraphicsRootSignature(m_metaRootSignature.Get());
    cmd->SetPipelineState(m_metaPipelineState.Get());

    if (m_particleSRV)
    {
        cmd->SetGraphicsRootDescriptorTable(0, m_particleSRV->HandleGPU);
    }
    if (m_metaConstantBuffer)
    {
        cmd->SetGraphicsRootConstantBufferView(1, m_metaConstantBuffer->GetAddress());
    }

    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd->DrawInstanced(3, 1, 0, 0);
}

bool GameScene::CreateMetaPipeline()
{
    ID3D12Device* device = g_Engine->Device();
    if (!device)
    {
        return false;
    }

    if (!graphics::MetaBallPipeline::CreateRootSignature(device, m_metaRootSignature))
    {
        return false;
    }

    if (!graphics::MetaBallPipeline::CreatePipelineState(
            device,
            m_metaRootSignature.Get(),
            DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_D32_FLOAT,
            m_metaPipelineState))
    {
        return false;
    }

    return true;
}

bool GameScene::CreateParticleBuffer(UINT count)
{
    ID3D12Device* device = g_Engine->Device();
    if (!device || count == 0)
    {
        return false;
    }

    m_particleCapacity = count;

    UINT64 bufferSize = sizeof(ParticleMeta) * static_cast<UINT64>(count);
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto desc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);

    HRESULT hr = device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(m_particleBuffer.ReleaseAndGetAddressOf()));
    if (FAILED(hr))
    {
        return false;
    }

    hr = m_particleBuffer->Map(0, nullptr, reinterpret_cast<void**>(&m_particleMapped));
    if (FAILED(hr) || !m_particleMapped)
    {
        m_particleMapped = nullptr;
        return false;
    }

    if (!m_particleSRV)
    {
        m_particleSRV = g_Engine->CbvSrvUavHeap()->RegisterBuffer(
            m_particleBuffer.Get(), count, sizeof(ParticleMeta));
    }
    else
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Buffer.NumElements = count;
        srvDesc.Buffer.StructureByteStride = sizeof(ParticleMeta);
        g_Engine->Device()->CreateShaderResourceView(
            m_particleBuffer.Get(), &srvDesc, m_particleSRV->HandleCPU);
    }

    return m_particleSRV != nullptr;
}

void GameScene::UpdateParticleBufferGPU()
{
    if (!m_particleMapped || m_particles.empty())
    {
        return;
    }

    std::memcpy(
        m_particleMapped,
        m_particles.data(),
        sizeof(ParticleMeta) * m_particles.size());
}

void GameScene::WriteMetaCB()
{
    Camera* camera = g_Engine->GetObj<Camera>("Camera");
    if (!camera || !m_metaConstantBuffer)
    {
        return;
    }

    XMMATRIX view = camera->GetViewMatrix();
    XMMATRIX proj = camera->GetProjMatrix();
    XMMATRIX viewProj = XMMatrixMultiply(view, proj);
    XMMATRIX invViewProj = XMMatrixInverse(nullptr, viewProj);

    XMMATRIX viewProjT = XMMatrixTranspose(viewProj);
    XMMATRIX invViewProjT = XMMatrixTranspose(invViewProj);

    auto ptr = m_metaConstantBuffer->GetPtr<MetaCB_CPU>();
    if (!ptr)
    {
        return;
    }

    XMStoreFloat4x4(&ptr->viewProj, viewProjT);
    XMStoreFloat4x4(&ptr->invViewProj, invViewProjT);

    XMFLOAT3 camPos = camera->GetPosition();
    ptr->camRadius = XMFLOAT4(camPos.x, camPos.y, camPos.z, m_radius);
    ptr->isoCount = XMFLOAT4(m_iso, static_cast<float>(m_particles.size()), m_step, 0.0f);
    ptr->gridMinCell = XMFLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
    ptr->gridDimX = ptr->gridDimY = ptr->gridDimZ = ptr->totalCells = 0;
    ptr->waterDeep = XMFLOAT4(0.10f, 0.20f, 0.90f, 0.35f);
    ptr->waterShallow = XMFLOAT4(0.50f, 0.85f, 1.00f, 0.25f);
    ptr->shadingParams = XMFLOAT4(0.35f, 0.25f, 64.0f, m_time);
}

