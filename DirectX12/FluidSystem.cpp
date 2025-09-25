#include "FluidSystem.h"
#include "Engine.h"
#include "MetaBallPipelineState.h"
#include "RandomUtil.h"
#include <d3dx12.h>
#include <algorithm>
#include <cmath>

using namespace DirectX;

namespace
{
    // PBFで使用するカーネル関数（Poly6）
    float Poly6(float r, float h)
    {
        if (r >= h)
        {
            return 0.0f;
        }
        const float diff = h * h - r * r;
        const float coeff = 315.0f / (64.0f * XM_PI * std::pow(h, 9));
        return coeff * diff * diff * diff;
    }

    // PBFで使用するカーネル勾配（Spiky）
    XMFLOAT3 GradSpiky(const XMFLOAT3& rij, float r, float h)
    {
        if (r <= 1e-6f || r >= h)
        {
            return XMFLOAT3(0.0f, 0.0f, 0.0f);
        }
        const float coeff = -45.0f / (XM_PI * std::pow(h, 6));
        const float scale = coeff * (h - r) * (h - r) / r;
        return XMFLOAT3(rij.x * scale, rij.y * scale, rij.z * scale);
    }

    XMFLOAT3 Add(const XMFLOAT3& a, const XMFLOAT3& b)
    {
        return XMFLOAT3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    XMFLOAT3 Sub(const XMFLOAT3& a, const XMFLOAT3& b)
    {
        return XMFLOAT3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    XMFLOAT3 Mul(const XMFLOAT3& v, float s)
    {
        return XMFLOAT3(v.x * s, v.y * s, v.z * s);
    }

    float Dot(const XMFLOAT3& a, const XMFLOAT3& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    float Length(const XMFLOAT3& v)
    {
        return std::sqrt(Dot(v, v));
    }

    // GPUへ渡す粒子メタデータ構造体
    struct ParticleMetaGPU
    {
        XMFLOAT3 position; // ワールド座標
        float radius;      // レイマーチ半径
    };
}

void FluidSystem::Init(ID3D12Device* device, DXGI_FORMAT rtvFormat, UINT maxParticles, UINT threadGroupCount)
{
    (void)device;
    (void)rtvFormat;
    (void)threadGroupCount;
    // GPU版は未実装なのでCPUで扱える粒子数に制限する
    m_maxParticles = std::min<UINT>(maxParticles, 512);

    m_cpuParticles.resize(m_maxParticles);
    m_particleCount = m_maxParticles;

    // 粒子初期配置（簡易的に立方体内へランダム配置）
    for (UINT i = 0; i < m_maxParticles; ++i)
    {
        FluidParticle& p = m_cpuParticles[i];
        p.position = XMFLOAT3(RandFloat(-0.5f, 0.5f), RandFloat(0.0f, 1.0f), RandFloat(-0.5f, 0.5f));
        p.velocity = XMFLOAT3(0.0f, 0.0f, 0.0f);
        p.x_pred = p.position;
        p.lambda = 0.0f;
        p.density = m_restDensity;
    }

    // 粒子メタデータを格納するアップロードバッファを生成
    const UINT64 bufferSize = sizeof(ParticleMetaGPU) * m_maxParticles;
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
    if (FAILED(device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE,
            &bufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(m_particleMetaBuffer.ReleaseAndGetAddressOf()))))
    {
        printf("FluidSystem: 粒子メタデータ用バッファ生成に失敗しました\n");
        return;
    }

    // SRVをディスクリプタヒープへ登録（シェーダーから粒子配列を参照するため）
    m_particleSRV = g_Engine->CbvSrvUavHeap()->RegisterBuffer(
        m_particleMetaBuffer.Get(), m_maxParticles, sizeof(ParticleMetaGPU));
    if (!m_particleSRV)
    {
        printf("FluidSystem: 粒子SRVの登録に失敗しました\n");
        return;
    }

    // メタボール描画用のルートシグネチャとPSOを生成
    graphics::MetaBallPipeline::CreateRootSignature(device, m_metaRootSignature);
    if (!m_metaRootSignature)
    {
        printf("FluidSystem: MetaBall用RootSignature生成に失敗しました\n");
        return;
    }

    graphics::MetaBallPipeline::CreatePipelineState(device, m_metaRootSignature.Get(), rtvFormat, m_metaPipelineState);
    if (!m_metaPipelineState)
    {
        printf("FluidSystem: MetaBall用PSO生成に失敗しました\n");
        return;
    }

    // メタボール描画に必要な定数バッファ（ダブルバッファリング）
    for (UINT i = 0; i < kFrameCount; ++i)
    {
        m_metaCB[i] = std::make_unique<ConstantBuffer>(sizeof(MetaConstants));
        if (!m_metaCB[i] || !m_metaCB[i]->IsValid())
        {
            printf("FluidSystem: MetaBall定数バッファ生成に失敗しました\n");
            return;
        }
    }

    // 初期データをGPUバッファへコピー
    UpdateParticleBuffer();

    m_initialized = true;
}

void FluidSystem::StepCPU(float dt)
{
    if (m_cpuParticles.empty())
    {
        return;
    }

    const float h = m_smoothingRadius;
    const float mass = m_particleMass;
    const float restDensity = m_restDensity;
    const float epsilon = m_epsilon;
    const float sCorrK = -0.001f;
    const float sCorrN = 4.0f;
    const float deltaQ = 0.3f * h;
    const float invRestDensity = 1.0f / restDensity;

    // 1. 外力適用と予測位置更新
    for (auto& p : m_cpuParticles)
    {
        p.velocity = Add(p.velocity, Mul(m_gravity, dt));
        p.x_pred = Add(p.position, Mul(p.velocity, dt));
        p.lambda = 0.0f;
    }

    // 2. コンストレイントソルバ
    for (int iter = 0; iter < m_solverIterations; ++iter)
    {
        // 密度とλの計算
        for (size_t i = 0; i < m_cpuParticles.size(); ++i)
        {
            auto& pi = m_cpuParticles[i];
            float density = 0.0f;
            for (size_t j = 0; j < m_cpuParticles.size(); ++j)
            {
                const auto& pj = m_cpuParticles[j];
                const XMFLOAT3 rij = Sub(pi.x_pred, pj.x_pred);
                const float r = Length(rij);
                density += mass * Poly6(r, h);
            }
            pi.density = (std::max)(density, restDensity * 0.1f);
            const float Ci = pi.density * invRestDensity - 1.0f;

            XMFLOAT3 gradSum = XMFLOAT3(0.0f, 0.0f, 0.0f);
            float sumGrad2 = 0.0f;
            for (size_t j = 0; j < m_cpuParticles.size(); ++j)
            {
                if (i == j)
                {
                    continue;
                }
                const auto& pj = m_cpuParticles[j];
                const XMFLOAT3 rij = Sub(pi.x_pred, pj.x_pred);
                const float r = Length(rij);
                XMFLOAT3 grad = GradSpiky(rij, r, h);
                grad = Mul(grad, mass * invRestDensity);
                gradSum = Add(gradSum, grad);
                sumGrad2 += Dot(grad, grad);
            }
            sumGrad2 += Dot(gradSum, gradSum);

            pi.lambda = Ci / (sumGrad2 + epsilon);
        }

        // Δpの計算
        for (size_t i = 0; i < m_cpuParticles.size(); ++i)
        {
            auto& pi = m_cpuParticles[i];
            XMFLOAT3 delta = XMFLOAT3(0.0f, 0.0f, 0.0f);
            for (size_t j = 0; j < m_cpuParticles.size(); ++j)
            {
                if (i == j)
                {
                    continue;
                }
                const auto& pj = m_cpuParticles[j];
                const XMFLOAT3 rij = Sub(pi.x_pred, pj.x_pred);
                const float r = Length(rij);
                if (r >= h)
                {
                    continue;
                }
                const float w = Poly6(r, h);
                float corr = 0.0f;
                const float wq = Poly6(deltaQ, h);
                if (wq > 0.0f)
                {
                    corr = sCorrK * std::pow(w / wq, sCorrN);
                }
                XMFLOAT3 grad = GradSpiky(rij, r, h);
                const float factor = (pi.lambda + pj.lambda + corr) * mass * invRestDensity;
                delta = Add(delta, Mul(grad, factor));
            }
            pi.x_pred = Add(pi.x_pred, delta);

            // 床（y=0）との衝突処理。沈み込み防止のため軽く押し戻す
            if (pi.x_pred.y < 0.0f)
            {
                pi.x_pred.y = 0.0f;
            }

            // 簡易的な境界（立方体）
            pi.x_pred.x = std::clamp(pi.x_pred.x, -1.0f, 1.0f);
            pi.x_pred.y = std::clamp(pi.x_pred.y, 0.0f, 2.0f);
            pi.x_pred.z = std::clamp(pi.x_pred.z, -1.0f, 1.0f);
        }
    }

    // 3. 速度と位置の更新
    for (auto& p : m_cpuParticles)
    {
        const XMFLOAT3 delta = Sub(p.x_pred, p.position);
        p.velocity = Mul(delta, 1.0f / dt);
        p.position = p.x_pred;
    }
}

void FluidSystem::UpdateParticleBuffer()
{
    if (!m_particleMetaBuffer)
    {
        return;
    }

    const UINT cpuCount = static_cast<UINT>(m_cpuParticles.size());
    const UINT count = (std::min)(cpuCount, m_maxParticles);
    m_particleCount = count;

    ParticleMetaGPU* mapped = nullptr;
    if (FAILED(m_particleMetaBuffer->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) || !mapped)
    {
        printf("FluidSystem : particle data map failed\n");
        return;
    }

    // 使用している粒子をGPUバッファへ書き戻す
    for (UINT i = 0; i < count; ++i)
    {
        mapped[i].position = m_cpuParticles[i].position;
        mapped[i].radius = m_renderRadius;
    }

    // 余剰領域は半径0で埋めて寄与を無効化しておく
    for (UINT i = count; i < m_maxParticles; ++i)
    {
        mapped[i].position = XMFLOAT3(0.0f, 0.0f, 0.0f);
        mapped[i].radius = 0.0f;
    }

    m_particleMetaBuffer->Unmap(0, nullptr);
}

void FluidSystem::Simulate(ID3D12GraphicsCommandList*, float dt)
{
    if (!m_initialized)
    {
        return;
    }
    const float clampedDt = (std::max)(dt, 1.0f / 240.0f); // 極端に小さいdtを避ける
    StepCPU(clampedDt);
    UpdateParticleBuffer();
}

void FluidSystem::Render(ID3D12GraphicsCommandList* cmd,
    const XMFLOAT4X4& invViewProj, const XMFLOAT3& camPos, float isoLevel)
{
    const UINT count = (std::min)(m_particleCount, m_maxParticles);

    if (!m_initialized || !cmd || count == 0 || !m_particleSRV)
    {
        return;
    }

    const UINT frameIndex = g_Engine->CurrentBackBufferIndex();
    auto* cb = m_metaCB[frameIndex]->GetPtr<MetaConstants>();

    // HLSL側は列優先行列を前提としているため、逆ビュー射影行列を転置してから書き込む
    DirectX::XMMATRIX invVP = DirectX::XMLoadFloat4x4(&invViewProj);
    invVP = DirectX::XMMatrixTranspose(invVP);
    DirectX::XMStoreFloat4x4(&cb->InvViewProj, invVP);

    cb->CamRadius = XMFLOAT4(camPos.x, camPos.y, camPos.z, m_renderRadius);
    // ガウスカーネルの特性上しきい値は1未満で扱いやすいので少しスケールダウン
    cb->IsoCount = XMFLOAT4(isoLevel * 0.6f, static_cast<float>(count), m_rayStepScale, 0.0f);

    // SRVテーブルを参照できるようにディスクリプタヒープをセット
    ID3D12DescriptorHeap* heaps[] = { g_Engine->CbvSrvUavHeap()->GetHeap() };
    cmd->SetDescriptorHeaps(1, heaps);

    cmd->SetGraphicsRootSignature(m_metaRootSignature.Get());
    cmd->SetPipelineState(m_metaPipelineState.Get());
    cmd->SetGraphicsRootDescriptorTable(0, m_particleSRV->HandleGPU);
    cmd->SetGraphicsRootConstantBufferView(1, m_metaCB[frameIndex]->GetAddress());

    // フルスクリーン三角形を描画してメタボールのスクリーンスペースレイマーチを実行
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd->DrawInstanced(3, 1, 0, 0);
}
