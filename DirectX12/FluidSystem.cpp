#include "FluidSystem.h"
#include "Engine.h"
#include "Camera.h"
#include "RandomUtil.h"
#include <algorithm>
#include <cmath>
#include <cstring>

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
}

void FluidSystem::Init(ID3D12Device* device, DXGI_FORMAT rtvFormat, UINT maxParticles, UINT threadGroupCount)
{
    (void)device;
    (void)rtvFormat;
    (void)threadGroupCount;
    // GPU版は未実装なのでCPUで扱える粒子数に制限する
    m_maxParticles = std::min<UINT>(maxParticles, 512);

    m_cpuParticles.resize(m_maxParticles);
    m_cpuVertices.resize(m_maxParticles);

    // 粒子初期配置（簡易的に立方体内へランダム配置）
    for (UINT i = 0; i < m_maxParticles; ++i)
    {
        FluidParticle& p = m_cpuParticles[i];
        p.position = XMFLOAT3(RandFloat(-0.5f, 0.5f), RandFloat(0.0f, 1.0f), RandFloat(-0.5f, 0.5f));
        p.velocity = XMFLOAT3(0.0f, 0.0f, 0.0f);
        p.x_pred = p.position;
        p.lambda = 0.0f;
        p.density = m_restDensity;
        m_cpuVertices[i].position = p.position;
    }

    // 頂点バッファをアップロードヒープに作成
    m_vertexBuffer = std::make_unique<VertexBuffer>(sizeof(ParticleVertex) * m_maxParticles,
        sizeof(ParticleVertex), m_cpuVertices.data());
    if (!m_vertexBuffer || !m_vertexBuffer->IsValid())
    {
        printf("FluidSystem: 頂点バッファ生成に失敗しました\n");
        return;
    }

    // ルートシグネチャとPSOを生成（ポイント描画）
    m_rootSignature = std::make_unique<RootSignature>();
    if (!m_rootSignature || !m_rootSignature->IsValid())
    {
        printf("FluidSystem: RootSignature生成に失敗しました\n");
        return;
    }

    m_pipelineState = std::make_unique<PipelineState>();
    m_pipelineState->SetInputLayout(ParticleVertex::InputLayout);
    m_pipelineState->SetRootSignature(m_rootSignature->Get());
    m_pipelineState->SetVS(L"ParticleVS.cso");
    m_pipelineState->SetPS(L"ParticlePS.cso");
    m_pipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT);
    if (!m_pipelineState || !m_pipelineState->IsValid())
    {
        printf("FluidSystem: PSO生成に失敗しました\n");
        return;
    }

    // Transform用定数バッファ（フレーム数分）
    for (UINT i = 0; i < kFrameCount; ++i)
    {
        m_transformCB[i] = std::make_unique<ConstantBuffer>(sizeof(Transform));
        if (!m_transformCB[i] || !m_transformCB[i]->IsValid())
        {
            printf("FluidSystem: ConstantBuffer生成に失敗しました\n");
            return;
        }

        auto* cb = m_transformCB[i]->GetPtr<Transform>();
        cb->World = XMMatrixIdentity();
        cb->View = XMMatrixIdentity();
        cb->Proj = XMMatrixIdentity();
    }

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
            pi.density = std::max(density, restDensity * 0.1f);
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

            pi.lambda = -Ci / (sumGrad2 + epsilon);
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

void FluidSystem::UpdateVertexBuffer()
{
    if (!m_vertexBuffer)
    {
        return;
    }

    for (UINT i = 0; i < m_maxParticles; ++i)
    {
        m_cpuVertices[i].position = m_cpuParticles[i].position;
    }

    void* mapped = nullptr;
    if (FAILED(m_vertexBuffer->GetResource()->Map(0, nullptr, &mapped)))
    {
        printf("FluidSystem: 頂点バッファのマップに失敗しました\n");
        return;
    }
    std::memcpy(mapped, m_cpuVertices.data(), sizeof(ParticleVertex) * m_maxParticles);
    m_vertexBuffer->GetResource()->Unmap(0, nullptr);
}

void FluidSystem::Simulate(ID3D12GraphicsCommandList*, float dt)
{
    if (!m_initialized)
    {
        return;
    }
    const float clampedDt = std::max(dt, 1.0f / 240.0f); // 極端に小さいdtを避ける
    StepCPU(clampedDt);
    UpdateVertexBuffer();
}

void FluidSystem::Render(ID3D12GraphicsCommandList* cmd,
    const XMFLOAT4X4&, const XMFLOAT3&, float)
{
    if (!m_initialized || !cmd)
    {
        return;
    }

    auto* camera = g_Engine->GetObj<Camera>("Camera");
    if (!camera)
    {
        return;
    }

    const UINT frameIndex = g_Engine->CurrentBackBufferIndex();
    auto* cb = m_transformCB[frameIndex]->GetPtr<Transform>();
    cb->World = XMMatrixIdentity();
    cb->View = camera->GetViewMatrix();
    cb->Proj = camera->GetProjMatrix();

    cmd->SetGraphicsRootSignature(m_rootSignature->Get());
    cmd->SetPipelineState(m_pipelineState->Get());
    cmd->SetGraphicsRootConstantBufferView(0, m_transformCB[frameIndex]->GetAddress());

    auto vbView = m_vertexBuffer->View();
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
    cmd->IASetVertexBuffers(0, 1, &vbView);
    cmd->DrawInstanced(m_maxParticles, 1, 0, 0);
}
