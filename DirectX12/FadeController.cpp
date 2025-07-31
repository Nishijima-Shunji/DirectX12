#include "FadeController.h"
#include "Engine.h"

using namespace DirectX;

FadeController::FadeController()
{
    Init();
}

bool FadeController::Init()
{
    struct Vertex { XMFLOAT3 pos; };
    Vertex vertices[] = {
        { {-1.f, -1.f, 0.f} },
        { {-1.f,  1.f, 0.f} },
        { { 1.f, -1.f, 0.f} },
        { { 1.f,  1.f, 0.f} },
    };
    m_vertexBuffer = std::make_unique<VertexBuffer>(sizeof(vertices), sizeof(Vertex), vertices);
    if (!m_vertexBuffer->IsValid()) return false;

    for (size_t i = 0; i < Engine::FRAME_BUFFER_COUNT; ++i) {
        m_constantBuffers[i] = std::make_unique<ConstantBuffer>(sizeof(XMFLOAT4));
        if (!m_constantBuffers[i]->IsValid()) return false;
    }

    // simple root signature with one constant buffer
    CD3DX12_ROOT_PARAMETER param; 
    param.InitAsConstantBufferView(0);
    D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
    rsDesc.NumParameters = 1;
    rsDesc.pParameters = &param;
    rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    m_rootSignature = std::make_unique<RootSignature>();
    if (!m_rootSignature->Init(rsDesc)) return false;

    m_pipelineState = std::make_unique<PipelineState>();
    D3D12_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
    };
    D3D12_INPUT_LAYOUT_DESC layoutDesc{ layout, 1 };
    m_pipelineState->SetInputLayout(layoutDesc);
    m_pipelineState->SetRootSignature(m_rootSignature->Get());
    m_pipelineState->SetVS(L"FadeVS.cso");
    m_pipelineState->SetPS(L"FadePS.cso");
    m_pipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLESTRIP);
    return m_pipelineState->IsValid();
}

void FadeController::StartFadeOut(float duration)
{
    m_fadeOut = true;
    m_active = true;
    m_duration = duration;
    m_timer = 0.0f;
    m_alpha = 0.0f;
}

void FadeController::StartFadeIn(float duration)
{
    m_fadeOut = false;
    m_active = true;
    m_duration = duration;
    m_timer = 0.0f;
    m_alpha = 1.0f;
}

void FadeController::Update(float dt)
{
    if (!m_active) return;
    m_timer += dt;
    float t = (m_duration > 0.0f) ? m_timer / m_duration : 1.0f;
    if (t >= 1.0f) {
        t = 1.0f;
        m_active = false;
    }
    if (m_fadeOut) {
        m_alpha = t;
    } else {
        m_alpha = 1.0f - t;
    }
}

void FadeController::Render()
{
    if (m_alpha <= 0.0f) return;
    auto cmd = g_Engine->CommandList();
    UINT idx = g_Engine->CurrentBackBufferIndex();
    auto colorPtr = m_constantBuffers[idx]->GetPtr<XMFLOAT4>();
    *colorPtr = XMFLOAT4(0.f, 0.f, 0.f, m_alpha);

    cmd->SetGraphicsRootSignature(m_rootSignature->Get());
    cmd->SetPipelineState(m_pipelineState->Get());
    cmd->SetGraphicsRootConstantBufferView(0, m_constantBuffers[idx]->GetAddress());

    auto vbv = m_vertexBuffer->View();
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    cmd->IASetVertexBuffers(0, 1, &vbv);
    cmd->DrawInstanced(4, 1, 0, 0);
}

