#include "TestCube.h"

#include <vector>

using namespace DirectX;

TestCube::TestCube() {
    Init();
}

TestCube::~TestCube() {
    delete m_vertexBuffer;
    delete m_indexBuffer;
    for (auto& cb : m_constantBuffer) {
        delete cb;
    }
    delete m_rootSignature;
    delete m_pipelineState;
}

bool TestCube::Init() {
    // 立方体の頂点データ（位置・法線・UV・色）
    std::vector<Vertex> vertices = {
        // +X 面
        {{ 0.5f,-0.5f,-0.5f},{ 1,0,0},{0,1},{0,0,0},{1,1,1,1}},
        {{ 0.5f, 0.5f,-0.5f},{ 1,0,0},{0,0},{0,0,0},{1,1,1,1}},
        {{ 0.5f, 0.5f, 0.5f},{ 1,0,0},{1,0},{0,0,0},{1,1,1,1}},
        {{ 0.5f,-0.5f, 0.5f},{ 1,0,0},{1,1},{0,0,0},{1,1,1,1}},
        // -X 面
        {{-0.5f,-0.5f, 0.5f},{-1,0,0},{0,1},{0,0,0},{1,1,1,1}},
        {{-0.5f, 0.5f, 0.5f},{-1,0,0},{0,0},{0,0,0},{1,1,1,1}},
        {{-0.5f, 0.5f,-0.5f},{-1,0,0},{1,0},{0,0,0},{1,1,1,1}},
        {{-0.5f,-0.5f,-0.5f},{-1,0,0},{1,1},{0,0,0},{1,1,1,1}},
        // +Y 面
        {{-0.5f, 0.5f,-0.5f},{0,1,0},{0,1},{0,0,0},{1,1,1,1}},
        {{-0.5f, 0.5f, 0.5f},{0,1,0},{0,0},{0,0,0},{1,1,1,1}},
        {{ 0.5f, 0.5f, 0.5f},{0,1,0},{1,0},{0,0,0},{1,1,1,1}},
        {{ 0.5f, 0.5f,-0.5f},{0,1,0},{1,1},{0,0,0},{1,1,1,1}},
        // -Y 面
        {{-0.5f,-0.5f, 0.5f},{0,-1,0},{0,1},{0,0,0},{1,1,1,1}},
        {{-0.5f,-0.5f,-0.5f},{0,-1,0},{0,0},{0,0,0},{1,1,1,1}},
        {{ 0.5f,-0.5f,-0.5f},{0,-1,0},{1,0},{0,0,0},{1,1,1,1}},
        {{ 0.5f,-0.5f, 0.5f},{0,-1,0},{1,1},{0,0,0},{1,1,1,1}},
        // +Z 面
        {{ 0.5f,-0.5f, 0.5f},{0,0,1},{0,1},{0,0,0},{1,1,1,1}},
        {{ 0.5f, 0.5f, 0.5f},{0,0,1},{0,0},{0,0,0},{1,1,1,1}},
        {{-0.5f, 0.5f, 0.5f},{0,0,1},{1,0},{0,0,0},{1,1,1,1}},
        {{-0.5f,-0.5f, 0.5f},{0,0,1},{1,1},{0,0,0},{1,1,1,1}},
        // -Z 面
        {{-0.5f,-0.5f,-0.5f},{0,0,-1},{0,1},{0,0,0},{1,1,1,1}},
        {{-0.5f, 0.5f,-0.5f},{0,0,-1},{0,0},{0,0,0},{1,1,1,1}},
        {{ 0.5f, 0.5f,-0.5f},{0,0,-1},{1,0},{0,0,0},{1,1,1,1}},
        {{ 0.5f,-0.5f,-0.5f},{0,0,-1},{1,1},{0,0,0},{1,1,1,1}},
    };

    uint32_t indices[] = {
        0,1,2, 0,2,3,
        4,5,6, 4,6,7,
        8,9,10, 8,10,11,
        12,13,14, 12,14,15,
        16,17,18, 16,18,19,
        20,21,22, 20,22,23
    };

    m_vertexBuffer = new VertexBuffer(sizeof(Vertex)*vertices.size(), sizeof(Vertex), vertices.data());
    if (!m_vertexBuffer->IsValid()) return false;

    m_indexBuffer = new IndexBuffer(sizeof(indices), indices);
    if (!m_indexBuffer->IsValid()) return false;

    for (int i = 0; i < Engine::FRAME_BUFFER_COUNT; ++i) {
        m_constantBuffer[i] = new ConstantBuffer(sizeof(Transform));
        auto ptr = m_constantBuffer[i]->GetPtr<Transform>();
        ptr->World = XMMatrixIdentity();
        ptr->View = XMMatrixIdentity();
        ptr->Proj = XMMatrixIdentity();
    }

    m_rootSignature = new RootSignature();
    if (!m_rootSignature->IsValid()) return false;

    m_pipelineState = new PipelineState();
    m_pipelineState->SetInputLayout(Vertex::InputLayout);
    m_pipelineState->SetRootSignature(m_rootSignature->Get());
    m_pipelineState->SetVS(L"SimpleVS.cso");
    m_pipelineState->SetPS(L"SimplePS.cso");
    m_pipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    if (!m_pipelineState->IsValid()) return false;

    return true;
}

void TestCube::Update(float deltaTime) {
    m_rotation.y += deltaTime; // 単純にY軸回転
    auto current = g_Engine->CurrentBackBufferIndex();
    auto camera = g_Engine->GetObj<Camera>("Camera");
    auto ptr = m_constantBuffer[current]->GetPtr<Transform>();
    ptr->World = XMMatrixRotationRollPitchYaw(m_rotation.x, m_rotation.y, m_rotation.z);
    ptr->View = camera->GetViewMatrix();
    ptr->Proj = camera->GetProjMatrix();
}

void TestCube::Render(ID3D12GraphicsCommandList* cmd) {
    auto current = g_Engine->CurrentBackBufferIndex();
    auto vbv = m_vertexBuffer->View();
    auto ibv = m_indexBuffer->View();

    cmd->SetGraphicsRootSignature(m_rootSignature->Get());
    cmd->SetPipelineState(m_pipelineState->Get());
    cmd->SetGraphicsRootConstantBufferView(0, m_constantBuffer[current]->GetAddress());
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd->IASetVertexBuffers(0,1,&vbv);
    cmd->IASetIndexBuffer(&ibv);
    cmd->DrawIndexedInstanced(36,1,0,0,0);
}

