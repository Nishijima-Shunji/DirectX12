#include "DebugCube.h"
#include <vector>

using namespace DirectX;

DebugCube::DebugCube() {
    // 8頂点のキューブを作成
    std::vector<Vertex> vertices = {
        {{-0.5f,-0.5f,-0.5f},{0,0,0},{0,0},{0,0,0},{1,0,0,1}},
        {{-0.5f, 0.5f,-0.5f},{0,0,0},{0,0},{0,0,0},{0,1,0,1}},
        {{ 0.5f, 0.5f,-0.5f},{0,0,0},{0,0},{0,0,0},{0,0,1,1}},
        {{ 0.5f,-0.5f,-0.5f},{0,0,0},{0,0},{0,0,0},{1,1,0,1}},
        {{-0.5f,-0.5f, 0.5f},{0,0,0},{0,0},{0,0,0},{1,0,1,1}},
        {{-0.5f, 0.5f, 0.5f},{0,0,0},{0,0},{0,0,0},{0,1,1,1}},
        {{ 0.5f, 0.5f, 0.5f},{0,0,0},{0,0},{0,0,0},{1,1,1,1}},
        {{ 0.5f,-0.5f, 0.5f},{0,0,0},{0,0},{0,0,0},{0,0,0,1}},
    };

    std::vector<uint32_t> indices = {
        0,1,2, 0,2,3, // -Z 面
        4,6,5, 4,7,6, // +Z 面
        4,5,1, 4,1,0, // -X 面
        3,2,6, 3,6,7, // +X 面
        1,5,6, 1,6,2, // +Y 面
        4,0,3, 4,3,7  // -Y 面
    };
    m_indexCount = static_cast<UINT>(indices.size());

    // 頂点/インデックスバッファを生成
    m_vertexBuffer = new VertexBuffer(sizeof(Vertex)*vertices.size(), sizeof(Vertex), vertices.data());
    m_indexBuffer  = new IndexBuffer(sizeof(uint32_t)*indices.size(), indices.data());

    // 定数バッファをフレーム分確保
    for (size_t i=0; i<Engine::FRAME_BUFFER_COUNT; ++i) {
        m_constantBuffer[i] = new ConstantBuffer(sizeof(Transform));
    }

    // デフォルトルートシグネチャ
    m_rootSignature = new RootSignature();

    // パイプラインステートの設定
    m_pipelineState = new PipelineState();
    m_pipelineState->SetInputLayout(Vertex::InputLayout);
    m_pipelineState->SetRootSignature(m_rootSignature->Get());
    m_pipelineState->SetVS(L"SimpleVS.cso");
    m_pipelineState->SetPS(L"ColorPS.cso");
    m_pipelineState->SetDepthStencilFormat(DXGI_FORMAT_D32_FLOAT); // 深度バッファ(D32)とPSOの設定を一致させる
    m_pipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
}

DebugCube::~DebugCube() {
    delete m_vertexBuffer;
    delete m_indexBuffer;
    delete m_rootSignature;
    delete m_pipelineState;
    for (auto& cb : m_constantBuffer) {
        delete cb;
    }
}

void DebugCube::SetWorldMatrix(const DirectX::XMMATRIX& world)
{
    m_world = world;
}

void DebugCube::Update(float /*deltaTime*/) {
    // カメラの行列を定数バッファへ設定
    auto cam = g_Engine->GetObj<Camera>("Camera");
    auto idx = g_Engine->CurrentBackBufferIndex();
    auto ptr = m_constantBuffer[idx]->GetPtr<Transform>();
    ptr->World = m_world;
    ptr->View  = cam->GetViewMatrix();
    ptr->Proj  = cam->GetProjMatrix();
}

void DebugCube::Render(ID3D12GraphicsCommandList* cmd) {
    auto idx = g_Engine->CurrentBackBufferIndex();
    cmd->SetGraphicsRootSignature(m_rootSignature->Get());
    cmd->SetPipelineState(m_pipelineState->Get());
    cmd->SetGraphicsRootConstantBufferView(0, m_constantBuffer[idx]->GetAddress());

    auto vbView = m_vertexBuffer->View();
    auto ibView = m_indexBuffer->View();
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd->IASetVertexBuffers(0,1,&vbView);
    cmd->IASetIndexBuffer(&ibView);
    cmd->DrawIndexedInstanced(m_indexCount,1,0,0,0);
}
