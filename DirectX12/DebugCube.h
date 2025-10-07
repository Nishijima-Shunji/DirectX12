#pragma once
#include "IActor.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "RootSignature.h"
#include "PipelineState.h"
#include "ConstantBuffer.h"
#include "Engine.h"
#include "SharedStruct.h"
#include "Camera.h"

// 描画位置確認用のキューブ
class DebugCube : public IActor {
private:
    VertexBuffer*   m_vertexBuffer = nullptr; // 頂点バッファ
    IndexBuffer*    m_indexBuffer  = nullptr; // インデックスバッファ
    RootSignature*  m_rootSignature = nullptr; // ルートシグネチャ
    PipelineState*  m_pipelineState = nullptr; // パイプラインステート
    ConstantBuffer* m_constantBuffer[Engine::FRAME_BUFFER_COUNT] = {}; // 各フレーム用定数バッファ
    DirectX::XMMATRIX m_world = DirectX::XMMatrixIdentity(); // ワールド行列
    UINT m_indexCount = 0; // 描画に使用するインデックス数
public:
    DebugCube();
    ~DebugCube();
    void SetWorldMatrix(const DirectX::XMMATRIX& world);
    void Update(float deltaTime) override;
    void Render(ID3D12GraphicsCommandList* cmd) override;
};
