#pragma once
// テスト用のシンプルな立方体を描画するアクタ
// 日本語コメントで分かりやすく説明する

#include "IActor.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "ConstantBuffer.h"
#include "RootSignature.h"
#include "PipelineState.h"
#include "Camera.h"
#include "SharedStruct.h"
#include "Engine.h"

#include <DirectXMath.h>

// テスト用の立方体を生成して描画するクラス
class TestCube : public IActor {
public:
    TestCube();
    ~TestCube();

    // 初期化処理
    bool Init();

    // 毎フレームの更新処理
    void Update(float deltaTime) override;

    // 描画処理
    void Render(ID3D12GraphicsCommandList* cmd) override;

private:
    VertexBuffer* m_vertexBuffer = nullptr;        // 頂点データ
    IndexBuffer*  m_indexBuffer  = nullptr;        // インデックスデータ
    ConstantBuffer* m_constantBuffer[Engine::FRAME_BUFFER_COUNT] = {}; // 行列用CB
    RootSignature* m_rootSignature = nullptr;      // ルートシグネチャ
    PipelineState* m_pipelineState = nullptr;      // パイプラインステート
    DirectX::XMFLOAT3 m_rotation = {0.f, 0.f, 0.f}; // 回転角
};

