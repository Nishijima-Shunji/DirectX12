#include "GameScene.h"
#include "DebugCube.h"
#include "Engine.h"
#include "Camera.h"
#include "VertexBuffer.h"
#include "PipelineState.h"
#include "RootSignature.h"
#include "ConstantBuffer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <array>
#include <Windows.h>

using namespace DirectX;

// 壁の情報(とりあえずでここに置く)
namespace GameSceneDetail
{
    struct WallVertex
    {
        XMFLOAT3 position; // 頂点座標
        XMFLOAT4 color;    // RGBA 色
    };

    struct WallConstant
    {
        XMFLOAT4X4 view; // ビュー行列
        XMFLOAT4X4 proj; // プロジェクション行列
    };

    // 透明な壁の描画を担当する補助クラス
    class TransparentWalls
    {
    public:
        bool Init(const FluidSystem::Bounds& bounds)
        {
            m_cachedBounds = bounds;
            BuildVertices(bounds);

            m_vertexBuffer = std::make_unique<VertexBuffer>(sizeof(WallVertex) * m_vertices.size(), sizeof(WallVertex), m_vertices.data());
            if (!m_vertexBuffer || !m_vertexBuffer->IsValid())
            {
                return false;
            }

            m_rootSignature = std::make_unique<RootSignature>();
            if (!m_rootSignature || !m_rootSignature->IsValid())
            {
                return false;
            }

            m_pipelineState = std::make_unique<PipelineState>();
            if (!m_pipelineState)
            {
                return false;
            }
            D3D12_INPUT_ELEMENT_DESC layout[] =
            {
                { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, offsetof(WallVertex, position), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
                { "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, offsetof(WallVertex, color),    D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            };

            m_pipelineState->SetInputLayout({ layout, _countof(layout) });
            m_pipelineState->SetRootSignature(m_rootSignature->Get());
            m_pipelineState->SetVS(L"ColorOnlyVS.cso");
            m_pipelineState->SetPS(L"ColorOnlyPS.cso");
            m_pipelineState->SetDepthStencilFormat(DXGI_FORMAT_D32_FLOAT);
            m_pipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
            if (!m_pipelineState->IsValid())
            {
                return false;
            }

            for (auto& cb : m_constantBuffers)
            {
                cb = std::make_unique<ConstantBuffer>(sizeof(WallConstant));
                if (!cb || !cb->IsValid())
                {
                    return false;
                }
            }
            return true;
        }

       // 壁の移動
        void Update(const FluidSystem::Bounds& bounds, const Camera& camera)
        {
            if (bounds.min.x != m_cachedBounds.min.x || bounds.min.y != m_cachedBounds.min.y ||
                bounds.min.z != m_cachedBounds.min.z || bounds.max.x != m_cachedBounds.max.x ||
                bounds.max.y != m_cachedBounds.max.y || bounds.max.z != m_cachedBounds.max.z)
            {
                m_cachedBounds = bounds;
                BuildVertices(bounds);
                if (m_vertexBuffer)
                {
                    void* mapped = nullptr;
                    auto resource = m_vertexBuffer->GetResource();
                    if (resource && SUCCEEDED(resource->Map(0, nullptr, &mapped)))
                    {
                        memcpy(mapped, m_vertices.data(), sizeof(WallVertex) * m_vertices.size());
                        resource->Unmap(0, nullptr);
                    }
                }
            }

            UINT frameIndex = g_Engine->CurrentBackBufferIndex();
            auto& cb = m_constantBuffers[frameIndex];
            if (!cb)
            {
                return;
            }
            WallConstant* constant = cb->GetPtr<WallConstant>();
            XMStoreFloat4x4(&constant->view, XMMatrixTranspose(camera.GetViewMatrix()));
            XMStoreFloat4x4(&constant->proj, XMMatrixTranspose(camera.GetProjMatrix()));
        }

        void Render(ID3D12GraphicsCommandList* cmd)
        {
            if (!cmd || !m_vertexBuffer || !m_pipelineState || !m_rootSignature)
            {
                return;
            }

            UINT frameIndex = g_Engine->CurrentBackBufferIndex();
            auto& cb = m_constantBuffers[frameIndex];
            if (!cb)
            {
                return;
            }

            cmd->SetGraphicsRootSignature(m_rootSignature->Get());
            cmd->SetPipelineState(m_pipelineState->Get());
            cmd->SetGraphicsRootConstantBufferView(0, cb->GetAddress());

            auto vbView = m_vertexBuffer->View();
            cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd->IASetVertexBuffers(0, 1, &vbView);
            cmd->DrawInstanced(static_cast<UINT>(m_vertices.size()), 1, 0, 0);
        }

    private:
        void BuildVertices(const FluidSystem::Bounds& bounds)
        {
            m_vertices.clear();
            const XMFLOAT4 wallColor{ 0.2f, 0.6f, 1.0f, 0.0f }; // 透明感を表現する薄い色
            auto addQuad = [this, &wallColor](const XMFLOAT3& a, const XMFLOAT3& b, const XMFLOAT3& c, const XMFLOAT3& d)
            {
                m_vertices.push_back({ a, wallColor });
                m_vertices.push_back({ b, wallColor });
                m_vertices.push_back({ c, wallColor });
                m_vertices.push_back({ a, wallColor });
                m_vertices.push_back({ c, wallColor });
                m_vertices.push_back({ d, wallColor });
            };

            const float minX = bounds.min.x;
            const float maxX = bounds.max.x;
            const float minY = bounds.min.y;
            const float maxY = bounds.max.y;
            const float minZ = bounds.min.z;
            const float maxZ = bounds.max.z;

            addQuad({ minX, minY, maxZ }, { minX, maxY, maxZ }, { minX, maxY, minZ }, { minX, minY, minZ }); // 左壁
            addQuad({ maxX, minY, minZ }, { maxX, maxY, minZ }, { maxX, maxY, maxZ }, { maxX, minY, maxZ }); // 右壁
            addQuad({ minX, minY, minZ }, { minX, maxY, minZ }, { maxX, maxY, minZ }, { maxX, minY, minZ }); // 奥壁
            addQuad({ maxX, minY, maxZ }, { maxX, maxY, maxZ }, { minX, maxY, maxZ }, { minX, minY, maxZ }); // 手前壁
        }

        std::vector<WallVertex> m_vertices;
        FluidSystem::Bounds m_cachedBounds{};
        std::unique_ptr<VertexBuffer> m_vertexBuffer;
        std::unique_ptr<RootSignature> m_rootSignature;
        std::unique_ptr<PipelineState> m_pipelineState;
        std::array<std::unique_ptr<ConstantBuffer>, Engine::FRAME_BUFFER_COUNT> m_constantBuffers;
    };
}

using GameSceneDetail::TransparentWalls;

// コンストラクタ
GameScene::GameScene(Game* game)
    : BaseScene(game)
{
    // 水平面の大きさ
    m_initialBounds.min = XMFLOAT3(-20.0f, 0.0f, -20.0f);
    m_initialBounds.max = XMFLOAT3(20.0f, 2.5f, 20.0f);
}

GameScene::~GameScene() = default;

// シーン初期化
bool GameScene::Init()
{
    auto* camera = new Camera();
    g_Engine->RegisterObj<Camera>("Camera", camera);

    if (!RecreateFluid())
    {
        return false; // 初期生成に失敗した場合はシーンの起動を中止
    }

    m_walls = std::make_unique<TransparentWalls>();
    if (!m_walls || !m_walls->Init(m_initialBounds))
    {
        return false;
    }

    m_debugCube = std::make_unique<DebugCube>();
    if (!m_debugCube)
    {
        return false;
    }
    XMFLOAT3 center{
        (m_initialBounds.min.x + m_initialBounds.max.x) * 0.5f,
        m_initialBounds.min.y + 0.3f,
        (m_initialBounds.min.z + m_initialBounds.max.z) * 0.5f };
    m_debugCube->SetWorldMatrix(XMMatrixScaling(0.4f, 0.4f, 0.4f) * XMMatrixTranslation(center.x, center.y, center.z));

    return true;
}

// 水生成
bool GameScene::RecreateFluid()
{
    auto newFluid = std::make_unique<FluidSystem>();
    if (!newFluid || !newFluid->Init(g_Engine->Device(), m_initialBounds, 0))
    {
        OutputDebugStringA("FluidSystem recreate failed.\n"); // 失敗をデバッグ出力して原因調査しやすくする
        return false;
    }

    m_fluid = std::move(newFluid);
    return true;
}

// 視点の先にある水面のXZ座標を求める
static bool RayHitWaterXZ(Camera& cam, float waterY, DirectX::XMFLOAT2& outXZ)
{
    using namespace DirectX;
    XMVECTOR eye = cam.GetEyePos();
    XMVECTOR dir = XMVector3Normalize(cam.GetTargetPos() - eye);
    float dy = XMVectorGetY(dir);
    if (fabsf(dy) < 1e-5f) return false;             // ほぼ水平で当たらない
    float t = (waterY - XMVectorGetY(eye)) / dy;
    if (t < 0.0f) return false;                       // 後ろ向き
    XMVECTOR p = eye + dir * t;
    outXZ = XMFLOAT2(XMVectorGetX(p), XMVectorGetZ(p));
    return true;
}

// カメラ操作に応じて水面を持ち上げる
void GameScene::HandleCameraLift(Camera& camera, float deltaTime)
{
    if (!m_fluid) return;

    using namespace DirectX;

    // --- ここで操作パラメータ（好みで調整可） ---
    const float kGrabRadius = 1.18f;   // 掴む円の半径[m]
    const float kLiftPerSec = 5.60f;   // 押下中に毎秒どれだけ持ち上げるか[m/s]
    const float kThrowScale = 1.00f;   // 投げ速度スケール
    const float kMinThrowSpd = 0.2f;   // これ未満なら投げない[m/s]
    // ---------------------------------------------

    // 関数内で状態を保持（簡単のためstatic使用）
    static bool sGrabbing = false;
    static XMFLOAT2 sPrevXZ{ 0,0 };
    static XMFLOAT2 sVelXZ{ 0,0 };         // 低域通過した投げ速度用
    static float    sTimeAccum = 0.0f;

    const bool lmbDown = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;

    // 現在の水位（FluidSystemに getter がある想定。なければ 0.0f などに置換）
    const float waterY = 0.0f;
#if 1
        m_fluid->GetWaterLevel();
#else
        0.0f;
#endif

    XMFLOAT2 hitXZ;
    const bool hit = RayHitWaterXZ(camera, waterY, hitXZ);

    if (!lmbDown)
    {
        // ボタンを離した瞬間：投げる
        if (sGrabbing)
        {
            // 速度ベクトル = 直近の平滑化した速度
            XMVECTOR v = XMLoadFloat2(&sVelXZ);
            float spd = XMVectorGetX(XMVector2Length(v)) * kThrowScale;
            if (spd >= kMinThrowSpd)
            {
                XMVECTOR dir = XMVector2Normalize(v);
                XMFLOAT2 throwDir, throwVel;
                XMStoreFloat2(&throwDir, dir);
                m_fluid->EndGrab(throwDir, spd);
            }
            else
            {
                // ほぼその場で離したときは投げなし
                XMFLOAT2 zero{ 0,0 };
                m_fluid->EndGrab(zero, 0.0f);
            }
        }
        // 状態リセット
        sGrabbing = false;
        sVelXZ = XMFLOAT2(0, 0);
        sTimeAccum = 0.0f;
        return;
    }

    // マウス押下中
    if (!hit) return; // 水面とレイが交差しないなら何もしない

    if (!sGrabbing)
    {
        // 掴み開始
        sGrabbing = true;
        sPrevXZ = hitXZ;
        sVelXZ = XMFLOAT2(0, 0);
        sTimeAccum = 0.0f;
        m_fluid->BeginGrab(hitXZ, kGrabRadius);
        return;
    }

    // 掴み継続：持ち上げ＋スピード推定
    sTimeAccum += deltaTime;

    // 位置差分→速度推定（2D）
    XMVECTOR cur = XMLoadFloat2(&hitXZ);
    XMVECTOR prv = XMLoadFloat2(&sPrevXZ);
    XMVECTOR d = cur - prv;
    if (deltaTime > 0.0f)
    {
        XMVECTOR instV = d / deltaTime;                 // 瞬間速度
        // 低域通過（スムーズな投げ方向/速度に）
        XMVECTOR oldV = XMLoadFloat2(&sVelXZ);
        XMVECTOR v = XMVectorLerp(oldV, instV, 0.5f);
        XMStoreFloat2(&sVelXZ, v);
    }
    XMStoreFloat2(&sPrevXZ, cur);

    // 水面を持ち上げる（押下中は常時少しずつ上げる）
    m_fluid->UpdateGrab(hitXZ, /*liftPerSec*/ kLiftPerSec, deltaTime);
}

// 壁の押し引き、スライド操作
void GameScene::HandleWallControl(Camera& camera, float deltaTime)
{
    if (!(GetAsyncKeyState(VK_SPACE) & 0x8000))
    {
        return;
    }

    float pushPull = 0.0f;
    if (GetAsyncKeyState('W') & 0x8000) pushPull += 15.0f;
    if (GetAsyncKeyState('S') & 0x8000) pushPull -= 15.0f;

    float slide = 0.0f;
    if (GetAsyncKeyState('D') & 0x8000) slide += 15.0f;
    if (GetAsyncKeyState('A') & 0x8000) slide -= 15.0f;

    if (std::fabs(pushPull) < 1e-3f && std::fabs(slide) < 1e-3f)
    {
        return;
    }

    // カメラ側から壁へ向かう水平方向ベクトルを算出
    XMVECTOR viewToCamera = XMVectorSubtract(camera.GetEyePos(), camera.GetTargetPos());
    viewToCamera = XMVectorSet(XMVectorGetX(viewToCamera), 0.0f, XMVectorGetZ(viewToCamera), 0.0f);
    if (XMVector3LengthSq(viewToCamera).m128_f32[0] < 1e-6f)
    {
        viewToCamera = XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f);
    }
    viewToCamera = XMVector3Normalize(viewToCamera);

    XMVECTOR right = XMVector3Cross(XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f), viewToCamera);
    if (XMVector3LengthSq(right).m128_f32[0] < 1e-6f)
    {
        right = XMVectorSet(1.0f, 0.0f, 0.0f, 0.0f);
    }
    right = XMVector3Normalize(right);

    if (!m_fluid)
    {
        return;
    }

    const float moveAmount = m_wallMoveSpeed * deltaTime;

    if (std::fabs(pushPull) >= 1e-3f)
    {
        XMFLOAT3 direction{};
        XMStoreFloat3(&direction, viewToCamera);
        // 観測中の壁を押したり引いたりする
        m_fluid->AdjustWall(direction, pushPull * moveAmount);
    }

    if (std::fabs(slide) >= 1e-3f)
    {
        XMFLOAT3 direction{};
        XMStoreFloat3(&direction, right);
        // 右方向ベクトルに沿って側面の壁も制御する
        m_fluid->AdjustWall(direction, slide * moveAmount, deltaTime);
    }
}

void GameScene::Update(float deltaTime)
{
    Camera* camera = g_Engine->GetObj<Camera>("Camera");
    if (!camera)
    {
        return;
    }

    camera->Update(deltaTime);
    HandleCameraLift(*camera, deltaTime);
    HandleWallControl(*camera, deltaTime);

    if (m_fluid)
    {
        m_fluid->Update(deltaTime);
    }

    if (m_debugCube && m_fluid)
    {
        const auto& bounds = m_fluid->GetBounds();
        XMFLOAT3 center{
            (bounds.min.x + bounds.max.x) * 0.5f,
            bounds.min.y + 0.3f,
            (bounds.min.z + bounds.max.z) * 0.5f };
        m_debugCube->SetWorldMatrix(XMMatrixScaling(0.4f, 0.4f, 0.4f) * XMMatrixTranslation(center.x, center.y, center.z));
        m_debugCube->Update(deltaTime);
    }

    if (m_walls && m_fluid)
    {
        m_walls->Update(m_fluid->GetBounds(), *camera);
    }
}

void GameScene::Draw()
{
    ID3D12GraphicsCommandList* cmd = g_Engine->CommandList();
    if (!cmd)
    {
        return;
    }

    Camera* camera = g_Engine->GetObj<Camera>("Camera");
    if (!camera)
    {
        return;
    }

    if (m_fluid)
    {
        m_fluid->Draw(cmd, *camera);
    }

    if (m_fluid && m_debugCube) {
        using namespace DirectX;

        m_fluid->ForEachDebugParticle([&](const auto& p) {
            const float s = std::max(0.02f, p.radius * 2.0f); // 小さすぎ対策
            // 少しだけ浮かせて深度衝突を避ける（見えない対策）
            const XMMATRIX W = XMMatrixScaling(s, s, s) *
                XMMatrixTranslation(p.pos.x, p.pos.y + 0.01f, p.pos.z);

            m_debugCube->SetWorldMatrix(W);
            m_debugCube->Update(0.0f);   // ★ CB更新を必ず呼ぶ
            m_debugCube->Render(cmd);    // 1粒子=1ドロー（少数 or サンプリング前提）
            });
    }


    if (m_walls)
    {
        m_walls->Render(cmd);
    }

    if (m_debugCube)
    {
        m_debugCube->Render(cmd);
    }
}
