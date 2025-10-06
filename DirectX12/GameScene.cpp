#include "GameScene.h"
#include "Game.h"
#include "Engine.h"
#include "App.h"
#include <algorithm>
#include <random>
#include <cstring>
#include <cstddef>

using namespace DirectX;

namespace
{
    constexpr DXGI_FORMAT kBackBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
    constexpr float kMaxRayDistance = 20.0f; // レイ判定距離の上限
}

GameScene::GameScene(Game* game)
    : BaseScene(game)
{
}

GameScene::~GameScene() = default;

bool GameScene::Init()
{
    // === カメラ初期化 ===
    auto* camera = new Camera();
    g_Engine->RegisterObj<Camera>("Camera", camera);

    // === 流体系初期化 ===
    m_fluid = std::make_unique<FluidSystem>();
    if (!m_fluid)
    {
        return false;
    }

    m_fluid->Init(g_Engine->Device(), kBackBufferFormat, 512, 128);
    m_fluid->UseGPU(false); // 安定性を優先してCPUでシミュレーション
    m_fluid->SetSimulationBounds(XMFLOAT3(-2.0f, 0.0f, -2.0f), XMFLOAT3(2.0f, 3.0f, 2.0f));
    m_fluid->SetWaterAppearance(
        XMFLOAT3(0.25f, 0.55f, 0.95f), // 浅い部分の色
        XMFLOAT3(0.07f, 0.22f, 0.38f), // 深い部分の色
        0.35f,                         // 吸収率
        0.45f,                         // 泡のしきい値
        0.38f,                         // 泡の強さ
        0.65f,                         // 反射率
        96.0f);                        // スペキュラ強度

    InitializeStageGeometry();

    // === ルートシグネチャとPSOの生成 ===
    m_colorRootSignature = std::make_unique<RootSignature>();
    if (!m_colorRootSignature || !m_colorRootSignature->IsValid())
    {
        return false;
    }

    static const D3D12_INPUT_ELEMENT_DESC kColorInputLayout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(ColorVertex, position), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, offsetof(ColorVertex, color),    D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };

    m_stagePipeline = std::make_unique<PipelineState>();
    m_stagePipeline->SetInputLayout({ kColorInputLayout, _countof(kColorInputLayout) });
    m_stagePipeline->SetRootSignature(m_colorRootSignature->Get());
    m_stagePipeline->SetVS(L"ColorOnlyVS.cso");
    m_stagePipeline->SetPS(L"ColorOnlyPS.cso");
    m_stagePipeline->SetDepthStencilFormat(DXGI_FORMAT_D32_FLOAT);
    m_stagePipeline->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    if (!m_stagePipeline->IsValid())
    {
        return false;
    }

    m_splashPipeline = std::make_unique<PipelineState>();
    m_splashPipeline->SetInputLayout({ kColorInputLayout, _countof(kColorInputLayout) });
    m_splashPipeline->SetRootSignature(m_colorRootSignature->Get());
    m_splashPipeline->SetVS(L"ColorOnlyVS.cso");
    m_splashPipeline->SetPS(L"ColorOnlyPS.cso");
    m_splashPipeline->SetDepthStencilFormat(DXGI_FORMAT_D32_FLOAT);
    m_splashPipeline->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    if (!m_splashPipeline->IsValid())
    {
        return false;
    }

    // === 定数バッファ確保 ===
    for (auto& cb : m_colorCB)
    {
        cb = std::make_unique<ConstantBuffer>(sizeof(ColorPassCB));
        if (!cb || !cb->IsValid())
        {
            return false;
        }
    }

    // === ステージ頂点バッファの生成 ===
    if (!m_stageVertices.empty())
    {
        m_stageVB = std::make_unique<VertexBuffer>(
            sizeof(ColorVertex) * m_stageVertices.size(),
            sizeof(ColorVertex),
            m_stageVertices.data());
        if (!m_stageVB || !m_stageVB->IsValid())
        {
            return false;
        }
        m_stageVertexCount = static_cast<UINT>(m_stageVertices.size());
    }

    return true;
}

void GameScene::InitializeStageGeometry()
{
    m_stageVertices.clear();
    auto addQuad = [this](const XMFLOAT3& a, const XMFLOAT3& b, const XMFLOAT3& c, const XMFLOAT3& d, const XMFLOAT4& color)
    {
        m_stageVertices.push_back({ a, color });
        m_stageVertices.push_back({ b, color });
        m_stageVertices.push_back({ c, color });
        m_stageVertices.push_back({ a, color });
        m_stageVertices.push_back({ c, color });
        m_stageVertices.push_back({ d, color });
    };

    const XMFLOAT4 floorColor{ 0.25f, 0.35f, 0.42f, 1.0f };
    const XMFLOAT4 wallColor{ 0.18f, 0.22f, 0.28f, 1.0f };
    const float minX = -2.0f;
    const float maxX = 2.0f;
    const float minZ = -2.0f;
    const float maxZ = 2.0f;
    const float maxY = 2.0f;

    // 床
    addQuad(
        XMFLOAT3(minX, 0.0f, minZ),
        XMFLOAT3(maxX, 0.0f, minZ),
        XMFLOAT3(maxX, 0.0f, maxZ),
        XMFLOAT3(minX, 0.0f, maxZ),
        floorColor);

    // 壁（左）
    addQuad(
        XMFLOAT3(minX, 0.0f, maxZ),
        XMFLOAT3(minX, maxY, maxZ),
        XMFLOAT3(minX, maxY, minZ),
        XMFLOAT3(minX, 0.0f, minZ),
        wallColor);

    // 壁（右）
    addQuad(
        XMFLOAT3(maxX, 0.0f, minZ),
        XMFLOAT3(maxX, maxY, minZ),
        XMFLOAT3(maxX, maxY, maxZ),
        XMFLOAT3(maxX, 0.0f, maxZ),
        wallColor);

    // 壁（奥）
    addQuad(
        XMFLOAT3(minX, 0.0f, minZ),
        XMFLOAT3(minX, maxY, minZ),
        XMFLOAT3(maxX, maxY, minZ),
        XMFLOAT3(maxX, 0.0f, minZ),
        wallColor);

    // 壁（手前）
    addQuad(
        XMFLOAT3(maxX, 0.0f, maxZ),
        XMFLOAT3(maxX, maxY, maxZ),
        XMFLOAT3(minX, maxY, maxZ),
        XMFLOAT3(minX, 0.0f, maxZ),
        wallColor);
}

void GameScene::Update(float deltaTime)
{
    m_deltaTime = deltaTime;

    auto* camera = g_Engine->GetObj<Camera>("Camera");
    if (camera)
    {
        camera->Update(deltaTime);
    }

    if (m_fluid)
    {
        m_fluid->ClearDynamicOperations();
    }

    UpdateGatherOperation(camera);
    UpdateSplashParticles(deltaTime, camera);

    // デバッグ用途：キー入力でマテリアルを切り替え
    if (GetAsyncKeyState('1') & 0x0001)
    {
        m_fluid->SetMaterialPreset(FluidMaterialPreset::Water);
    }
    if (GetAsyncKeyState('2') & 0x0001)
    {
        m_fluid->SetMaterialPreset(FluidMaterialPreset::Magma);
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

    UpdateCameraConstantBuffer(camera);
    RenderStage(cmd);

    XMMATRIX view = camera->GetViewMatrix();
    XMMATRIX proj = camera->GetProjMatrix();
    XMMATRIX viewProj = XMMatrixMultiply(view, proj);
    XMFLOAT4X4 viewM, projM, viewProjM;
    XMStoreFloat4x4(&viewM, view);
    XMStoreFloat4x4(&projM, proj);
    XMStoreFloat4x4(&viewProjM, viewProj);
    XMFLOAT3 cameraPos = camera->GetPosition();

    if (m_fluid)
    {
        m_fluid->Simulate(cmd, m_deltaTime);
        m_fluid->Render(cmd, viewM, projM, viewProjM, cameraPos, 0.18f);
        m_fluid->Composite(cmd,
            g_Engine->CurrentRenderTargetResource(),
            g_Engine->DepthStencilBuffer(),
            g_Engine->CurrentBackBufferView());
    }

    SpawnSplashFromCollisions();
    UploadSplashVertices(camera);
    RenderSplash(cmd);
}

void GameScene::UpdateCameraConstantBuffer(const Camera* camera)
{
    if (!camera)
    {
        return;
    }
    UINT frameIndex = g_Engine->CurrentBackBufferIndex();
    auto& cb = m_colorCB[frameIndex];
    if (!cb)
    {
        return;
    }

    ColorPassCB* data = cb->GetPtr<ColorPassCB>();
    XMMATRIX view = camera->GetViewMatrix();
    XMMATRIX proj = camera->GetProjMatrix();
    XMStoreFloat4x4(&data->view, XMMatrixTranspose(view));
    XMStoreFloat4x4(&data->proj, XMMatrixTranspose(proj));
}

void GameScene::RenderStage(ID3D12GraphicsCommandList* cmd)
{
    if (!m_stageVB || !m_stagePipeline || !m_colorRootSignature)
    {
        return;
    }
    UINT frameIndex = g_Engine->CurrentBackBufferIndex();
    auto& cb = m_colorCB[frameIndex];
    if (!cb)
    {
        return;
    }

    cmd->SetGraphicsRootSignature(m_colorRootSignature->Get());
    cmd->SetPipelineState(m_stagePipeline->Get());
    cmd->SetGraphicsRootConstantBufferView(0, cb->GetAddress());

    auto vbView = m_stageVB->View();
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd->IASetVertexBuffers(0, 1, &vbView);
    cmd->DrawInstanced(m_stageVertexCount, 1, 0, 0);
}

void GameScene::RenderSplash(ID3D12GraphicsCommandList* cmd)
{
    if (!m_splashVB || !m_splashPipeline || m_splashVertexCount == 0)
    {
        return;
    }

    UINT frameIndex = g_Engine->CurrentBackBufferIndex();
    auto& cb = m_colorCB[frameIndex];
    if (!cb)
    {
        return;
    }

    cmd->SetGraphicsRootSignature(m_colorRootSignature->Get());
    cmd->SetPipelineState(m_splashPipeline->Get());
    cmd->SetGraphicsRootConstantBufferView(0, cb->GetAddress());

    auto vbView = m_splashVB->View();
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd->IASetVertexBuffers(0, 1, &vbView);
    cmd->DrawInstanced(m_splashVertexCount, 1, 0, 0);
}

void GameScene::UpdateGatherOperation(Camera* camera)
{
    if (!m_fluid || !camera)
    {
        return;
    }

    bool leftDown = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;

    if (leftDown && !m_leftButtonDown)
    {
        m_leftButtonDown = BeginGather(camera);
    }
    else if (!leftDown && m_leftButtonDown)
    {
        ReleaseGather(camera);
        m_leftButtonDown = false;
    }

    if (leftDown && m_gatherState.active)
    {
        m_gatherState.holdTime += m_deltaTime;

        XMFLOAT3 camPos = camera->GetPosition();
        XMFLOAT3 target{
            camPos.x + m_gatherState.cameraOffset.x,
            camPos.y + m_gatherState.cameraOffset.y,
            camPos.z + m_gatherState.cameraOffset.z };

        // カメラ前方へ少し押し出して手前にまとまるようにする
        XMMATRIX invView = XMMatrixInverse(nullptr, camera->GetViewMatrix());
        XMVECTOR forward = XMVector3Normalize(invView.r[2]);
        XMFLOAT3 forwardF;
        XMStoreFloat3(&forwardF, forward);
        target.x += forwardF.x * 0.2f;
        target.y += forwardF.y * 0.2f;
        target.z += forwardF.z * 0.2f;

        float radius = std::clamp(0.35f + 0.25f / (1.0f + m_gatherState.holdTime), 0.2f, 0.6f);
        float strength = m_gatherState.baseStrength + m_gatherState.holdTime * 6.0f;
        m_fluid->QueueGather(target, radius, strength);

        // 収束中は揺らぎを抑えるために追加の軽い吸引を入れる
        m_fluid->QueueGather(target, radius * 0.5f, strength * 0.5f);
    }
}

bool GameScene::BeginGather(Camera* camera)
{
    if (!camera || !m_fluid)
    {
        return false;
    }

    POINT cursor;
    GetCursorPos(&cursor);
    ScreenToClient(g_hWnd, &cursor);

    XMFLOAT3 origin, direction;
    if (!ScreenPointToRay(cursor.x, cursor.y, origin, direction))
    {
        return false;
    }

    XMFLOAT3 hitPos;
    if (!m_fluid->Raycast(origin, direction, kMaxRayDistance, 0.25f, hitPos))
    {
        return false;
    }

    XMFLOAT3 camPos = camera->GetPosition();
    m_gatherState.active = true;
    m_gatherState.holdTime = 0.0f;
    m_gatherState.hitPosition = hitPos;
    m_gatherState.cameraOffset = XMFLOAT3(
        hitPos.x - camPos.x,
        hitPos.y - camPos.y,
        hitPos.z - camPos.z);
    m_gatherState.radius = 0.45f;
    m_gatherState.baseStrength = 14.0f;

    // 初期段階で軽く吸引して集まり始めるようにする
    m_fluid->QueueGather(hitPos, m_gatherState.radius, m_gatherState.baseStrength);

    return true;
}

void GameScene::ReleaseGather(Camera* camera)
{
    if (!m_fluid || !camera || !m_gatherState.active)
    {
        m_gatherState.active = false;
        return;
    }

    XMFLOAT3 camPos = camera->GetPosition();
    XMMATRIX invView = XMMatrixInverse(nullptr, camera->GetViewMatrix());
    XMVECTOR forwardVec = XMVector3Normalize(invView.r[2]);
    XMFLOAT3 forward;
    XMStoreFloat3(&forward, forwardVec);

    XMFLOAT3 center{
        camPos.x + m_gatherState.cameraOffset.x + forward.x * 0.2f,
        camPos.y + m_gatherState.cameraOffset.y + forward.y * 0.2f,
        camPos.z + m_gatherState.cameraOffset.z + forward.z * 0.2f };

    float radius = std::clamp(0.35f + 0.25f / (1.0f + m_gatherState.holdTime), 0.2f, 0.6f);
    float impulse = 3.5f + m_gatherState.holdTime * 2.0f;
    float directional = 12.0f + m_gatherState.holdTime * 6.0f;

    m_fluid->QueueDirectionalImpulse(center, radius, forward, directional);
    m_fluid->QueueSplash(center, radius, impulse);

    m_gatherState.active = false;
}

bool GameScene::ScreenPointToRay(int mouseX, int mouseY, XMFLOAT3& outOrigin, XMFLOAT3& outDirection) const
{
    const Camera* camera = g_Engine->GetObj<Camera>("Camera");
    if (!camera)
    {
        return false;
    }

    float width = static_cast<float>(g_Engine->FrameBufferWidth());
    float height = static_cast<float>(g_Engine->FrameBufferHeight());
    float ndcX = (2.0f * mouseX / width) - 1.0f;
    float ndcY = -((2.0f * mouseY / height) - 1.0f);

    XMMATRIX view = camera->GetViewMatrix();
    XMMATRIX proj = camera->GetProjMatrix();
    XMMATRIX invViewProj = XMMatrixInverse(nullptr, XMMatrixMultiply(view, proj));
    XMVECTOR nearPoint = XMVectorSet(ndcX, ndcY, 0.0f, 1.0f);
    XMVECTOR farPoint = XMVectorSet(ndcX, ndcY, 1.0f, 1.0f);

    nearPoint = XMVector3TransformCoord(nearPoint, invViewProj);
    farPoint = XMVector3TransformCoord(farPoint, invViewProj);

    XMVECTOR rayDir = XMVector3Normalize(XMVectorSubtract(farPoint, nearPoint));
    XMStoreFloat3(&outOrigin, nearPoint);
    XMStoreFloat3(&outDirection, rayDir);
    return true;
}

void GameScene::SpawnSplashFromCollisions()
{
    if (!m_fluid)
    {
        return;
    }

    std::vector<FluidCollisionEvent> events;
    m_fluid->PopCollisionEvents(events);
    if (events.empty())
    {
        return;
    }

    static std::mt19937 rng{ std::random_device{}() };
    std::uniform_real_distribution<float> random01(0.0f, 1.0f);

    for (const auto& evt : events)
    {
        if (evt.strength < 0.5f)
        {
            continue;
        }

        const int spawnCount = std::clamp(static_cast<int>(evt.strength * 0.4f), 2, 6);
        XMVECTOR normal = XMLoadFloat3(&evt.normal);

        for (int i = 0; i < spawnCount; ++i)
        {
            SplashParticle particle{};
            particle.position = evt.position;
            particle.age = 0.0f;
            particle.lifetime = 0.45f + random01(rng) * 0.35f;
            particle.initialStrength = evt.strength;
            particle.size = 0.18f + evt.strength * 0.05f;

            float spread = 0.5f + random01(rng) * 0.6f;
            float speed = evt.strength * (0.45f + random01(rng) * 0.25f);

            // 法線方向をベースに少し散らばるよう乱数を加える
            XMVECTOR tangent = XMVector3Normalize(XMVectorSet(random01(rng) * 2.0f - 1.0f, 0.0f, random01(rng) * 2.0f - 1.0f, 0.0f));
            XMVECTOR side = XMVector3Normalize(XMVector3Cross(normal, tangent));
            XMVECTOR velocity = XMVectorAdd(XMVectorScale(normal, speed), XMVectorScale(side, spread));
            XMStoreFloat3(&particle.velocity, velocity);

            m_splashParticles.push_back(particle);
        }
    }
}

void GameScene::UpdateSplashParticles(float deltaTime, const Camera* /*camera*/)
{
    if (m_splashParticles.empty())
    {
        return;
    }

    const XMFLOAT3 gravity{ 0.0f, -9.8f, 0.0f };
    for (auto& particle : m_splashParticles)
    {
        particle.velocity.x += gravity.x * deltaTime;
        particle.velocity.y += gravity.y * deltaTime;
        particle.velocity.z += gravity.z * deltaTime;

        particle.position.x += particle.velocity.x * deltaTime;
        particle.position.y += particle.velocity.y * deltaTime;
        particle.position.z += particle.velocity.z * deltaTime;

        particle.age += deltaTime;
    }

    m_splashParticles.erase(
        std::remove_if(m_splashParticles.begin(), m_splashParticles.end(), [](const SplashParticle& p)
            {
                return p.age >= p.lifetime;
            }),
        m_splashParticles.end());
}

void GameScene::UploadSplashVertices(const Camera* camera)
{
    if (!camera)
    {
        return;
    }

    if (m_splashParticles.empty())
    {
        m_splashVertexCount = 0;
        return;
    }

    const size_t vertexPerParticle = 6;
    size_t requiredVertices = m_splashParticles.size() * vertexPerParticle;
    if (!m_splashVB || m_splashVertexCapacity < requiredVertices)
    {
        m_splashVB = std::make_unique<VertexBuffer>(
            sizeof(ColorVertex) * requiredVertices,
            sizeof(ColorVertex),
            nullptr);
        if (!m_splashVB || !m_splashVB->IsValid())
        {
            m_splashVertexCount = 0;
            m_splashVertexCapacity = 0;
            return;
        }
        m_splashVertexCapacity = requiredVertices;
    }

    std::vector<ColorVertex> vertices(requiredVertices);

    XMMATRIX invView = XMMatrixInverse(nullptr, camera->GetViewMatrix());
    XMVECTOR right = XMVector3Normalize(invView.r[0]);
    XMVECTOR up = XMVector3Normalize(invView.r[1]);

    size_t v = 0;
    for (const auto& particle : m_splashParticles)
    {
        float lifeRatio = std::clamp(particle.age / particle.lifetime, 0.0f, 1.0f);
        float alpha = std::clamp(1.0f - lifeRatio, 0.0f, 1.0f);
        float size = particle.size * (0.6f + 0.4f * (1.0f - lifeRatio));

        XMVECTOR center = XMLoadFloat3(&particle.position);
        XMVECTOR halfRight = XMVectorScale(right, size);
        XMVECTOR halfUp = XMVectorScale(up, size);

        XMVECTOR corners[4];
        corners[0] = XMVectorSubtract(XMVectorSubtract(center, halfRight), halfUp);
        corners[1] = XMVectorAdd(XMVectorSubtract(center, halfUp), halfRight);
        corners[2] = XMVectorAdd(center, XMVectorAdd(halfRight, halfUp));
        corners[3] = XMVectorSubtract(XMVectorAdd(center, halfUp), halfRight);

        XMFLOAT4 color{
            0.4f + 0.3f * (particle.initialStrength * 0.05f),
            0.7f + 0.2f * (1.0f - lifeRatio),
            1.0f,
            alpha * 0.85f };

        XMStoreFloat3(&vertices[v + 0].position, corners[0]);
        vertices[v + 0].color = color;
        XMStoreFloat3(&vertices[v + 1].position, corners[1]);
        vertices[v + 1].color = color;
        XMStoreFloat3(&vertices[v + 2].position, corners[2]);
        vertices[v + 2].color = color;
        XMStoreFloat3(&vertices[v + 3].position, corners[0]);
        vertices[v + 3].color = color;
        XMStoreFloat3(&vertices[v + 4].position, corners[2]);
        vertices[v + 4].color = color;
        XMStoreFloat3(&vertices[v + 5].position, corners[3]);
        vertices[v + 5].color = color;
        v += vertexPerParticle;
    }

    void* mapped = nullptr;
    if (SUCCEEDED(m_splashVB->GetResource()->Map(0, nullptr, &mapped)) && mapped)
    {
        std::memcpy(mapped, vertices.data(), sizeof(ColorVertex) * vertices.size());
        m_splashVB->GetResource()->Unmap(0, nullptr);
        m_splashVertexCount = static_cast<UINT>(vertices.size());
    }
    else
    {
        m_splashVertexCount = 0;
    }
}

