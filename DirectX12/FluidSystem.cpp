#include "FluidSystem.h"
#include "Engine.h"
#include "RandomUtil.h"
#include "ComputePipelineState.h"
#include <d3dcompiler.h>
#include <d3dx12.h>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <random>
#include <filesystem>

#pragma comment(lib, "d3dcompiler.lib")

using namespace DirectX;

namespace
{
    std::wstring ResolveShaderPath(const std::wstring& fileName)
    {
        std::vector<std::filesystem::path> searchDirectories;
        searchDirectories.push_back(std::filesystem::current_path());

        std::filesystem::path exeDir;
#ifdef _WIN32
        wchar_t path[MAX_PATH] = {};
        DWORD length = GetModuleFileNameW(nullptr, path, MAX_PATH);
        if (length > 0 && length < MAX_PATH)
        {
            exeDir = std::filesystem::path(path).parent_path();
        }
#endif
        if (!exeDir.empty())
        {
            auto iter = exeDir;
            for (int i = 0; i < 4 && !iter.empty(); ++i)
            {
                searchDirectories.push_back(iter);
                iter = iter.parent_path();
            }
        }

        for (const auto& dir : searchDirectories)
        {
            std::filesystem::path candidate = dir / fileName;
            if (std::filesystem::exists(candidate))
            {
                return candidate.wstring();
            }
        }
        return fileName;
    }

    bool LoadOrCompileShader(const std::wstring& sourceName, const char* entryPoint, const char* target,
        Microsoft::WRL::ComPtr<ID3DBlob>& outBlob)
    {
        std::filesystem::path sourcePath = ResolveShaderPath(sourceName);
        std::filesystem::path csoPath = sourcePath;
        csoPath.replace_extension(L".cso");

        if (std::filesystem::exists(csoPath))
        {
            if (SUCCEEDED(D3DReadFileToBlob(csoPath.c_str(), &outBlob)))
            {
                return true;
            }
        }

        if (!std::filesystem::exists(sourcePath))
        {
            wprintf(L"FluidSystem: シェーダーファイル %ls が見つかりません\n", sourceName.c_str());
            return false;
        }

        UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
        flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

        Microsoft::WRL::ComPtr<ID3DBlob> error;
        HRESULT hr = D3DCompileFromFile(
            sourcePath.c_str(),
            nullptr,
            D3D_COMPILE_STANDARD_FILE_INCLUDE,
            entryPoint,
            target,
            flags,
            0,
            &outBlob,
            &error);

        if (FAILED(hr))
        {
            if (error)
            {
                printf("FluidSystem: %s\n", static_cast<const char*>(error->GetBufferPointer()));
            }
            else
            {
                wprintf(L"FluidSystem: シェーダー %ls のコンパイルに失敗しました (0x%08X)\n", sourcePath.c_str(), hr);
            }
            return false;
        }

        if (!csoPath.empty())
        {
            D3DWriteBlobToFile(outBlob.Get(), csoPath.c_str(), TRUE);
        }

        return true;
    }

    // GPUバッファ用の粒子構造体
    struct GPUFluidParticle
    {
        XMFLOAT3 position;
        float    pad0 = 0.0f;
        XMFLOAT3 velocity;
        float    pad1 = 0.0f;
    };

    struct ParticleMetaGPU
    {
        XMFLOAT3 position;
        float    radius;
    };

    // PBFカーネル関数（Poly6）
    float Poly6(float r, float h)
    {
        if (r >= h)
        {
            return 0.0f;
        }
        float diff = h * h - r * r;
        float coeff = 315.0f / (64.0f * XM_PI * std::pow(h, 9));
        return coeff * diff * diff * diff;
    }

    // PBFカーネルの勾配（Spiky）
    XMFLOAT3 GradSpiky(const XMFLOAT3& rij, float r, float h)
    {
        if (r <= 1e-6f || r >= h)
        {
            return XMFLOAT3(0.0f, 0.0f, 0.0f);
        }
        float coeff = -45.0f / (XM_PI * std::pow(h, 6));
        float scale = coeff * (h - r) * (h - r) / r;
        return XMFLOAT3(rij.x * scale, rij.y * scale, rij.z * scale);
    }

    inline float Length(const XMFLOAT3& v)
    {
        return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }
}

FluidMaterial CreateFluidMaterial(FluidMaterialPreset preset)
{
    FluidMaterial material{};
    switch (preset)
    {
    case FluidMaterialPreset::Magma:
        material.restDensity = 1500.0f;
        material.particleMass = 1.2f;
        material.smoothingRadius = 0.14f;
        material.viscosity = 0.25f;      // 高い粘性でゆっくり流れる
        material.stiffness = 350.0f;
        material.renderRadius = 0.12f;
        material.lambdaEpsilon = 150.0f;
        material.xsphC = 0.15f;
        material.solverIterations = 6;
        break;
    case FluidMaterialPreset::Water:
    default:
        material = FluidMaterial();
        break;
    }
    return material;
}

FluidSystem::FluidSystem()
    : m_spatialGrid(0.12f)
{
    m_material = CreateFluidMaterial(FluidMaterialPreset::Water);
    m_boundsMin = XMFLOAT3(-2.0f, 0.0f, -2.0f);
    m_boundsMax = XMFLOAT3(2.0f, 4.0f, 2.0f);
    m_gridDim = XMUINT3(1, 1, 1);
    // 見た目系のデフォルト値を設定
    m_waterColorDeep = XMFLOAT3(0.07f, 0.22f, 0.38f);
    m_waterColorShallow = XMFLOAT3(0.25f, 0.55f, 0.95f);
}

FluidSystem::~FluidSystem()
{
    // GPU用フェンスイベントを確実にクローズしてリークを防ぐ
    if (m_computeFenceEvent)
    {
        CloseHandle(m_computeFenceEvent);
        m_computeFenceEvent = nullptr;
    }
}

void FluidSystem::Init(ID3D12Device* device, DXGI_FORMAT rtvFormat, UINT maxParticles, UINT threadGroupCount)
{
    (void)threadGroupCount;
    m_device = device;
    m_rtvFormat = rtvFormat;
    m_maxParticles = std::max<UINT>(1u, maxParticles);
    m_cpuParticles.clear();
    m_cpuParticles.reserve(m_maxParticles);
    m_particleCount = 0;

    UpdateGridSettings();

    CreateGPUResources(device);

    if (!CreateSSFRResources(device, rtvFormat))
    {
        printf("FluidSystem ERROR: SSFR用リソースの初期化に失敗したため描画品質を低下させます\n");
    }

    // GPU・CPUリソースの生成が完了したタイミングで初期化済みフラグを立てる
    m_initialized = true;

    // ひとまず初期状態として軽く粒子を生成しておく
    SpawnParticlesSphere(XMFLOAT3(0.0f, 1.0f, 0.0f), 0.6f, m_maxParticles / 2);

    UpdateParticleBuffer();
}

void FluidSystem::UseGPU(bool enable)
{
    if (!m_initialized)
    {
        return;
    }

    if (enable && !m_gpuAvailable)
    {
        CreateGPUResources(m_device);
    }

    m_useGPU = enable && m_gpuAvailable;
}

FluidSimulationMode FluidSystem::Mode() const
{
    return (m_useGPU && m_gpuAvailable) ? FluidSimulationMode::GPU : FluidSimulationMode::CPU;
}

void FluidSystem::SetMaterialPreset(FluidMaterialPreset preset)
{
    SetMaterial(CreateFluidMaterial(preset));
}

void FluidSystem::SetMaterial(const FluidMaterial& material)
{
    m_material = material;
    m_spatialGrid.SetCellSize(m_material.smoothingRadius);
    UpdateGridSettings();

    m_particleCount = static_cast<UINT>(std::min<size_t>(m_cpuParticles.size(), m_maxParticles));

    m_cpuDirty = true;
    m_gpuDirty = true;

    // マテリアル変更時はGPUリソースも再作成して整合を取る
    if (m_device)
    {
        CreateGPUResources(m_device);
        CreateSSFRResources(m_device, m_rtvFormat);
    }
}

void FluidSystem::SetSimulationBounds(const XMFLOAT3& minBound, const XMFLOAT3& maxBound)
{
    // 境界設定を更新し、SPHの探索グリッドも再構築する
    m_boundsMin = minBound;
    m_boundsMax = maxBound;
    UpdateGridSettings();
}

// ============================
// 流体生成
// ============================
void FluidSystem::SpawnParticlesSphere(const XMFLOAT3& center, float radius, UINT count)
{
    if (!m_initialized || count == 0)
    {
        return;
    }

    std::mt19937 rng{ std::random_device{}() };
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (UINT i = 0; i < count && m_particleCount < m_maxParticles; ++i)
    {
        float u = dist(rng);
        float v = dist(rng);
        float theta = 2.0f * XM_PI * u;
        float phi = std::acos(2.0f * v - 1.0f);
        float r = radius * std::cbrt(dist(rng));

        XMFLOAT3 offset{
            r * std::sin(phi) * std::cos(theta),
            r * std::cos(phi),
            r * std::sin(phi) * std::sin(theta)
        };

        FluidParticle particle{};
        particle.position = XMFLOAT3(center.x + offset.x, center.y + offset.y, center.z + offset.z);
        particle.velocity = XMFLOAT3(0.0f, 0.0f, 0.0f);
        particle.predicted = particle.position;
        particle.density = m_material.restDensity;
        particle.lambda = 0.0f;
        particle.collisionMask = 0;

        m_cpuParticles.push_back(particle);
        ++m_particleCount;
    }

    m_cpuDirty = true;
    m_gpuDirty = true;
}

// ============================
// 流体削除
// ============================
void FluidSystem::RemoveParticlesSphere(const XMFLOAT3& center, float radius)
{
    if (!m_initialized || m_particleCount == 0)
    {
        return;
    }

    float r2 = radius * radius;
    auto it = std::remove_if(m_cpuParticles.begin(), m_cpuParticles.end(), [&](const FluidParticle& p)
        {
            XMFLOAT3 diff{ p.position.x - center.x, p.position.y - center.y, p.position.z - center.z };
            float len2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
            return len2 <= r2;
        });
    m_cpuParticles.erase(it, m_cpuParticles.end());
    m_particleCount = static_cast<UINT>(std::min<size_t>(m_cpuParticles.size(), m_maxParticles));

    m_cpuDirty = true;
    m_gpuDirty = true;
}


void FluidSystem::QueueGather(const XMFLOAT3& target, float radius, float strength)
{
    m_gatherOps.push_back({ target, radius, strength });
}

void FluidSystem::QueueSplash(const XMFLOAT3& position, float radius, float impulse)
{
    m_splashOps.push_back({ position, radius, impulse });
}

void FluidSystem::QueueDirectionalImpulse(const XMFLOAT3& center, float radius, const XMFLOAT3& direction, float strength)
{
    XMVECTOR dirVec = XMLoadFloat3(&direction);
    if (XMVectorGetX(XMVector3LengthSq(dirVec)) < 1e-6f)
    {
        return;
    }
    DirectionalImpulseOperation op{};
    op.center = center;
    op.radius = radius;
    op.direction = direction;
    op.strength = strength;
    m_directionalOps.push_back(op);
}

void FluidSystem::ClearDynamicOperations()
{
    m_gatherOps.clear();
    m_splashOps.clear();
    m_directionalOps.clear();
}

float FluidSystem::EffectiveTimeStep(float dt) const
{
    const float minStep = 1.0f / 240.0f;
    const float maxStep = 1.0f / 30.0f;
    return std::clamp(dt, minStep, maxStep);
}

void FluidSystem::Simulate(ID3D12GraphicsCommandList* cmd, float dt)
{
    if (!m_initialized)
    {
        return;
    }

    m_particleCount = static_cast<UINT>(std::min<size_t>(m_cpuParticles.size(), m_maxParticles));
    if (m_particleCount == 0)
    {
        return;
    }

    float step = EffectiveTimeStep(dt);

    if (Mode() == FluidSimulationMode::GPU)
    {
        if (!HasValidGPUResources())
        {
            // GPU用リソースが欠けている場合は安全のためCPUシミュレーションへ切り替える
            printf("FluidSystem ERROR: 必要なGPUリソースが未初期化のためCPUシミュレーションにフォールバックします\n");
            m_gpuAvailable = false;

            ApplyExternalOperationsCPU(step);
            StepCPU(step);
            UpdateParticleBuffer();
            m_activeMetaSRV = m_cpuMetaSRV;
            return;
        }

        // 前フレームの結果を読み戻してCPU側と同期
        if (m_pendingReadback)
        {
            ReadbackGPUToCPU();
        }

        ApplyExternalOperationsCPU(step);
        UpdateComputeParams(step);

        // GPU用コマンドリストをリセットして記録を開始
        ID3D12GraphicsCommandList* computeCmd = BeginComputeCommandList();
        if (!computeCmd)
        {
            // 取得に失敗した場合はGPUモードを諦めてCPUシミュレーションにフォールバック
            StepCPU(step);
            UpdateParticleBuffer();
            m_activeMetaSRV = m_cpuMetaSRV;
            return;
        }

        UploadCPUToGPU(computeCmd);
        StepGPU(computeCmd, step);
        SubmitComputeCommandList();
        m_activeMetaSRV = m_gpuMetaSRV;
    }
    else
    {
        ApplyExternalOperationsCPU(step);
        StepCPU(step);
        UpdateParticleBuffer();
        m_activeMetaSRV = m_cpuMetaSRV;
    }

    // 累積時間はシェーダー側のアニメーションに利用する
    m_totalSimulatedTime += step;
    if (m_totalSimulatedTime > 10000.0f)
    {
        m_totalSimulatedTime = std::fmod(m_totalSimulatedTime, 10000.0f);
    }
}


void FluidSystem::Render(ID3D12GraphicsCommandList* cmd,
    const XMFLOAT4X4& view,
    const XMFLOAT4X4& proj,
    const XMFLOAT4X4& viewProj,
    const XMFLOAT3& camPos,
    float isoLevel)
{
    (void)isoLevel;
    (void)camPos;

    if (!m_initialized || !cmd || m_particleCount == 0 ||
        !m_particleRootSignature || !m_particlePipelineState ||
        !m_blurPipelineState || !m_normalPipelineState ||
        !m_particleDepthTexture || !m_smoothedDepthTexture ||
        !m_normalTexture || !m_thicknessTexture ||
        !m_particleDepthSRV || !m_particleDepthUAV || !m_smoothedDepthSRV || !m_smoothedDepthUAV ||
        !m_normalSRV || !m_normalUAV || !m_thicknessSRV || !m_thicknessUAV)
    {
        return;
    }

    if (!m_activeMetaSRV)
    {
        // 粒子SRVが未設定の場合は描画できないため早期リターン
        return;
    }

    UINT frameIndex = g_Engine->CurrentBackBufferIndex();
    auto& cameraCB = m_cameraCB[frameIndex];
    if (!cameraCB)
    {
        return;
    }

    UINT width = std::max<UINT>(1u, g_Engine->FrameBufferWidth());
    UINT height = std::max<UINT>(1u, g_Engine->FrameBufferHeight());

    SSFRCameraConstants* camera = cameraCB->GetPtr<SSFRCameraConstants>();
    XMMATRIX viewMatrix = XMLoadFloat4x4(&view);
    XMMATRIX projMatrix = XMLoadFloat4x4(&proj);
    XMMATRIX viewProjMatrix = XMLoadFloat4x4(&viewProj);

    XMStoreFloat4x4(&camera->View, XMMatrixTranspose(viewMatrix));
    XMStoreFloat4x4(&camera->Proj, XMMatrixTranspose(projMatrix));
    XMStoreFloat4x4(&camera->ViewProj, XMMatrixTranspose(viewProjMatrix));
    camera->ScreenSize = XMFLOAT2(static_cast<float>(width), static_cast<float>(height));
    camera->InvScreen = XMFLOAT2(1.0f / static_cast<float>(width), 1.0f / static_cast<float>(height));
    camera->NearZ = 0.1f;
    camera->FarZ = 1000.0f;
    float f0 = std::clamp(m_reflectionStrength, 0.0f, 1.0f);
    camera->IorF0 = XMFLOAT3(f0, f0, f0);
    camera->Absorption = m_waterAbsorption;

    // ブラー定数は1度だけ設定しておけばよいので、呼び出し毎に更新不要
    if (m_blurParamsCB)
    {
        auto* blur = m_blurParamsCB->GetPtr<SSFRBlurParams>();
        blur->Sigma = std::max(0.1f, blur->Sigma);
        blur->DepthK = std::max(0.01f, blur->DepthK);
        blur->NormalK = std::max(0.1f, blur->NormalK);
    }

    ID3D12DescriptorHeap* heaps[] = { g_Engine->CbvSrvUavHeap()->GetHeap() };
    cmd->SetDescriptorHeaps(1, heaps);

    auto transition = [&](std::unique_ptr<Texture2D>& texture, D3D12_RESOURCE_STATES targetState, D3D12_RESOURCE_STATES& currentState)
    {
        if (texture && currentState != targetState)
        {
            auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(texture->Resource(), currentState, targetState);
            cmd->ResourceBarrier(1, &barrier);
            currentState = targetState;
        }
    };

    // 粒子深度パス（UAVへ書き込み）
    transition(m_particleDepthTexture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, m_particleDepthState);
    transition(m_smoothedDepthTexture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, m_smoothedDepthState);
    transition(m_thicknessTexture, D3D12_RESOURCE_STATE_RENDER_TARGET, m_thicknessState);

    const float clearDepth[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    // ClearUnorderedAccessViewFloat ではCPU可視のUAVハンドルが必要なので、専用ヒープで確保したハンドルを使用する
    cmd->ClearUnorderedAccessViewFloat(m_particleDepthUAV->HandleGPU, m_particleDepthUavCpuHandle, m_particleDepthTexture->Resource(), clearDepth, 0, nullptr);
    cmd->ClearUnorderedAccessViewFloat(m_smoothedDepthUAV->HandleGPU, m_smoothedDepthUavCpuHandle, m_smoothedDepthTexture->Resource(), clearDepth, 0, nullptr);
    const float clearThickness[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    cmd->ClearRenderTargetView(m_thicknessRTV, clearThickness, 0, nullptr);

    cmd->SetGraphicsRootSignature(m_particleRootSignature.Get());
    cmd->SetPipelineState(m_particlePipelineState.Get());
    cmd->SetGraphicsRootConstantBufferView(0, cameraCB->GetAddress());
    // StructuredBuffer<ParticleData> を VS へ渡す
    cmd->SetGraphicsRootDescriptorTable(1, m_activeMetaSRV->HandleGPU);
    cmd->SetGraphicsRootDescriptorTable(2, m_particleDepthUAV->HandleGPU);
    cmd->SetGraphicsRootDescriptorTable(3, m_smoothedDepthUAV->HandleGPU);
    cmd->SetGraphicsRootDescriptorTable(4, m_thicknessUAV->HandleGPU);
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd->DrawInstanced(3, 1, 0, 0);

    // 深度平滑化（コンピュート）
    transition(m_particleDepthTexture, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, m_particleDepthState);
    transition(m_smoothedDepthTexture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, m_smoothedDepthState);

    if (m_blurPipelineState)
    {
        cmd->SetComputeRootSignature(m_blurRootSignature.Get());
        cmd->SetPipelineState(m_blurPipelineState.Get());
        cmd->SetComputeRootDescriptorTable(0, m_particleDepthSRV->HandleGPU);
        cmd->SetComputeRootDescriptorTable(1, m_particleDepthSRV->HandleGPU);
        cmd->SetComputeRootDescriptorTable(2, m_smoothedDepthUAV->HandleGPU);
        // CameraCB (register b0) を必ずバインドする
        cmd->SetComputeRootConstantBufferView(3, cameraCB->GetAddress());
        if (m_blurParamsCB)
        {
            // BilateralParams (register b1) をバインドする
            cmd->SetComputeRootConstantBufferView(4, m_blurParamsCB->GetAddress());
        }
        UINT groupX = (width + 15) / 16;
        UINT groupY = (height + 15) / 16;
        cmd->Dispatch(groupX, groupY, 1);
    }

    // 法線生成
    transition(m_smoothedDepthTexture, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, m_smoothedDepthState);
    transition(m_normalTexture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, m_normalState);

    if (m_normalPipelineState)
    {
        cmd->SetComputeRootSignature(m_normalRootSignature.Get());
        cmd->SetPipelineState(m_normalPipelineState.Get());
        cmd->SetComputeRootDescriptorTable(0, m_smoothedDepthSRV->HandleGPU);
        cmd->SetComputeRootDescriptorTable(1, m_normalUAV->HandleGPU);
        cmd->SetComputeRootConstantBufferView(2, cameraCB->GetAddress());
        UINT groupX = (width + 31) / 32;
        UINT groupY = (height + 31) / 32;
        cmd->Dispatch(groupX, groupY, 1);
    }

    // 合成のために必要なリソース状態へ戻す
    transition(m_particleDepthTexture, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, m_particleDepthState);
    transition(m_smoothedDepthTexture, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, m_smoothedDepthState);
    transition(m_normalTexture, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, m_normalState);
    transition(m_thicknessTexture, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, m_thicknessState);
}

void FluidSystem::Composite(ID3D12GraphicsCommandList* cmd,
    ID3D12Resource* sceneColor,
    ID3D12Resource* sceneDepth,
    D3D12_CPU_DESCRIPTOR_HANDLE sceneRTV)
{
    if (!m_initialized || !cmd || m_particleCount == 0 ||
        !sceneColor || !sceneDepth ||
        !m_sceneColorCopy || !m_sceneColorSRV || !m_sceneDepthSRV ||
        !m_smoothedDepthSRV || !m_normalSRV || !m_thicknessSRV ||
        !m_compositeRootSignature || !m_compositePipelineState)
    {
        return;
    }

    UINT frameIndex = g_Engine->CurrentBackBufferIndex();
    auto& cameraCB = m_cameraCB[frameIndex];
    if (!cameraCB)
    {
        return;
    }

    ID3D12DescriptorHeap* heaps[] = { g_Engine->CbvSrvUavHeap()->GetHeap() };
    cmd->SetDescriptorHeaps(1, heaps);

    // 1. シーンカラーをサンプルするためバックバッファをコピー元へ遷移
    auto toCopySource = CD3DX12_RESOURCE_BARRIER::Transition(
        sceneColor,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_COPY_SOURCE);
    cmd->ResourceBarrier(1, &toCopySource);

    // 2. コピー先テクスチャを COPY_DEST へ揃える
    if (m_sceneColorCopyState != D3D12_RESOURCE_STATE_COPY_DEST)
    {
        auto toCopyDest = CD3DX12_RESOURCE_BARRIER::Transition(
            m_sceneColorCopy.Get(),
            m_sceneColorCopyState,
            D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->ResourceBarrier(1, &toCopyDest);
        m_sceneColorCopyState = D3D12_RESOURCE_STATE_COPY_DEST;
    }

    // 3. バックバッファの内容をシェーダー参照用テクスチャにコピー
    cmd->CopyResource(m_sceneColorCopy.Get(), sceneColor);

    // 4. コピー結果をピクセルシェーダー用に遷移
    auto toSceneSRV = CD3DX12_RESOURCE_BARRIER::Transition(
        m_sceneColorCopy.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &toSceneSRV);
    m_sceneColorCopyState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    // 5. バックバッファを再度レンダーターゲット状態に戻す
    auto toRenderTarget = CD3DX12_RESOURCE_BARRIER::Transition(
        sceneColor,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_RENDER_TARGET);
    cmd->ResourceBarrier(1, &toRenderTarget);

    // 6. 深度バッファをピクセルシェーダーから参照できる状態へ
    if (m_sceneDepthState != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
    {
        auto toDepthSRV = CD3DX12_RESOURCE_BARRIER::Transition(
            sceneDepth,
            m_sceneDepthState,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        cmd->ResourceBarrier(1, &toDepthSRV);
        m_sceneDepthState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    }

    // 7. 合成先のRTVを再設定（深度は不要なので未設定）
    cmd->OMSetRenderTargets(1, &sceneRTV, FALSE, nullptr);

    // 8. フルスクリーン三角形でシーンカラーと流体テクスチャを合成
    cmd->SetGraphicsRootSignature(m_compositeRootSignature.Get());
    cmd->SetPipelineState(m_compositePipelineState.Get());
    cmd->SetGraphicsRootDescriptorTable(0, m_smoothedDepthSRV->HandleGPU);
    cmd->SetGraphicsRootDescriptorTable(1, m_normalSRV->HandleGPU);
    cmd->SetGraphicsRootDescriptorTable(2, m_thicknessSRV->HandleGPU);
    cmd->SetGraphicsRootDescriptorTable(3, m_sceneDepthSRV->HandleGPU);
    cmd->SetGraphicsRootDescriptorTable(4, m_sceneColorSRV->HandleGPU);
    cmd->SetGraphicsRootConstantBufferView(5, cameraCB->GetAddress());
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd->DrawInstanced(3, 1, 0, 0);

    // 9. 深度バッファを次のフレームで書き込めるように戻す
    auto toDepthWrite = CD3DX12_RESOURCE_BARRIER::Transition(
        sceneDepth,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_DEPTH_WRITE);
    cmd->ResourceBarrier(1, &toDepthWrite);
    m_sceneDepthState = D3D12_RESOURCE_STATE_DEPTH_WRITE;

    // 10. シーンカラーのコピーを次フレームのコピー先に戻す
    auto resetSceneCopy = CD3DX12_RESOURCE_BARRIER::Transition(
        m_sceneColorCopy.Get(),
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST);
    cmd->ResourceBarrier(1, &resetSceneCopy);
    m_sceneColorCopyState = D3D12_RESOURCE_STATE_COPY_DEST;
}

void FluidSystem::SetWaterAppearance(const XMFLOAT3& shallowColor,
    const XMFLOAT3& deepColor,
    float absorption,
    float foamThreshold,
    float foamStrength,
    float reflectionStrength,
    float specularPower)
{
    // ゲーム側で水の雰囲気をコントロールしやすいようにクランプしてから採用
    m_waterColorShallow = shallowColor;
    m_waterColorDeep = deepColor;
    m_waterAbsorption = std::max(absorption, 0.0f);
    m_foamCurvatureThreshold = std::clamp(foamThreshold, 0.05f, 1.5f);
    m_foamStrength = std::clamp(foamStrength, 0.0f, 1.0f);
    m_reflectionStrength = std::clamp(reflectionStrength, 0.0f, 1.0f);
    m_specularPower = std::max(specularPower, 1.0f);
}

void FluidSystem::ApplyExternalOperationsCPU(float dt)
{
    if (m_cpuParticles.empty())
    {
        return;
    }

    bool modified = false;

    // 集束処理
    for (const auto& op : m_gatherOps)
    {
        XMVECTOR target = XMLoadFloat3(&op.target);
        for (auto& particle : m_cpuParticles)
        {
            XMVECTOR pos = XMLoadFloat3(&particle.position);
            XMVECTOR diff = XMVectorSubtract(target, pos);
            float dist = XMVectorGetX(XMVector3Length(diff));
            if (dist < op.radius && dist > 1e-5f)
            {
                float weight = 1.0f - (dist / op.radius);
                float accel = op.strength * weight;
                XMVECTOR dir = XMVector3Normalize(diff);
                XMVECTOR vel = XMLoadFloat3(&particle.velocity);
                vel = XMVectorAdd(vel, XMVectorScale(dir, accel * dt));
                XMStoreFloat3(&particle.velocity, vel);
                modified = true;
            }
        }
    }

    // 発散処理は1回で取り除く
    if (!m_splashOps.empty())
    {
        for (const auto& op : m_splashOps)
        {
            XMVECTOR origin = XMLoadFloat3(&op.origin);
            for (auto& particle : m_cpuParticles)
            {
                XMVECTOR pos = XMLoadFloat3(&particle.position);
                XMVECTOR diff = XMVectorSubtract(pos, origin);
                float dist = XMVectorGetX(XMVector3Length(diff));
                if (dist < op.radius && dist > 1e-5f)
                {
                    float weight = 1.0f - (dist / op.radius);
                    XMVECTOR dir = XMVector3Normalize(diff);
                    XMVECTOR vel = XMLoadFloat3(&particle.velocity);
                    vel = XMVectorAdd(vel, XMVectorScale(dir, op.impulse * weight));
                    XMStoreFloat3(&particle.velocity, vel);
                    modified = true;
                }
            }
        }
        m_splashOps.clear();
    }

    // 指向性インパルス（例：前方へ飛ばす操作）
    if (!m_directionalOps.empty())
    {
        for (const auto& op : m_directionalOps)
        {
            XMVECTOR center = XMLoadFloat3(&op.center);
            XMVECTOR dir = XMLoadFloat3(&op.direction);
            dir = XMVector3Normalize(dir);
            for (auto& particle : m_cpuParticles)
            {
                XMVECTOR pos = XMLoadFloat3(&particle.position);
                XMVECTOR diff = XMVectorSubtract(pos, center);
                float dist = XMVectorGetX(XMVector3Length(diff));
                if (dist < op.radius)
                {
                    float weight = 1.0f - (dist / op.radius);
                    XMVECTOR vel = XMLoadFloat3(&particle.velocity);
                    vel = XMVectorAdd(vel, XMVectorScale(dir, op.strength * weight));
                    XMStoreFloat3(&particle.velocity, vel);
                    modified = true;
                }
            }
        }
        m_directionalOps.clear();
    }

    if (modified)
    {
        m_cpuDirty = true;
    }
}

void FluidSystem::StepCPU(float dt)
{
    if (m_cpuParticles.empty())
    {
        return;
    }

    const float h = m_material.smoothingRadius;
    const float mass = m_material.particleMass;
    const float restDensity = m_material.restDensity;
    const float epsilon = m_material.lambdaEpsilon;
    const float sCorrK = -0.001f;
    const float sCorrN = 4.0f;
    const float deltaQ = 0.3f * h;
    const float invRestDensity = 1.0f / restDensity;
    const XMFLOAT3 gravity = XMFLOAT3(0.0f, -9.8f, 0.0f);

    m_spatialGrid.Clear();

    // 外力適用と予測位置計算
    for (size_t i = 0; i < m_cpuParticles.size(); ++i)
    {
        auto& particle = m_cpuParticles[i];
        particle.collisionMask = 0;
        XMVECTOR vel = XMLoadFloat3(&particle.velocity);
        XMVECTOR grav = XMLoadFloat3(&gravity);
        vel = XMVectorAdd(vel, XMVectorScale(grav, dt));
        XMStoreFloat3(&particle.velocity, vel);

        XMVECTOR pos = XMLoadFloat3(&particle.position);
        XMVECTOR pred = XMVectorAdd(pos, XMVectorScale(vel, dt));
        XMStoreFloat3(&particle.predicted, pred);
        particle.lambda = 0.0f;

        m_spatialGrid.Insert(i, particle.predicted);
    }

    std::vector<size_t> neighbors;
    neighbors.reserve(64);
    std::vector<size_t> xsphNeighbors;
    xsphNeighbors.reserve(64);

    for (int iteration = 0; iteration < m_material.solverIterations; ++iteration)
    {
        // λ計算
        for (size_t i = 0; i < m_cpuParticles.size(); ++i)
        {
            FluidParticle& pi = m_cpuParticles[i];
            neighbors.clear();
            m_spatialGrid.Query(pi.predicted, h, neighbors);

            float density = 0.0f;
            for (size_t idx : neighbors)
            {
                const FluidParticle& pj = m_cpuParticles[idx];
                XMFLOAT3 rij{ pi.predicted.x - pj.predicted.x, pi.predicted.y - pj.predicted.y, pi.predicted.z - pj.predicted.z };
                float r = Length(rij);
                density += mass * Poly6(r, h);
            }
            pi.density = std::max(density, restDensity * 0.1f);
            float Ci = pi.density * invRestDensity - 1.0f;

            XMFLOAT3 gradSum{ 0.0f, 0.0f, 0.0f };
            float sumGrad2 = 0.0f;

            for (size_t idx : neighbors)
            {
                if (idx == i)
                {
                    continue;
                }
                const FluidParticle& pj = m_cpuParticles[idx];
                XMFLOAT3 rij{ pi.predicted.x - pj.predicted.x, pi.predicted.y - pj.predicted.y, pi.predicted.z - pj.predicted.z };
                float r = Length(rij);
                XMFLOAT3 grad = GradSpiky(rij, r, h);
                grad.x *= mass * invRestDensity;
                grad.y *= mass * invRestDensity;
                grad.z *= mass * invRestDensity;
                gradSum.x += grad.x;
                gradSum.y += grad.y;
                gradSum.z += grad.z;
                sumGrad2 += grad.x * grad.x + grad.y * grad.y + grad.z * grad.z;
            }

            sumGrad2 += gradSum.x * gradSum.x + gradSum.y * gradSum.y + gradSum.z * gradSum.z;
            pi.lambda = -Ci / (sumGrad2 + epsilon);
        }

        // Δp計算
        for (size_t i = 0; i < m_cpuParticles.size(); ++i)
        {
            FluidParticle& pi = m_cpuParticles[i];
            neighbors.clear();
            m_spatialGrid.Query(pi.predicted, h, neighbors);

            XMFLOAT3 delta{ 0.0f, 0.0f, 0.0f };
            for (size_t idx : neighbors)
            {
                if (idx == i)
                {
                    continue;
                }
                const FluidParticle& pj = m_cpuParticles[idx];
                XMFLOAT3 rij{ pi.predicted.x - pj.predicted.x, pi.predicted.y - pj.predicted.y, pi.predicted.z - pj.predicted.z };
                float r = Length(rij);
                if (r >= h)
                {
                    continue;
                }
                float w = Poly6(r, h);
                float corr = 0.0f;
                float wq = Poly6(deltaQ, h);
                if (wq > 0.0f)
                {
                    corr = sCorrK * std::pow(w / wq, sCorrN);
                }
                XMFLOAT3 grad = GradSpiky(rij, r, h);
                float factor = (pi.lambda + pj.lambda + corr) * mass * invRestDensity;
                delta.x += grad.x * factor;
                delta.y += grad.y * factor;
                delta.z += grad.z * factor;
            }

            pi.predicted.x += delta.x;
            pi.predicted.y += delta.y;
            pi.predicted.z += delta.z;

            pi.collisionMask |= ResolveBounds(pi);
        }
    }

    // 速度と位置の更新（XSPHによる安定化込み）
    for (auto& particle : m_cpuParticles)
    {
        XMFLOAT3 delta{ particle.predicted.x - particle.position.x,
            particle.predicted.y - particle.position.y,
            particle.predicted.z - particle.position.z };
        particle.velocity = XMFLOAT3(delta.x / dt, delta.y / dt, delta.z / dt);

        const float preBounceSpeed = XMVectorGetX(XMVector3Length(XMLoadFloat3(&particle.velocity)));

        // XSPH粘性
        xsphNeighbors.clear();
        m_spatialGrid.Query(particle.predicted, h, xsphNeighbors);
        XMFLOAT3 xsph{ 0.0f, 0.0f, 0.0f };
        for (size_t idx : xsphNeighbors)
        {
            if (&particle == &m_cpuParticles[idx])
            {
                continue;
            }
            const FluidParticle& pj = m_cpuParticles[idx];
            XMFLOAT3 vij{ pj.velocity.x - particle.velocity.x,
                pj.velocity.y - particle.velocity.y,
                pj.velocity.z - particle.velocity.z };
            XMFLOAT3 rij{ particle.predicted.x - pj.predicted.x,
                particle.predicted.y - pj.predicted.y,
                particle.predicted.z - pj.predicted.z };
            float r = Length(rij);
            xsph.x += Poly6(r, h) * vij.x;
            xsph.y += Poly6(r, h) * vij.y;
            xsph.z += Poly6(r, h) * vij.z;
        }
        particle.velocity.x += m_material.xsphC * xsph.x;
        particle.velocity.y += m_material.xsphC * xsph.y;
        particle.velocity.z += m_material.xsphC * xsph.z;

        if (particle.collisionMask != 0)
        {
            const float restitution = 0.45f; // 境界からの跳ね返り係数
            XMFLOAT3 normal{ 0.0f, 0.0f, 0.0f };

            if ((particle.collisionMask & kCollisionMinX) && particle.velocity.x < 0.0f)
            {
                particle.velocity.x = -particle.velocity.x * restitution;
                normal.x += 1.0f;
            }
            if ((particle.collisionMask & kCollisionMaxX) && particle.velocity.x > 0.0f)
            {
                particle.velocity.x = -particle.velocity.x * restitution;
                normal.x -= 1.0f;
            }
            if ((particle.collisionMask & kCollisionMinY) && particle.velocity.y < 0.0f)
            {
                particle.velocity.y = -particle.velocity.y * restitution;
                normal.y += 1.0f;
            }
            if ((particle.collisionMask & kCollisionMaxY) && particle.velocity.y > 0.0f)
            {
                particle.velocity.y = -particle.velocity.y * restitution;
                normal.y -= 1.0f;
            }
            if ((particle.collisionMask & kCollisionMinZ) && particle.velocity.z < 0.0f)
            {
                particle.velocity.z = -particle.velocity.z * restitution;
                normal.z += 1.0f;
            }
            if ((particle.collisionMask & kCollisionMaxZ) && particle.velocity.z > 0.0f)
            {
                particle.velocity.z = -particle.velocity.z * restitution;
                normal.z -= 1.0f;
            }

            float normalLen = XMVectorGetX(XMVector3Length(XMLoadFloat3(&normal)));
            if (normalLen > 0.0f && preBounceSpeed > 0.2f)
            {
                float invLen = 1.0f / normalLen;
                normal.x *= invLen;
                normal.y *= invLen;
                normal.z *= invLen;
                FluidCollisionEvent evt{};
                evt.position = particle.predicted;
                evt.normal = normal;
                evt.strength = preBounceSpeed;
                m_collisionEvents.push_back(evt);
            }

            particle.collisionMask = 0;
        }

        particle.position = particle.predicted;
    }

    m_cpuDirty = true;
}

UINT FluidSystem::ResolveBounds(FluidParticle& p) const
{
    UINT mask = 0;
    if (p.predicted.x < m_boundsMin.x)
    {
        p.predicted.x = m_boundsMin.x;
        mask |= kCollisionMinX;
    }
    else if (p.predicted.x > m_boundsMax.x)
    {
        p.predicted.x = m_boundsMax.x;
        mask |= kCollisionMaxX;
    }

    if (p.predicted.y < m_boundsMin.y)
    {
        p.predicted.y = m_boundsMin.y;
        mask |= kCollisionMinY;
    }
    else if (p.predicted.y > m_boundsMax.y)
    {
        p.predicted.y = m_boundsMax.y;
        mask |= kCollisionMaxY;
    }

    if (p.predicted.z < m_boundsMin.z)
    {
        p.predicted.z = m_boundsMin.z;
        mask |= kCollisionMinZ;
    }
    else if (p.predicted.z > m_boundsMax.z)
    {
        p.predicted.z = m_boundsMax.z;
        mask |= kCollisionMaxZ;
    }
    return mask;
}

bool FluidSystem::Raycast(const XMFLOAT3& origin, const XMFLOAT3& direction, float maxDistance, float radius, XMFLOAT3& hitPosition) const
{
    if (m_cpuParticles.empty())
    {
        return false;
    }

    XMVECTOR rayOrigin = XMLoadFloat3(&origin);
    XMVECTOR rayDir = XMVector3Normalize(XMLoadFloat3(&direction));
    float bestDistance = maxDistance;
    bool hit = false;

    for (const auto& particle : m_cpuParticles)
    {
        XMVECTOR pos = XMLoadFloat3(&particle.position);
        XMVECTOR diff = XMVectorSubtract(pos, rayOrigin);
        float t = XMVectorGetX(XMVector3Dot(diff, rayDir));
        if (t < 0.0f || t > bestDistance)
        {
            continue;
        }
        XMVECTOR closest = XMVectorAdd(rayOrigin, XMVectorScale(rayDir, t));
        float distanceToRay = XMVectorGetX(XMVector3Length(XMVectorSubtract(pos, closest)));
        if (distanceToRay <= radius)
        {
            bestDistance = t;
            XMStoreFloat3(&hitPosition, pos);
            hit = true;
        }
    }

    return hit;
}

void FluidSystem::PopCollisionEvents(std::vector<FluidCollisionEvent>& outEvents)
{
    if (m_collisionEvents.empty())
    {
        outEvents.clear();
        return;
    }

    outEvents.insert(outEvents.end(), m_collisionEvents.begin(), m_collisionEvents.end());
    m_collisionEvents.clear();
}

void FluidSystem::UploadCPUToGPU(ID3D12GraphicsCommandList* cmd)
{
    if (!m_gpuAvailable || !cmd || !m_gpuUpload)
    {
        return;
    }

    if (!m_cpuDirty && !m_gpuDirty)
    {
        return;
    }

    const UINT64 bufferSize = sizeof(GPUFluidParticle) * m_particleCount;
    if (bufferSize == 0)
    {
        return;
    }

    GPUFluidParticle* mapped = nullptr;
    if (SUCCEEDED(m_gpuUpload->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) && mapped)
    {
        for (UINT i = 0; i < m_particleCount; ++i)
        {
            mapped[i].position = m_cpuParticles[i].position;
            mapped[i].velocity = m_cpuParticles[i].velocity;
        }
        m_gpuUpload->Unmap(0, nullptr);
    }

    for (int i = 0; i < 2; ++i)
    {
        auto& buffer = m_gpuParticleBuffers[i];
        if (!buffer.resource)
        {
            continue;
        }
        // すでに目的の状態と同じならバリアを張らずにスキップする
        if (buffer.state != D3D12_RESOURCE_STATE_COPY_DEST)
        {
            auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(buffer.resource.Get(), buffer.state, D3D12_RESOURCE_STATE_COPY_DEST);
            cmd->ResourceBarrier(1, &toCopy);
            buffer.state = D3D12_RESOURCE_STATE_COPY_DEST;
        }
        cmd->CopyBufferRegion(buffer.resource.Get(), 0, m_gpuUpload.Get(), 0, sizeof(GPUFluidParticle) * m_particleCount);
        if (buffer.state != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        {
            auto toSRV = CD3DX12_RESOURCE_BARRIER::Transition(buffer.resource.Get(), buffer.state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            cmd->ResourceBarrier(1, &toSRV);
            buffer.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
    }

    m_cpuDirty = false;
    m_gpuDirty = false;
}

void FluidSystem::UpdateComputeParams(float dt)
{
    if (!m_computeParamsCB)
    {
        m_computeParamsCB = std::make_unique<ConstantBuffer>(sizeof(GPUParams));
    }

    GPUParams* params = m_computeParamsCB->GetPtr<GPUParams>();
    params->restDensity = m_material.restDensity;
    params->particleMass = m_material.particleMass;
    params->viscosity = m_material.viscosity;
    params->stiffness = m_material.stiffness;
    params->radius = m_material.smoothingRadius;
    params->timeStep = dt;
    params->particleCount = m_particleCount;
    params->pad0 = 0;
    params->gridMin = m_boundsMin;
    params->pad1 = 0.0f;
    params->gridDim = m_gridDim;
    params->pad2 = 0;

    if (!m_dummyViewCB)
    {
        m_dummyViewCB = std::make_unique<ConstantBuffer>(sizeof(XMFLOAT4X4));
        XMFLOAT4X4 identity;
        XMStoreFloat4x4(&identity, XMMatrixIdentity());
        *m_dummyViewCB->GetPtr<XMFLOAT4X4>() = identity;
    }
}

void FluidSystem::StepGPU(ID3D12GraphicsCommandList* cmd, float dt)
{
    if (!m_gpuAvailable || !cmd || !m_buildGridPipeline || !m_particlePipeline || !m_clearGridPipeline)
    {
        return;
    }

    const UINT readIndex = m_gpuReadIndex;
    const UINT writeIndex = 1 - m_gpuReadIndex;

    auto& readBuffer = m_gpuParticleBuffers[readIndex];
    auto& writeBuffer = m_gpuParticleBuffers[writeIndex];

    if (readBuffer.state != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
    {
        auto toSRV = CD3DX12_RESOURCE_BARRIER::Transition(readBuffer.resource.Get(), readBuffer.state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        cmd->ResourceBarrier(1, &toSRV);
        readBuffer.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }

    if (writeBuffer.state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
    {
        auto toUAV = CD3DX12_RESOURCE_BARRIER::Transition(writeBuffer.resource.Get(), writeBuffer.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd->ResourceBarrier(1, &toUAV);
        writeBuffer.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }

    auto metaToUAV = CD3DX12_RESOURCE_BARRIER::Transition(m_gpuMetaBuffer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->ResourceBarrier(1, &metaToUAV);

    auto gridCountToUAV = CD3DX12_RESOURCE_BARRIER::Transition(m_gpuGridCount.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->ResourceBarrier(1, &gridCountToUAV);

    auto gridTableToUAV = CD3DX12_RESOURCE_BARRIER::Transition(m_gpuGridTable.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->ResourceBarrier(1, &gridTableToUAV);

    ID3D12DescriptorHeap* heaps[] = { g_Engine->CbvSrvUavHeap()->GetHeap() };
    cmd->SetDescriptorHeaps(1, heaps);
    cmd->SetComputeRootSignature(m_computeRootSignature.Get());

    // グリッド構築
    // 毎フレームグリッド情報をゼロ初期化して、古いデータによる不正な粒子カウントを防ぐ
    cmd->SetPipelineState(m_clearGridPipeline->Get());
    cmd->SetComputeRootConstantBufferView(0, m_computeParamsCB->GetAddress());
    cmd->SetComputeRootConstantBufferView(1, m_dummyViewCB->GetAddress());
    cmd->SetComputeRootDescriptorTable(2, readBuffer.srv->HandleGPU);
    cmd->SetComputeRootDescriptorTable(3, writeBuffer.uav->HandleGPU);
    cmd->SetComputeRootDescriptorTable(4, m_gpuMetaUAV->HandleGPU);
    cmd->SetComputeRootDescriptorTable(5, m_gpuGridCountUAV->HandleGPU);
    cmd->SetComputeRootDescriptorTable(6, m_gpuGridTableUAV->HandleGPU);

    UINT cellCount = m_gridDim.x * m_gridDim.y * m_gridDim.z;
    UINT64 tableCount64 = static_cast<UINT64>(cellCount) * static_cast<UINT64>(kMaxParticlesPerCell);
    UINT clearThreads = static_cast<UINT>(std::min<UINT64>(UINT_MAX, std::max<UINT64>(tableCount64, static_cast<UINT64>(cellCount))));
    UINT clearGroups = (clearThreads + 255) / 256;
    cmd->Dispatch(clearGroups, 1, 1);

    cmd->SetPipelineState(m_buildGridPipeline->Get());
    cmd->SetComputeRootConstantBufferView(0, m_computeParamsCB->GetAddress());
    cmd->SetComputeRootConstantBufferView(1, m_dummyViewCB->GetAddress());
    cmd->SetComputeRootDescriptorTable(2, readBuffer.srv->HandleGPU);
    cmd->SetComputeRootDescriptorTable(3, writeBuffer.uav->HandleGPU);
    cmd->SetComputeRootDescriptorTable(4, m_gpuMetaUAV->HandleGPU);
    cmd->SetComputeRootDescriptorTable(5, m_gpuGridCountUAV->HandleGPU);
    cmd->SetComputeRootDescriptorTable(6, m_gpuGridTableUAV->HandleGPU);

    UINT totalThreads = std::max(m_particleCount, cellCount);
    UINT groups = (totalThreads + 255) / 256;
    cmd->Dispatch(groups, 1, 1);

    // 粒子更新
    cmd->SetPipelineState(m_particlePipeline->Get());
    cmd->SetComputeRootConstantBufferView(0, m_computeParamsCB->GetAddress());
    cmd->SetComputeRootConstantBufferView(1, m_dummyViewCB->GetAddress());
    cmd->SetComputeRootDescriptorTable(2, readBuffer.srv->HandleGPU);
    cmd->SetComputeRootDescriptorTable(3, writeBuffer.uav->HandleGPU);
    cmd->SetComputeRootDescriptorTable(4, m_gpuMetaUAV->HandleGPU);
    cmd->SetComputeRootDescriptorTable(5, m_gpuGridCountUAV->HandleGPU);
    cmd->SetComputeRootDescriptorTable(6, m_gpuGridTableUAV->HandleGPU);

    UINT groupsParticle = (m_particleCount + 255) / 256;
    cmd->Dispatch(groupsParticle, 1, 1);

    // 書き込み完了後の状態遷移
    auto metaToSRV = CD3DX12_RESOURCE_BARRIER::Transition(
        m_gpuMetaBuffer.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &metaToSRV);
    auto gridCountToSRV = CD3DX12_RESOURCE_BARRIER::Transition(
        m_gpuGridCount.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &gridCountToSRV);
    auto gridTableToSRV = CD3DX12_RESOURCE_BARRIER::Transition(
        m_gpuGridTable.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmd->ResourceBarrier(1, &gridTableToSRV);

    // 新しい結果を読み戻し用にコピー
    if (writeBuffer.state != D3D12_RESOURCE_STATE_COPY_SOURCE)
    {
        auto toCopySrc = CD3DX12_RESOURCE_BARRIER::Transition(writeBuffer.resource.Get(), writeBuffer.state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmd->ResourceBarrier(1, &toCopySrc);
        writeBuffer.state = D3D12_RESOURCE_STATE_COPY_SOURCE;
    }
    cmd->CopyBufferRegion(m_gpuReadback.Get(), 0, writeBuffer.resource.Get(), 0, sizeof(GPUFluidParticle) * m_particleCount);
    if (writeBuffer.state != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
    {
        auto backToSRV = CD3DX12_RESOURCE_BARRIER::Transition(writeBuffer.resource.Get(), writeBuffer.state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        cmd->ResourceBarrier(1, &backToSRV);
        writeBuffer.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }

    m_gpuReadIndex = writeIndex;
    m_pendingReadback = true;
}

void FluidSystem::ReadbackGPUToCPU()
{
    if (!m_gpuReadback || !m_pendingReadback)
    {
        return;
    }

    // フェンスが完了していなければイベントで待機して安全に読み出す
    if (m_computeFence && m_lastSubmittedComputeFence != 0)
    {
        UINT64 completed = m_computeFence->GetCompletedValue();
        if (completed < m_lastSubmittedComputeFence && m_computeFenceEvent)
        {
            m_computeFence->SetEventOnCompletion(m_lastSubmittedComputeFence, m_computeFenceEvent);
            WaitForSingleObject(m_computeFenceEvent, INFINITE);
        }
    }

    GPUFluidParticle* mapped = nullptr;
    if (SUCCEEDED(m_gpuReadback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) && mapped)
    {
        for (UINT i = 0; i < m_particleCount; ++i)
        {
            m_cpuParticles[i].position = mapped[i].position;
            m_cpuParticles[i].velocity = mapped[i].velocity;
            m_cpuParticles[i].predicted = mapped[i].position;
        }
        m_gpuReadback->Unmap(0, nullptr);
    }

    m_pendingReadback = false;
}

void FluidSystem::UpdateParticleBuffer()
{
    if (!m_cpuMetaBuffer)
    {
        return;
    }

    ParticleMetaGPU* mapped = nullptr;
    if (FAILED(m_cpuMetaBuffer->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) || !mapped)
    {
        return;
    }

    for (UINT i = 0; i < m_particleCount; ++i)
    {
        mapped[i].position = m_cpuParticles[i].position;
        mapped[i].radius = m_material.renderRadius;
    }

    for (UINT i = m_particleCount; i < m_maxParticles; ++i)
    {
        mapped[i].position = XMFLOAT3(0.0f, 0.0f, 0.0f);
        mapped[i].radius = 0.0f;
    }

    m_cpuMetaBuffer->Unmap(0, nullptr);
    m_activeMetaSRV = m_cpuMetaSRV;
}


bool FluidSystem::CreateSSFRResources(ID3D12Device* device, DXGI_FORMAT rtvFormat)
{
    if (!device)
    {
        return false;
    }

    DestroySSFRResources();

    UINT width = std::max<UINT>(1u, g_Engine->FrameBufferWidth());
    UINT height = std::max<UINT>(1u, g_Engine->FrameBufferHeight());
    if (width == 0 || height == 0)
    {
        printf("FluidSystem: buckbuffer failed\n");
        return false;
    }

    for (UINT i = 0; i < kFrameCount; ++i)
    {
        if (!m_cameraCB[i])
        {
            m_cameraCB[i] = std::make_unique<ConstantBuffer>(sizeof(SSFRCameraConstants));
        }
    }

    if (!m_blurParamsCB)
    {
        m_blurParamsCB = std::make_unique<ConstantBuffer>(sizeof(SSFRBlurParams));
        auto params = m_blurParamsCB->GetPtr<SSFRBlurParams>();
        params->Sigma = 2.5f;
        params->DepthK = 1.0f;
        params->NormalK = 8.0f;
        params->Padding = 0.0f;
    }

    UINT64 bufferSize = sizeof(ParticleMetaGPU) * m_maxParticles;

    if (!m_cpuMetaBuffer || m_cpuMetaBuffer->GetDesc().Width < bufferSize)
    {
        m_cpuMetaBuffer.Reset();
        auto cpuHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        auto cpuDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
        HRESULT hr = device->CreateCommittedResource(&cpuHeap, D3D12_HEAP_FLAG_NONE, &cpuDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(m_cpuMetaBuffer.ReleaseAndGetAddressOf()));
        if (FAILED(hr))
        {
            printf("FluidSystem: CPUメタデータバッファの生成に失敗しました (0x%08X)\n", hr);
            return false;
        }

        if (!m_cpuMetaSRV)
        {
            m_cpuMetaSRV = g_Engine->CbvSrvUavHeap()->RegisterBuffer(m_cpuMetaBuffer.Get(), m_maxParticles, sizeof(ParticleMetaGPU));
        }
        else
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            desc.Buffer.NumElements = m_maxParticles;
            desc.Buffer.StructureByteStride = sizeof(ParticleMetaGPU);
            g_Engine->Device()->CreateShaderResourceView(m_cpuMetaBuffer.Get(), &desc, m_cpuMetaSRV->HandleCPU);
        }
    }

    auto gpuHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto gpuDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (!m_gpuMetaBuffer || m_gpuMetaBuffer->GetDesc().Width < bufferSize)
    {
        m_gpuMetaBuffer.Reset();
        HRESULT hr = device->CreateCommittedResource(&gpuHeap, D3D12_HEAP_FLAG_NONE, &gpuDesc,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
            IID_PPV_ARGS(m_gpuMetaBuffer.ReleaseAndGetAddressOf()));
        if (FAILED(hr))
        {
            printf("FluidSystem: GPUメタデータバッファの生成に失敗しました (0x%08X)\n", hr);
            return false;
        }
    }

    if (!m_gpuMetaSRV)
    {
        m_gpuMetaSRV = g_Engine->CbvSrvUavHeap()->RegisterBuffer(m_gpuMetaBuffer.Get(), m_maxParticles, sizeof(ParticleMetaGPU));
    }
    else
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc.Buffer.NumElements = m_maxParticles;
        desc.Buffer.StructureByteStride = sizeof(ParticleMetaGPU);
        g_Engine->Device()->CreateShaderResourceView(m_gpuMetaBuffer.Get(), &desc, m_gpuMetaSRV->HandleCPU);
    }

    if (!m_gpuMetaUAV)
    {
        m_gpuMetaUAV = g_Engine->CbvSrvUavHeap()->RegisterBufferUAV(m_gpuMetaBuffer.Get(), m_maxParticles, sizeof(ParticleMetaGPU));
    }
    else
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        desc.Buffer.NumElements = m_maxParticles;
        desc.Buffer.StructureByteStride = sizeof(ParticleMetaGPU);
        g_Engine->Device()->CreateUnorderedAccessView(m_gpuMetaBuffer.Get(), nullptr, &desc, m_gpuMetaUAV->HandleCPU);
    }

    m_activeMetaSRV = m_cpuMetaSRV;

    if (!m_ssfrRtvHeap)
    {
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
        heapDesc.NumDescriptors = 3;
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        HRESULT hr = device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(m_ssfrRtvHeap.ReleaseAndGetAddressOf()));
        if (FAILED(hr))
        {
            printf("FluidSystem: SSFR用RTVヒープの生成に失敗しました (0x%08X)\n", hr);
            return false;
        }
        m_ssfrRtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    }

    if (!m_ssfrCpuUavHeap)
    {
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
        heapDesc.NumDescriptors = 3;
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        HRESULT hr = device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(m_ssfrCpuUavHeap.ReleaseAndGetAddressOf()));
        if (FAILED(hr))
        {
            printf("FluidSystem: SSFR用CPU UAVヒープの生成に失敗しました (0x%08X)\n", hr);
            return false;
        }
        m_ssfrCpuUavDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        // UAVのクリアではCPU可視ディスクリプタが必須なため、ここでオフラインハンドルを確保する
        auto cpuHandle = m_ssfrCpuUavHeap->GetCPUDescriptorHandleForHeapStart();
        m_particleDepthUavCpuHandle = cpuHandle;
        cpuHandle.ptr += m_ssfrCpuUavDescriptorSize;
        m_smoothedDepthUavCpuHandle = cpuHandle;
    }

    CD3DX12_HEAP_PROPERTIES defaultHeap(D3D12_HEAP_TYPE_DEFAULT);

    auto rtvHandle = m_ssfrRtvHeap->GetCPUDescriptorHandleForHeapStart();
    m_particleDepthRTV = rtvHandle;
    rtvHandle.ptr += m_ssfrRtvDescriptorSize;
    m_smoothedDepthRTV = rtvHandle;
    rtvHandle.ptr += m_ssfrRtvDescriptorSize;
    m_thicknessRTV = rtvHandle;

    auto createTexture = [&](DXGI_FORMAT format,
        const wchar_t* name,
        std::unique_ptr<Texture2D>& texture,
        DescriptorHandle*& srvHandle,
        DescriptorHandle*& uavHandle,
        D3D12_CPU_DESCRIPTOR_HANDLE rtv,
        D3D12_CPU_DESCRIPTOR_HANDLE uavCpuHandle,
        D3D12_RESOURCE_STATES& state,
        bool createRTV)
    {
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (createRTV)
        {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        }

        CD3DX12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Tex2D(format, width, height, 1, 1);
        desc.Flags = flags;

        D3D12_CLEAR_VALUE clear{};
        clear.Format = format;
        clear.Color[0] = 0.0f;
        clear.Color[1] = 0.0f;
        clear.Color[2] = 0.0f;
        clear.Color[3] = (format == DXGI_FORMAT_R32_FLOAT) ? 1.0f : 0.0f;

        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        HRESULT hr = device->CreateCommittedResource(
            &defaultHeap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            createRTV ? &clear : nullptr,
            IID_PPV_ARGS(resource.ReleaseAndGetAddressOf()));
        if (FAILED(hr))
        {
            printf("FluidSystem: テクスチャ %ls の生成に失敗しました (0x%08X)\n", name, hr);
            return false;
        }
        resource->SetName(name);

        texture = std::unique_ptr<Texture2D>(new Texture2D(resource.Get()));

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = texture->ViewDesc();
        if (!srvHandle)
        {
            srvHandle = g_Engine->CbvSrvUavHeap()->Register(texture.get());
        }
        else
        {
            g_Engine->Device()->CreateShaderResourceView(resource.Get(), &srvDesc, srvHandle->HandleCPU);
        }
        if (!srvHandle)
        {
            printf("FluidSystem: %ls のSRV登録に失敗しました\n", name);
            return false;
        }

        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
        uavDesc.Format = format;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uavDesc.Texture2D.MipSlice = 0;
        uavDesc.Texture2D.PlaneSlice = 0;

        if (!uavHandle)
        {
            uavHandle = g_Engine->CbvSrvUavHeap()->RegisterTextureUAV(resource.Get(), format);
        }
        if (!uavHandle)
        {
            printf("FluidSystem: %ls のUAV登録に失敗しました\n", name);
            return false;
        }
        g_Engine->Device()->CreateUnorderedAccessView(resource.Get(), nullptr, &uavDesc, uavHandle->HandleCPU);

        if (uavCpuHandle.ptr != 0)
        {
            g_Engine->Device()->CreateUnorderedAccessView(resource.Get(), nullptr, &uavDesc, uavCpuHandle);
        }

        if (createRTV)
        {
            device->CreateRenderTargetView(resource.Get(), nullptr, rtv);
        }

        state = D3D12_RESOURCE_STATE_COMMON;
        return true;
    };

    if (!createTexture(DXGI_FORMAT_R32_FLOAT, L"FluidParticleDepth", m_particleDepthTexture, m_particleDepthSRV, m_particleDepthUAV, m_particleDepthRTV, m_particleDepthUavCpuHandle, m_particleDepthState, true))
    {
        return false;
    }

    if (!createTexture(DXGI_FORMAT_R32_FLOAT, L"FluidSmoothedDepth", m_smoothedDepthTexture, m_smoothedDepthSRV, m_smoothedDepthUAV, m_smoothedDepthRTV, m_smoothedDepthUavCpuHandle, m_smoothedDepthState, true))
    {
        return false;
    }

    if (!createTexture(DXGI_FORMAT_R8G8B8A8_UNORM, L"FluidNormal", m_normalTexture, m_normalSRV, m_normalUAV, {}, {}, m_normalState, false))
    {
        return false;
    }

    if (!createTexture(DXGI_FORMAT_R16_FLOAT, L"FluidThickness", m_thicknessTexture, m_thicknessSRV, m_thicknessUAV, m_thicknessRTV, {}, m_thicknessState, true))
    {
        return false;
    }

    // シーンカラーを参照するためのコピー先テクスチャとSRVを用意
    {
        bool needCreate = !m_sceneColorCopy;
        if (m_sceneColorCopy)
        {
            auto desc = m_sceneColorCopy->GetDesc();
            needCreate = desc.Width != width || desc.Height != height;
        }

        if (needCreate)
        {
            m_sceneColorCopy.Reset();
            auto colorDesc = CD3DX12_RESOURCE_DESC::Tex2D(rtvFormat, width, height, 1, 1);
            HRESULT hr = device->CreateCommittedResource(
                &gpuHeap,
                D3D12_HEAP_FLAG_NONE,
                &colorDesc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                IID_PPV_ARGS(m_sceneColorCopy.ReleaseAndGetAddressOf()));
            if (FAILED(hr))
            {
                printf("FluidSystem: シーンカラーコピー用テクスチャの生成に失敗しました (0x%08X)\n", hr);
                return false;
            }
            m_sceneColorCopyState = D3D12_RESOURCE_STATE_COPY_DEST;
        }

        D3D12_SHADER_RESOURCE_VIEW_DESC sceneColorDesc{};
        sceneColorDesc.Format = rtvFormat;
        sceneColorDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sceneColorDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        sceneColorDesc.Texture2D.MipLevels = 1;
        if (!m_sceneColorSRV)
        {
            m_sceneColorSRV = g_Engine->CbvSrvUavHeap()->Register(m_sceneColorCopy.Get(), sceneColorDesc);
        }
        else
        {
            g_Engine->Device()->CreateShaderResourceView(m_sceneColorCopy.Get(), &sceneColorDesc, m_sceneColorSRV->HandleCPU);
        }
        if (!m_sceneColorSRV)
        {
            printf("FluidSystem: シーンカラーSRVの登録に失敗しました\n");
            return false;
        }
    }

    // 深度バッファを参照するためのSRVを作成
    if (ID3D12Resource* depthBuffer = g_Engine->DepthStencilBuffer())
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC depthDesc{};
        depthDesc.Format = DXGI_FORMAT_R32_FLOAT;
        depthDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        depthDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        depthDesc.Texture2D.MipLevels = 1;
        if (!m_sceneDepthSRV)
        {
            m_sceneDepthSRV = g_Engine->CbvSrvUavHeap()->Register(depthBuffer, depthDesc);
        }
        else
        {
            g_Engine->Device()->CreateShaderResourceView(depthBuffer, &depthDesc, m_sceneDepthSRV->HandleCPU);
        }
        if (!m_sceneDepthSRV)
        {
            printf("FluidSystem: シーン深度SRVの登録に失敗しました\n");
            return false;
        }
        m_sceneDepthState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    }

    // 粒子描画ルートシグネチャ
    {
        CD3DX12_DESCRIPTOR_RANGE srvRange;
        // 頂点シェーダーで StructuredBuffer<ParticleData> (t0) を読むための SRV テーブル
        srvRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);

        CD3DX12_DESCRIPTOR_RANGE uavRanges[3];
        // ピクセルシェーダーの RTV0 と競合しないよう、UAV はレジスタ u1 以降へ配置
        uavRanges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1);
        uavRanges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2);
        uavRanges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 3);

        CD3DX12_ROOT_PARAMETER params[5];
        params[0].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_ALL);
        params[1].InitAsDescriptorTable(1, &srvRange, D3D12_SHADER_VISIBILITY_VERTEX);
        params[2].InitAsDescriptorTable(1, &uavRanges[0], D3D12_SHADER_VISIBILITY_PIXEL);
        params[3].InitAsDescriptorTable(1, &uavRanges[1], D3D12_SHADER_VISIBILITY_PIXEL);
        params[4].InitAsDescriptorTable(1, &uavRanges[2], D3D12_SHADER_VISIBILITY_PIXEL);

        CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params, 0, nullptr,
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS);

        Microsoft::WRL::ComPtr<ID3DBlob> blob, error;
        HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error);
        if (FAILED(hr))
        {
            if (error)
            {
                printf("FluidSystem: 粒子用ルートシグネチャ生成失敗 -> %s", static_cast<const char*>(error->GetBufferPointer()));
            }
            return false;
        }
        hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(m_particleRootSignature.ReleaseAndGetAddressOf()));
        if (FAILED(hr))
        {
            printf("FluidSystem: 粒子用ルートシグネチャ作成に失敗しました (0x%08X)", hr);
            return false;
        }

        Microsoft::WRL::ComPtr<ID3DBlob> vs, ps;
        if (!LoadOrCompileShader(L"SSFRParticleVS.hlsl", "main", "vs_5_0", vs))
        {
            return false;
        }
        if (!LoadOrCompileShader(L"SSFRParticlePS.hlsl", "main", "ps_5_0", ps))
        {
            return false;
        }

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = m_particleRootSignature.Get();
        pso.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
        pso.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
        pso.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        pso.SampleMask = UINT_MAX;
        pso.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        pso.DepthStencilState.DepthEnable = FALSE;
        pso.DepthStencilState.StencilEnable = FALSE;
        pso.InputLayout = { nullptr, 0 };
        pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso.NumRenderTargets = 1;
        pso.RTVFormats[0] = rtvFormat;
        // 深度を使用しないPSOなので、DSV形式はUNKNOWNにして空のDSVバインドを許可する
        pso.DSVFormat = DXGI_FORMAT_UNKNOWN;
        pso.SampleDesc.Count = 1;
        pso.SampleDesc.Quality = 0;

        hr = device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(m_particlePipelineState.ReleaseAndGetAddressOf()));
        if (FAILED(hr))
        {
            printf("FluidSystem: 粒子描画PSOの生成に失敗しました (0x%08X)", hr);
            return false;
        }
    }

    // バイラテラルブラー
    {
        CD3DX12_DESCRIPTOR_RANGE srvRanges[2];
        srvRanges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
        srvRanges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);
        CD3DX12_DESCRIPTOR_RANGE uavRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
        CD3DX12_ROOT_PARAMETER params[5];
        params[0].InitAsDescriptorTable(1, &srvRanges[0]);
        params[1].InitAsDescriptorTable(1, &srvRanges[1]);
        params[2].InitAsDescriptorTable(1, &uavRange);
        params[3].InitAsConstantBufferView(0);
        params[4].InitAsConstantBufferView(1);

        CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params, 0, nullptr,
            D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS);

        Microsoft::WRL::ComPtr<ID3DBlob> blob, error;
        HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error);
        if (FAILED(hr))
        {
            if (error)
            {
                printf("FluidSystem: ブラー用ルートシグネチャ生成失敗 -> %s", static_cast<const char*>(error->GetBufferPointer()));
            }
            return false;
        }
        hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(m_blurRootSignature.ReleaseAndGetAddressOf()));
        if (FAILED(hr))
        {
            printf("FluidSystem: ブラー用ルートシグネチャ作成に失敗しました (0x%08X)", hr);
            return false;
        }

        Microsoft::WRL::ComPtr<ID3DBlob> cs;
        if (!LoadOrCompileShader(L"SSFRBilateralCS.hlsl", "main", "cs_5_0", cs))
        {
            return false;
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = m_blurRootSignature.Get();
        pso.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
        HRESULT hr2 = device->CreateComputePipelineState(&pso, IID_PPV_ARGS(m_blurPipelineState.ReleaseAndGetAddressOf()));
        if (FAILED(hr2))
        {
            printf("FluidSystem: ブラー用PSO作成に失敗しました (0x%08X)", hr2);
            return false;
        }
    }

    // 法線生成
    {
        CD3DX12_DESCRIPTOR_RANGE srvRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
        CD3DX12_DESCRIPTOR_RANGE uavRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
        CD3DX12_ROOT_PARAMETER params[3];
        params[0].InitAsDescriptorTable(1, &srvRange);
        params[1].InitAsDescriptorTable(1, &uavRange);
        params[2].InitAsConstantBufferView(0);

        CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params, 0, nullptr,
            D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS);

        Microsoft::WRL::ComPtr<ID3DBlob> blob, error;
        HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error);
        if (FAILED(hr))
        {
            if (error)
            {
                printf("FluidSystem: 法線CSルートシグネチャ生成失敗 -> %s", static_cast<const char*>(error->GetBufferPointer()));
            }
            return false;
        }
        hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(m_normalRootSignature.ReleaseAndGetAddressOf()));
        if (FAILED(hr))
        {
            printf("FluidSystem: 法線CSルートシグネチャ作成に失敗しました (0x%08X)", hr);
            return false;
        }

        Microsoft::WRL::ComPtr<ID3DBlob> cs;
        if (!LoadOrCompileShader(L"SSFRNormalCS.hlsl", "main", "cs_5_0", cs))
        {
            return false;
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = m_normalRootSignature.Get();
        pso.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
        HRESULT hr2 = device->CreateComputePipelineState(&pso, IID_PPV_ARGS(m_normalPipelineState.ReleaseAndGetAddressOf()));
        if (FAILED(hr2))
        {
            printf("FluidSystem: 法線生成PSO作成に失敗しました (0x%08X)", hr2);
            return false;
        }
    }

    // 合成
    {
        CD3DX12_DESCRIPTOR_RANGE srvRanges[5];
        srvRanges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
        srvRanges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);
        srvRanges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);
        srvRanges[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 3);
        srvRanges[4].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 4);
        CD3DX12_ROOT_PARAMETER params[6];
        params[0].InitAsDescriptorTable(1, &srvRanges[0], D3D12_SHADER_VISIBILITY_PIXEL);
        params[1].InitAsDescriptorTable(1, &srvRanges[1], D3D12_SHADER_VISIBILITY_PIXEL);
        params[2].InitAsDescriptorTable(1, &srvRanges[2], D3D12_SHADER_VISIBILITY_PIXEL);
        params[3].InitAsDescriptorTable(1, &srvRanges[3], D3D12_SHADER_VISIBILITY_PIXEL);
        params[4].InitAsDescriptorTable(1, &srvRanges[4], D3D12_SHADER_VISIBILITY_PIXEL);
        params[5].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_PIXEL);

        CD3DX12_STATIC_SAMPLER_DESC sampler(0);
        sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;

        CD3DX12_ROOT_SIGNATURE_DESC desc(_countof(params), params, 1, &sampler,
            D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS);

        Microsoft::WRL::ComPtr<ID3DBlob> blob, error;
        HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error);
        if (FAILED(hr))
        {
            if (error)
            {
                printf("FluidSystem: 合成ルートシグネチャ生成失敗 -> %s", static_cast<const char*>(error->GetBufferPointer()));
            }
            return false;
        }
        hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(m_compositeRootSignature.ReleaseAndGetAddressOf()));
        if (FAILED(hr))
        {
            printf("FluidSystem: 合成ルートシグネチャ作成に失敗しました (0x%08X)", hr);
            return false;
        }

        Microsoft::WRL::ComPtr<ID3DBlob> vs, ps;
        if (!LoadOrCompileShader(L"FullscreenVS.hlsl", "main", "vs_5_0", vs))
        {
            return false;
        }
        if (!LoadOrCompileShader(L"SSFRCompositePS.hlsl", "main", "ps_5_0", ps))
        {
            return false;
        }

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = m_compositeRootSignature.Get();
        pso.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
        pso.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
        pso.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        pso.SampleMask = UINT_MAX;
        pso.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        pso.DepthStencilState.DepthEnable = FALSE;
        pso.DepthStencilState.StencilEnable = FALSE;
        pso.InputLayout = { nullptr, 0 };
        pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso.NumRenderTargets = 1;
        pso.RTVFormats[0] = rtvFormat;
        pso.SampleDesc.Count = 1;
        pso.SampleDesc.Quality = 0;

        HRESULT hr2 = device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(m_compositePipelineState.ReleaseAndGetAddressOf()));
        if (FAILED(hr2))
        {
            printf("FluidSystem: 合成PSOの生成に失敗しました (0x%08X)", hr2);
            return false;
        }
    }

    return true;
}

void FluidSystem::DestroySSFRResources()
{
    m_particleDepthTexture.reset();
    m_smoothedDepthTexture.reset();
    m_normalTexture.reset();
    m_thicknessTexture.reset();
    m_particleDepthState = D3D12_RESOURCE_STATE_COMMON;
    m_smoothedDepthState = D3D12_RESOURCE_STATE_COMMON;
    m_normalState = D3D12_RESOURCE_STATE_COMMON;
    m_thicknessState = D3D12_RESOURCE_STATE_COMMON;
    m_sceneColorCopy.Reset();
    m_sceneColorCopyState = D3D12_RESOURCE_STATE_COMMON;
    m_sceneDepthState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    m_sceneColorSRV = nullptr;
    m_sceneDepthSRV = nullptr;
    m_ssfrCpuUavHeap.Reset();
    m_ssfrCpuUavDescriptorSize = 0;
    m_particleDepthUavCpuHandle = {};
    m_smoothedDepthUavCpuHandle = {};

    m_particleRootSignature.Reset();
    m_particlePipelineState.Reset();
    m_blurRootSignature.Reset();
    m_blurPipelineState.Reset();
    m_normalRootSignature.Reset();
    m_normalPipelineState.Reset();
    m_compositeRootSignature.Reset();
    m_compositePipelineState.Reset();
}

void FluidSystem::CreateGPUResources(ID3D12Device* device)
{
    if (!device)
    {
        return;
    }

    UpdateGridSettings();

    UINT particleStride = sizeof(GPUFluidParticle);
    UINT metaStride = sizeof(ParticleMetaGPU);
    UINT particleBufferSize = particleStride * m_maxParticles;
    UINT metaBufferSize = metaStride * m_maxParticles;
    UINT cellCount = std::max<UINT>(1u, m_gridDim.x * m_gridDim.y * m_gridDim.z);

    auto defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    for (int i = 0; i < 2; ++i)
    {
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(particleBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        HRESULT hr = device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
            IID_PPV_ARGS(m_gpuParticleBuffers[i].resource.ReleaseAndGetAddressOf()));
        if (FAILED(hr))
        {
            printf("FluidSystem 6 : GPU粒子バッファ生成に失敗しました (%d)\n", i);
            m_gpuAvailable = false;
            return;
        }
        m_gpuParticleBuffers[i].state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        if (!m_gpuParticleBuffers[i].srv)
        {
            m_gpuParticleBuffers[i].srv = g_Engine->CbvSrvUavHeap()->RegisterBuffer(m_gpuParticleBuffers[i].resource.Get(), m_maxParticles, particleStride);
        }
        else
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC descSrv{};
            descSrv.Format = DXGI_FORMAT_UNKNOWN;
            descSrv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            descSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            descSrv.Buffer.NumElements = m_maxParticles;
            descSrv.Buffer.StructureByteStride = particleStride;
            g_Engine->Device()->CreateShaderResourceView(m_gpuParticleBuffers[i].resource.Get(), &descSrv, m_gpuParticleBuffers[i].srv->HandleCPU);
        }
        if (!m_gpuParticleBuffers[i].srv)
        {
            printf("FluidSystem 7 : GPU粒子SRVの登録に失敗しました (%d)\n", i);
            m_gpuAvailable = false;
            return;
        }

        if (!m_gpuParticleBuffers[i].uav)
        {
            m_gpuParticleBuffers[i].uav = g_Engine->CbvSrvUavHeap()->RegisterBufferUAV(m_gpuParticleBuffers[i].resource.Get(), m_maxParticles, particleStride);
        }
        else
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC descUAV{};
            descUAV.Format = DXGI_FORMAT_UNKNOWN;
            descUAV.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            descUAV.Buffer.NumElements = m_maxParticles;
            descUAV.Buffer.StructureByteStride = particleStride;
            g_Engine->Device()->CreateUnorderedAccessView(m_gpuParticleBuffers[i].resource.Get(), nullptr, &descUAV, m_gpuParticleBuffers[i].uav->HandleCPU);
        }
        if (!m_gpuParticleBuffers[i].uav)
        {
            printf("FluidSystem 8 : GPU粒子UAVの登録に失敗しました (%d)\n", i);
            m_gpuAvailable = false;
            return;
        }
    }

    // グリッドバッファ
    auto gridDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(UINT) * cellCount, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    HRESULT hrGrid = device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &gridDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
        IID_PPV_ARGS(m_gpuGridCount.ReleaseAndGetAddressOf()));
    if (FAILED(hrGrid))
    {
        printf("FluidSystem 9 : グリッドカウントバッファ生成に失敗しました\n");
        m_gpuAvailable = false;
        return;
    }
    if (!m_gpuGridCountSRV)
    {
        m_gpuGridCountSRV = g_Engine->CbvSrvUavHeap()->RegisterBuffer(m_gpuGridCount.Get(), cellCount, sizeof(UINT));
    }
    else
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc.Buffer.NumElements = cellCount;
        desc.Buffer.StructureByteStride = sizeof(UINT);
        g_Engine->Device()->CreateShaderResourceView(m_gpuGridCount.Get(), &desc, m_gpuGridCountSRV->HandleCPU);
    }
    if (!m_gpuGridCountSRV)
    {
        printf("FluidSystem 10a : グリッドカウントSRVの登録に失敗しました\n");
        m_gpuAvailable = false;
        return;
    }

    if (!m_gpuGridCountUAV)
    {
        m_gpuGridCountUAV = g_Engine->CbvSrvUavHeap()->RegisterBufferUAV(m_gpuGridCount.Get(), cellCount, sizeof(UINT));
    }
    else
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        desc.Buffer.NumElements = cellCount;
        desc.Buffer.StructureByteStride = sizeof(UINT);
        g_Engine->Device()->CreateUnorderedAccessView(m_gpuGridCount.Get(), nullptr, &desc, m_gpuGridCountUAV->HandleCPU);
    }
    if (!m_gpuGridCountUAV)
    {
        printf("FluidSystem 10 : グリッドカウントUAVの登録に失敗しました\n");
        m_gpuAvailable = false;
        return;
    }

    auto tableDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(UINT) * cellCount * kMaxParticlesPerCell, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    HRESULT hrTable = device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &tableDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
        IID_PPV_ARGS(m_gpuGridTable.ReleaseAndGetAddressOf()));
    if (FAILED(hrTable))
    {
        printf("FluidSystem 11 : グリッドテーブル生成に失敗しました\n");
        m_gpuAvailable = false;
        return;
    }
    if (!m_gpuGridTableSRV)
    {
        m_gpuGridTableSRV = g_Engine->CbvSrvUavHeap()->RegisterBuffer(m_gpuGridTable.Get(), cellCount * kMaxParticlesPerCell, sizeof(UINT));
    }
    else
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc.Buffer.NumElements = cellCount * kMaxParticlesPerCell;
        desc.Buffer.StructureByteStride = sizeof(UINT);
        g_Engine->Device()->CreateShaderResourceView(m_gpuGridTable.Get(), &desc, m_gpuGridTableSRV->HandleCPU);
    }
    if (!m_gpuGridTableSRV)
    {
        printf("FluidSystem 11a : グリッドテーブルSRVの登録に失敗しました\n");
        m_gpuAvailable = false;
        return;
    }

    if (!m_gpuGridTableUAV)
    {
        m_gpuGridTableUAV = g_Engine->CbvSrvUavHeap()->RegisterBufferUAV(m_gpuGridTable.Get(), cellCount * kMaxParticlesPerCell, sizeof(UINT));
    }
    else
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        desc.Buffer.NumElements = cellCount * kMaxParticlesPerCell;
        desc.Buffer.StructureByteStride = sizeof(UINT);
        g_Engine->Device()->CreateUnorderedAccessView(m_gpuGridTable.Get(), nullptr, &desc, m_gpuGridTableUAV->HandleCPU);
    }
    if (!m_gpuGridTableUAV)
    {
        printf("FluidSystem 12 : グリッドテーブルUAVの登録に失敗しました\n");
        m_gpuAvailable = false;
        return;
    }

    // アップロード・リードバック
    auto uploadHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto uploadDesc = CD3DX12_RESOURCE_DESC::Buffer(particleBufferSize);
    HRESULT hrUpload = device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &uploadDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(m_gpuUpload.ReleaseAndGetAddressOf()));
    if (FAILED(hrUpload))
    {
        printf("FluidSystem 13 : GPUアップロードバッファ生成に失敗しました\n");
        m_gpuAvailable = false;
        return;
    }

    auto readbackHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    auto readbackDesc = CD3DX12_RESOURCE_DESC::Buffer(particleBufferSize);
    HRESULT hrReadback = device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE, &readbackDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(m_gpuReadback.ReleaseAndGetAddressOf()));
    if (FAILED(hrReadback))
    {
        printf("FluidSystem 14 : GPUリードバックバッファ生成に失敗しました\n");
        m_gpuAvailable = false;
        return;
    }

    // コンピュート用のルートシグネチャとPSOを構築する。
    // それぞれのリソース種別に応じてテーブル／ルートディスクリプタを定義し、
    // RootSignature の不整合による CreateComputePipelineState 失敗を避けるため、
    // RegisterSpace や DescriptorRangeFlag を明示しておく。
    CD3DX12_DESCRIPTOR_RANGE1 srvRange;
    srvRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC);

    CD3DX12_DESCRIPTOR_RANGE1 uavRanges[4];
    uavRanges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC);
    uavRanges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC);
    uavRanges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC);
    uavRanges[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 3, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC);

    CD3DX12_ROOT_PARAMETER1 params[7] = {};
    params[0].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC);
    params[1].InitAsConstantBufferView(1, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC);
    params[2].InitAsDescriptorTable(1, &srvRange);
    params[3].InitAsDescriptorTable(1, &uavRanges[0]);
    params[4].InitAsDescriptorTable(1, &uavRanges[1]);
    params[5].InitAsDescriptorTable(1, &uavRanges[2]);
    params[6].InitAsDescriptorTable(1, &uavRanges[3]);

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootDesc;
    rootDesc.Init_1_1(_countof(params), params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> serialized;
    ComPtr<ID3DBlob> errors;
    HRESULT hrSerialize = D3D12SerializeVersionedRootSignature(&rootDesc, serialized.GetAddressOf(), errors.GetAddressOf());
    if (FAILED(hrSerialize))
    {
        if (errors)
        {
            printf("Compute root signature error: %s\n", static_cast<const char*>(errors->GetBufferPointer()));
        }
        return;
    }
    HRESULT hrRoot = device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(m_computeRootSignature.ReleaseAndGetAddressOf()));
    if (FAILED(hrRoot))
    {
        printf("FluidSystem 15 : コンピュート用ルートシグネチャ生成に失敗しました\n");
        m_gpuAvailable = false;
        return;
    }

    m_clearGridPipeline = std::make_unique<ComputePipelineState>();
    m_clearGridPipeline->SetDevice(device);
    m_clearGridPipeline->SetRootSignature(m_computeRootSignature.Get());
    m_clearGridPipeline->SetCS(L"ClearGridCS.cso");
    if (!m_clearGridPipeline->Create())
    {
        m_clearGridPipeline.reset();
        m_buildGridPipeline.reset();
        m_particlePipeline.reset();
        m_gpuAvailable = false;
        return;
    }

    m_buildGridPipeline = std::make_unique<ComputePipelineState>();
    m_buildGridPipeline->SetDevice(device);
    m_buildGridPipeline->SetRootSignature(m_computeRootSignature.Get());
    m_buildGridPipeline->SetCS(L"BuildGridCS.cso");
    if (!m_buildGridPipeline->Create())
    {
        m_buildGridPipeline.reset();
        m_particlePipeline.reset();
        m_clearGridPipeline.reset();
        m_gpuAvailable = false;
        return;
    }

    m_particlePipeline = std::make_unique<ComputePipelineState>();
    m_particlePipeline->SetDevice(device);
    m_particlePipeline->SetRootSignature(m_computeRootSignature.Get());
    m_particlePipeline->SetCS(L"ParticleCS.cso");
    if (!m_particlePipeline->Create())
    {
        m_particlePipeline.reset();
        m_buildGridPipeline.reset();
        m_clearGridPipeline.reset();
        m_gpuAvailable = false;
        return;
    }

    // コンピュート用のコマンドアロケーター／リスト／フェンスを準備
    if (!m_computeAllocator)
    {
        HRESULT hrAlloc = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(m_computeAllocator.ReleaseAndGetAddressOf()));
        if (FAILED(hrAlloc))
        {
            printf("FluidSystem 16 : コンピュート用アロケーター生成に失敗しました\n");
            m_gpuAvailable = false;
            return;
        }
    }

    if (!m_computeCommandList)
    {
        HRESULT hrList = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_computeAllocator.Get(), nullptr, IID_PPV_ARGS(m_computeCommandList.ReleaseAndGetAddressOf()));
        if (FAILED(hrList))
        {
            printf("FluidSystem 17 : コンピュート用コマンドリスト生成に失敗しました\n");
            m_gpuAvailable = false;
            return;
        }
        // 生成直後は開いているので一度閉じておく
        m_computeCommandList->Close();
    }

    if (!m_computeFence)
    {
        HRESULT hrFence = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(m_computeFence.ReleaseAndGetAddressOf()));
        if (FAILED(hrFence))
        {
            printf("FluidSystem 18 : コンピュート用フェンス生成に失敗しました\n");
            m_gpuAvailable = false;
            return;
        }
        m_computeFenceValue = 0;
        m_lastSubmittedComputeFence = 0;
    }

    if (!m_computeFenceEvent)
    {
        m_computeFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (!m_computeFenceEvent)
        {
            printf("FluidSystem 19 : フェンスイベント生成に失敗しました\n");
            m_gpuAvailable = false;
            return;
        }
    }

    m_gpuReadIndex = 0;
    m_pendingReadback = false;
    m_gpuDirty = true;

    // 必須リソースが揃ったか最終チェックし、足りない場合はGPUモードを無効化する
    m_gpuAvailable = HasValidGPUResources();
    if (!m_gpuAvailable)
    {
        printf("FluidSystem ERROR: GPUリソースの作成が不完全なためGPUシミュレーションを無効化しました\n");
        m_useGPU = false;
    }
}

// FluidSystem.cpp: BeginComputeCommandList
ID3D12GraphicsCommandList* FluidSystem::BeginComputeCommandList()
{
    if (!m_computeAllocator || !m_computeCommandList || !m_computeFence || !m_computeFenceEvent)
    {
        return nullptr;
    }

    // 直前に発行したコンピュートコマンドの実行が終わるまで待機する
    if (m_computeFence->GetCompletedValue() < m_lastSubmittedComputeFence)
    {
        // GPUが完了したときにイベントが発火するように設定し、待機する
        HRESULT hr = m_computeFence->SetEventOnCompletion(m_lastSubmittedComputeFence, m_computeFenceEvent);
        if (FAILED(hr))
        {
            printf("FluidSystem ERROR: Failed to set event on completion. Disabling GPU simulation.\n");
            m_gpuAvailable = false;
            return nullptr;
        }
        WaitForSingleObject(m_computeFenceEvent, INFINITE);
    }

    // GPUの処理が完了したので、アロケーターとコマンドリストを安全にリセットできる
    HRESULT hrAlloc = m_computeAllocator->Reset();
    if (FAILED(hrAlloc))
    {
        // HRESULT をログに出力して詳細な原因を追跡しやすくする
        printf("FluidSystem 20 : Compute allocator reset failed (HRESULT: 0x%08X).\n", hrAlloc);
        if (hrAlloc == DXGI_ERROR_DEVICE_REMOVED)
        {
            HRESULT reason = m_device->GetDeviceRemovedReason();
            printf("    Reason: Device Removed (0x%08X)\n", reason);
        }
        // 回復不能なエラーとみなし、GPUシミュレーションを無効化してCPUにフォールバックさせる
        printf("    Disabling GPU simulation due to an unrecoverable error.\n");
        m_gpuAvailable = false;
        return nullptr;
    }

    HRESULT hrCmd = m_computeCommandList->Reset(m_computeAllocator.Get(), nullptr);
    if (FAILED(hrCmd))
    {
        // HRESULT をログに出力
        printf("FluidSystem 21 : Compute Commandlist Reset failed (HRESULT: 0x%08X).\n", hrCmd);
        if (hrCmd == DXGI_ERROR_DEVICE_REMOVED)
        {
            HRESULT reason = m_device->GetDeviceRemovedReason();
            printf("    Reason: Device Removed (0x%08X)\n", reason);
        }
        // こちらも同様に、エラー発生時はGPUシミュレーションを無効化
        printf("    Disabling GPU simulation due to an unrecoverable error.\n");
        m_gpuAvailable = false;
        return nullptr;
    }

    return m_computeCommandList.Get();
}

// FluidSystem.cpp: SubmitComputeCommandList (修正後)
void FluidSystem::SubmitComputeCommandList()
{
    if (!m_computeCommandList)
    {
        return;
    }

    HRESULT hrClose = m_computeCommandList->Close();
    if (FAILED(hrClose))
    {
        printf("FluidSystem 22 : compute command list close failed\n");
        m_gpuAvailable = false;
        m_useGPU = false;
        return;
    }

    ID3D12CommandList* lists[] = { m_computeCommandList.Get() };
    ID3D12CommandQueue* queue = g_Engine->ComputeCommandQueue();
    if (!queue)
    {
        // コンピュートキューが無ければ描画キューで代用
        queue = g_Engine->CommandQueue();
    }

    if (!queue)
    {
        return;
    }

    queue->ExecuteCommandLists(1, lists);

    if (m_computeFence)
    {
        // 待つべき目標値を先に更新する
        m_computeFenceValue++;
        m_lastSubmittedComputeFence = m_computeFenceValue;

        if (FAILED(queue->Signal(m_computeFence.Get(), m_lastSubmittedComputeFence)))
        {
            // Signalの失敗は致命的なので、GPUシミュレーションを無効にする
            printf("FluidSystem ERROR: Failed to signal the compute fence. Disabling GPU simulation.\n");
            m_gpuAvailable = false;
            return; // これ以上続行しない
        }

        // グラフィックスキューはコンピュート結果を待ってから描画を継続
        if (ID3D12CommandQueue* graphicsQueue = g_Engine->CommandQueue())
        {
            if (graphicsQueue != queue)
            {
                graphicsQueue->Wait(m_computeFence.Get(), m_lastSubmittedComputeFence);
            }
        }
    }
}

bool FluidSystem::HasValidGPUResources() const
{
    // ディスクリプタハンドルが有効かを確認するラムダ
    auto isValidHandle = [](const DescriptorHandle* handle)
    {
        return handle && handle->HandleCPU.ptr != 0 && handle->HandleGPU.ptr != 0;
    };

    if (!m_gpuMetaBuffer || !isValidHandle(m_gpuMetaSRV) || !isValidHandle(m_gpuMetaUAV))
    {
        return false;
    }

    for (const auto& buffer : m_gpuParticleBuffers)
    {
        if (!buffer.resource || !isValidHandle(buffer.srv) || !isValidHandle(buffer.uav))
        {
            return false;
        }
    }

    if (!m_gpuGridCount || !isValidHandle(m_gpuGridCountSRV) || !isValidHandle(m_gpuGridCountUAV))
    {
        return false;
    }

    if (!m_gpuGridTable || !isValidHandle(m_gpuGridTableSRV) || !isValidHandle(m_gpuGridTableUAV))
    {
        return false;
    }

    if (!m_gpuUpload || !m_gpuReadback)
    {
        return false;
    }

    if (!m_buildGridPipeline || !m_particlePipeline || !m_clearGridPipeline || !m_computeRootSignature)
    {
        return false;
    }

    return true;
}

void FluidSystem::UpdateGridSettings()
{
    float cellSize = std::max(0.02f, m_material.smoothingRadius);
    m_spatialGrid.SetCellSize(cellSize);

    float width = m_boundsMax.x - m_boundsMin.x;
    float height = m_boundsMax.y - m_boundsMin.y;
    float depth = m_boundsMax.z - m_boundsMin.z;

    m_gridDim.x = std::max<UINT>(1u, static_cast<UINT>(std::ceil(width / cellSize)));
    m_gridDim.y = std::max<UINT>(1u, static_cast<UINT>(std::ceil(height / cellSize)));
    m_gridDim.z = std::max<UINT>(1u, static_cast<UINT>(std::ceil(depth / cellSize)));
}
