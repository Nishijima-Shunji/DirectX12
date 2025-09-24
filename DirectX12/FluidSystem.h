#pragma once
#include "ComPtr.h"
#include <d3d12.h>
#include <wrl.h>
#include <DirectXMath.h>
#include "ConstantBuffer.h"
#include "RootSignature.h"
#include "PipelineState.h"
#include "VertexBuffer.h"
#include "SharedStruct.h"
#include <array>
#include <memory>
#include <vector>

struct FluidParticle {
    DirectX::XMFLOAT3 position; // 現在位置
    DirectX::XMFLOAT3 velocity; // 速度
    DirectX::XMFLOAT3 x_pred;   // 予測位置
    float             lambda = 0.0f; // PBF用ラグランジュ乗数
    float             density = 0.0f; // 推定密度
};

class FluidSystem {
private:
    static constexpr UINT kFrameCount = 2; // ダブルバッファ分のCB

    std::vector<FluidParticle>          m_cpuParticles;   // CPU上の粒子データ
    std::vector<ParticleVertex>         m_cpuVertices;    // GPU転送用の頂点配列
    std::unique_ptr<RootSignature>      m_rootSignature;  // 描画用ルートシグネチャ
    std::unique_ptr<PipelineState>      m_pipelineState;  // 描画用PSO
    std::unique_ptr<VertexBuffer>       m_vertexBuffer;   // 粒子座標用VB
    std::array<std::unique_ptr<ConstantBuffer>, kFrameCount> m_transformCB; // TransformCB

    UINT        m_maxParticles = 0; // 粒子数
    float       m_timeStep = 1.0f / 60.0f; // 1フレームの時間
    bool        m_initialized = false;     // 初期化済みかどうか

    // PBFパラメータ（CPU版の簡易実装）
    float                      m_restDensity = 1000.0f;   // 基準密度
    float                      m_particleMass = 1.0f;     // 粒子質量
    float                      m_smoothingRadius = 0.12f; // カーネル半径
    float                      m_epsilon = 100.0f;        // 安定化項
    int                        m_solverIterations = 3;    // ソルバ反復回数
    DirectX::XMFLOAT3          m_gravity = { 0.0f, -9.8f, 0.0f }; // 重力

    void StepCPU(float dt);             // CPUでPBF計算を行う
    void UpdateVertexBuffer();          // GPUへ頂点データを転送

public:
    void Init(ID3D12Device* device, DXGI_FORMAT rtvFormat, UINT maxParticles, UINT threadGroupCount);
    void UseGPU(bool enable) { /* GPU実装は未対応なのでダミー */ }

    void Simulate(ID3D12GraphicsCommandList* cmd, float dt); // PBF更新
    void Render(ID3D12GraphicsCommandList* cmd,
        const DirectX::XMFLOAT4X4& invViewProj,
        const DirectX::XMFLOAT3& camPos,
        float isoLevel);

    void StartDrag(int, int, class Camera*) {}
    void Drag(int, int, class Camera*) {}
    void EndDrag() {}
};
