#pragma once
#include "ComPtr.h"
#include <d3d12.h>
#include <wrl.h>
#include <DirectXMath.h>
#include "ConstantBuffer.h"
#include "DescriptorHeap.h"
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

    struct MetaConstants
    {
        DirectX::XMFLOAT4X4 InvViewProj; // ビュー射影逆行列
        DirectX::XMFLOAT4   CamRadius;   // xyz: カメラ座標 / w: 粒子半径
        DirectX::XMFLOAT4   IsoCount;    // x: 等値面しきい値 / y: 粒子数 / z: レイステップ係数 / w: 予約
    };

    std::vector<FluidParticle>          m_cpuParticles;   // CPU上の粒子データ
    ComPtr<ID3D12RootSignature>         m_metaRootSignature; // メタボール描画用ルートシグネチャ
    ComPtr<ID3D12PipelineState>         m_metaPipelineState; // メタボール描画用PSO
    std::array<std::unique_ptr<ConstantBuffer>, kFrameCount> m_metaCB; // メタボール用定数バッファ
    ComPtr<ID3D12Resource>              m_particleMetaBuffer; // 粒子メタデータ（位置＋半径）
    DescriptorHandle*                   m_particleSRV = nullptr; // 粒子メタデータ用SRVハンドル
    UINT                                m_particleCount = 0; // 実際に描画する粒子数

    UINT        m_maxParticles = 0; // 粒子数
    float       m_timeStep = 1.0f / 60.0f; // 1フレームの時間
    bool        m_initialized = false;     // 初期化済みかどうか

    float       m_renderRadius = 0.10f; // メタボール半径（描画用）
    float       m_rayStepScale = 0.4f;  // レイマーチの移動係数

    // PBFパラメータ（CPU版の簡易実装）
    float                      m_restDensity = 1000.0f;   // 基準密度
    float                      m_particleMass = 1.0f;     // 粒子質量
    float                      m_smoothingRadius = 0.12f; // カーネル半径
    float                      m_epsilon = 100.0f;        // 安定化項
    int                        m_solverIterations = 3;    // ソルバ反復回数
    DirectX::XMFLOAT3          m_gravity = { 0.0f, -9.8f, 0.0f }; // 重力

    void StepCPU(float dt);             // CPUでPBF計算を行う
    void UpdateParticleBuffer();        // GPUへ粒子メタデータを転送

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
