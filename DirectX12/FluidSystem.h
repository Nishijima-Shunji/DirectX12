#pragma once
#include "ComPtr.h"
#include <d3d12.h>
#include <wrl.h>
#include <DirectXMath.h>
#include "ConstantBuffer.h"
#include <vector>

struct FluidParticle {
    DirectX::XMFLOAT3 position; float _pad0 = 0;
    DirectX::XMFLOAT3 velocity; float _pad1 = 0;
    DirectX::XMFLOAT3 x_pred;   float lambda = 0;
    float density = 0; float _pad2[3] = { 0,0,0 };
};

class FluidSystem {
private:
    // CPU側
    std::vector<FluidParticle>             m_cpuParticles;

    // 粒子バッファ（SRV/UAV）
    ComPtr<ID3D12Resource>                 m_particleBuffer;

    // 近傍グリッド（GPU）
    ComPtr<ID3D12Resource>                 m_cellStart;
    ComPtr<ID3D12Resource>                 m_cellEnd;
    ComPtr<ID3D12Resource>                 m_sortedIndices;

    // ヒープ
    ComPtr<ID3D12DescriptorHeap>           m_uavHeap;
    ComPtr<ID3D12DescriptorHeap>           m_graphicsSrvHeap;

    // PBF: RS/PSO/CB
    ComPtr<ID3D12RootSignature>            m_rsPBF;
    ComPtr<ID3D12PipelineState>            m_psoPredict;
    ComPtr<ID3D12PipelineState>            m_psoLambda;
    ComPtr<ID3D12PipelineState>            m_psoDeltaP;
    ComPtr<ID3D12PipelineState>            m_psoVelocity;
    ConstantBuffer* m_pbfParamCB = nullptr;

    // SSFR: RS/PSO/RT
    ComPtr<ID3D12RootSignature>            m_rsSSFR_Gfx;
    ComPtr<ID3D12RootSignature>            m_rsSSFR_Compute;
    ComPtr<ID3D12PipelineState>            m_psoSSFR_Particle;
    ComPtr<ID3D12PipelineState>            m_psoSSFR_Bilateral;
    ComPtr<ID3D12PipelineState>            m_psoSSFR_Normal;
    ComPtr<ID3D12PipelineState>            m_psoSSFR_Composite;
    ComPtr<ID3D12Resource>                 m_texFluidDepth;
    ComPtr<ID3D12Resource>                 m_texThickness;
    ComPtr<ID3D12Resource>                 m_texFluidNormal;
    ComPtr<ID3D12Resource>                 m_cbSSFR_Camera;

    // 表示設定
    UINT                                   m_viewWidth = 0, m_viewHeight = 0;
    DXGI_FORMAT                            m_mainRTFormat{};
    UINT                                   m_sphereIndexCount = 0; // 粒子スプラット用
    UINT                                   m_ssfrScale = 2;

    // グリッド設定
    UINT                                   m_gridDimX = 1, m_gridDimY = 1, m_gridDimZ = 1;
    UINT                                   m_cellCount = 1;
    DirectX::XMFLOAT3                      m_gridMin{ -1.0f, -1.0f, -1.0f };

    // 状態
    bool                                   m_particleInSrvState = false;
    bool                                   m_gridInSrvState = false;
    bool                                   m_useGpu = true;

    // ドラッグ
    int                                    m_dragIndex = -1;
    float                                  m_dragDepth = 0.0f;

public:
    void Init(ID3D12Device* device, DXGI_FORMAT rtvFormat, UINT maxParticles, UINT threadGroupCount);
    void UseGPU(bool enable) { m_useGpu = enable; }

    void Simulate(ID3D12GraphicsCommandList* cmd, float dt); // PBF更新
    void Render(ID3D12GraphicsCommandList* cmd,
        const DirectX::XMFLOAT4X4& invViewProj,
        const DirectX::XMFLOAT3& camPos,
        float isoLevel); // 中でSSFR合成を呼ぶ（isoLevelは互換のため残置OK）

    void StartDrag(int mouseX, int mouseY, class Camera* cam);
    void Drag(int mouseX, int mouseY, class Camera* cam);
    void EndDrag();
};
