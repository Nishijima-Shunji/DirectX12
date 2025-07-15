#pragma once
#include "Comptr.h"
#include "ComputePipelineState.h"
#include "PipelineState.h"
#include <d3d12.h>
#include <wrl.h>
#include <DirectXMath.h>
using Microsoft::WRL::ComPtr;

// CPU→GPU 共有の粒子メタデータ
struct ParticleMeta {
    DirectX::XMFLOAT3 pos;   // ワールド空間位置
    float               r;   // 半径
};

class FluidSystem {
public:
    // 初期化: デバイス・RTV形式・最大粒子数・スレッドグループ数
    void Init(ID3D12Device* device, DXGI_FORMAT rtvFormat,
        UINT maxParticles, UINT threadGroupCount);

    // CPU/GPU 切り替え
    void UseGPU(bool enable) { m_useGpu = enable; }

    // シミュレーション (CPU or ComputeShader)
    void Simulate(ID3D12GraphicsCommandList* cmd, float dt);

    // 3Dメタボール描画
    void Render(ID3D12GraphicsCommandList* cmd,
        const DirectX::XMFLOAT4X4& invViewProj,
        const DirectX::XMFLOAT3& camPos,
        float isoLevel);

private:
    // CPU 側パーティクル配列
    std::vector<ParticleMeta>     m_cpuParticles;

    // GPU 用バッファ (SRV/UAV共用)
    ComPtr<ID3D12Resource>        m_particleBuffer;
    ComPtr<ID3D12DescriptorHeap>  m_uavHeap;      // UAV を持つヒープ
    ComPtr<ID3D12DescriptorHeap>  m_graphicsSrvHeap; // SRV 用ヒープ

    // コンピュート用パイプライン
    ComPtr<ID3D12RootSignature>    m_computeRS;
    ComputePipelineState           m_computePS;

    // 描画用パイプライン
    ComPtr<ID3D12RootSignature>    m_graphicsRS;
    PipelineState* m_graphicsPS;
    ComPtr<ID3D12Resource>         m_graphicsCB;

    UINT m_maxParticles;
    UINT m_threadGroupCount;
    bool m_useGpu = false;
};
