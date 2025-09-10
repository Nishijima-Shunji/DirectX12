#pragma once
#include "ComPtr.h"
#include "ComputePipelineState.h"
#include "PipelineState.h"
#include <d3d12.h>
#include <wrl.h>
#include <DirectXMath.h>
#include "SharedStruct.h"
#include "ConstantBuffer.h"
#include "SpatialGrid.h"
#include <vector>
#include "FullscreenPSO.h"

struct FluidParticle {
    DirectX::XMFLOAT3 position;
    DirectX::XMFLOAT3 velocity;
};

class FluidSystem {
private:
    // CPU 側パーティクル配列
    std::vector<FluidParticle>    m_cpuParticles;

    // GPU 用バッファ (SRV/UAV共用)
    ComPtr<ID3D12Resource>        m_particleBuffer;     // シミュレーションするパーティクル情報
	ComPtr<ID3D12Resource>        m_metaBuffer;         // レンダリング用メタボール情報
    ComPtr<ID3D12DescriptorHeap>  m_uavHeap;            // UAVを持つヒープ
    ComPtr<ID3D12DescriptorHeap>  m_graphicsSrvHeap;    // SRV用ヒープ

    // グリッド情報
	ComPtr<ID3D12Resource>        m_gridCount;          // 各セルの粒子数
	ComPtr<ID3D12Resource>        m_gridTable;          // セルごとの粒子インデックステーブル
	ComPtr<ID3D12Resource>        m_particleUpload;     // パーティクル転送用
	ComPtr<ID3D12Resource>        m_metaUpload;         // メタボール転送用

    // Compute 用定数バッファ
    ConstantBuffer*               m_sphParamCB = nullptr;
    ConstantBuffer*               m_viewProjCB = nullptr;

    // コンピュート用パイプライン
	ComPtr<ID3D12RootSignature>    m_computeRS;         // ルートシグネチャー
    ComputePipelineState           m_computePS;         // SPH計算
    ComputePipelineState           m_buildGridPS;       // グリッド生成

    // 描画用パイプライン
    ComPtr<ID3D12RootSignature>    m_graphicsRS;
    PipelineState* m_graphicsPS;
    ComPtr<ID3D12Resource>         m_graphicsCB;

    UINT m_maxParticles;
    UINT m_threadGroupCount;
    bool m_useGpu = false;

    // グリッド設定
    static const UINT MAX_PARTICLES_PER_CELL = 64;
    UINT m_gridDimX = 1;
    UINT m_gridDimY = 1;
    UINT m_gridDimZ = 1;
    UINT m_cellCount = 1;
    DirectX::XMFLOAT3 m_gridMin{-1.0f, -1.0f, -1.0f};

    // CPU 用パラメータ
    struct SPHParams {
        float restDensity = 1000.0f;
        float particleMass = 1.0f;
        float viscosity = 1.0f;
        float stiffness = 200.0f;
        float radius = 0.1f;
        float timeStep = 0.016f;
    } m_params;

    SpatialGrid m_grid{ 0.1f };

    // CPU シミュレーション用一時バッファ
    std::vector<float> m_density;
    std::vector<size_t> m_neighborBuffer;

    // メタバッファがシェーダリソース状態にあるか
    bool m_metaInSrvState = false;
    // パーティクルバッファがシェーダリソース状態にあるか
    bool m_particleInSrvState = false;

    // ドラッグ中の粒子インデックスとその距離
    int   m_dragIndex = -1;
    float m_dragDepth = 0.0f;


        // 初期状態ではスクリーンスペースエフェクトを無効化
        bool m_useScreenSpace = false;

    // 低解像度（1/2）蓄積RT と ブラー用RT
    Microsoft::WRL::ComPtr<ID3D12Resource> m_accumTex; // R16_FLOAT or R32_FLOAT
    Microsoft::WRL::ComPtr<ID3D12Resource> m_blurTex;

    // RTV（CPU）と SRV（GPU可視）
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_rtvHeapSSA;       // RTV×2
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_srvHeapSSA;       // SRV×2 + サンプラは既存の静的サンプラでOK

    D3D12_CPU_DESCRIPTOR_HANDLE m_accumRTV = {};
    D3D12_CPU_DESCRIPTOR_HANDLE m_blurRTV = {};
    D3D12_GPU_DESCRIPTOR_HANDLE m_accumSRV = {};
    D3D12_GPU_DESCRIPTOR_HANDLE m_blurSRV = {};

    // ルートシグネチャとPSO
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rsAccum;     // 粒子→蓄積
    FullscreenPSO m_psoAccum;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rsBlur;      // ブラー
    FullscreenPSO m_psoBlur;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_rsComposite; // 合成
    FullscreenPSO m_psoComposite;

    // 定数バッファ（各パス用）
    Microsoft::WRL::ComPtr<ID3D12Resource> m_cbAccum;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_cbBlur;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_cbComposite;

    // 画面サイズ・縮小係数
    UINT m_viewWidth = 0;
    UINT m_viewHeight = 0;
    UINT m_ssaScale = 2; // 1/2解像度

    // 粒子SRV
    D3D12_GPU_DESCRIPTOR_HANDLE m_particleSRV = {};
    // 画面フォーマット保存
    DXGI_FORMAT m_mainRTFormat = DXGI_FORMAT_UNKNOWN;


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

        // 画面座標から粒子を選択/ドラッグするためのヘルパー
        void StartDrag(int mouseX, int mouseY, class Camera* cam);
        void Drag(int mouseX, int mouseY, class Camera* cam);
        void EndDrag();

        // 格子サイズ変更（CPU シミュレーション用）
        void SetSpatialCellSize(float s) { m_grid.SetCellSize(s); }

        // 初期化＆サイズ変更＆描画ヘルパ
        void CreateSSAResources(ID3D12Device* device, DXGI_FORMAT mainRTFormat, UINT viewW, UINT viewH);
        void DestroySSAResources();
        void CreateSSAPipelines(ID3D12Device* device, DXGI_FORMAT accumFormat);
        void UpdateSSAConstantBuffers(ID3D12GraphicsCommandList* cmd);
        void RenderSSA(ID3D12GraphicsCommandList* cmd);

};
