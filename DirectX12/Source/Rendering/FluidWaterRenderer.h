#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <memory>
#include <cstdint>
#include <DirectXMath.h>
#include "ConstantBuffer.h"

class Camera;
class FluidSystem;
class SSRRenderer;

// ================================
// 水レンダリング用の定数バッファ構造体
// ================================
struct FluidCameraConstants
{
    DirectX::XMFLOAT4X4 View;        // ビュー行列（転置済み）
    DirectX::XMFLOAT4X4 Proj;        // プロジェクション行列（転置済み）
    DirectX::XMFLOAT2   ScreenSize;  // 画面サイズ
    DirectX::XMFLOAT2   InvScreenSize; // 逆画面サイズ
    float NearZ;                     // ニアクリップ
    float FarZ;                      // ファークリップ
    DirectX::XMFLOAT3 IorF0;         // フレネルF0
    float Absorb;                    // 吸収係数
    DirectX::XMUINT4 Options;        // x:DebugView y:SSRQuality z:Planar(0/1) w:GridDebug(0/1)
};

struct FluidParamConstants
{
    float Radius;            // 粒子半径
    float ThicknessScale;    // 厚みスケール
    uint32_t ParticleCount;  // 粒子数
    float Downsample;        // ダウンサンプル倍率（1/2/4）
};

enum class FluidDebugView
{
    Composite = 0,
    Depth = 1,
    Thickness = 2,
    Normal = 3
};

class FluidWaterRenderer
{
public:
    FluidWaterRenderer();
    bool Init(ID3D12Device* device, UINT width, UINT height);
    void Resize(UINT width, UINT height);
    void BeginSceneRender(ID3D12GraphicsCommandList* cmd);
    void EndSceneRender(ID3D12GraphicsCommandList* cmd);
    void RenderDepthThickness(ID3D12GraphicsCommandList* cmd, const Camera& camera, FluidSystem& fluid);
    void BlurAndNormal(ID3D12GraphicsCommandList* cmd);
    void Composite(ID3D12GraphicsCommandList* cmd);

    // デバッグ表示設定
    void SetDebugView(FluidDebugView view) { m_debugView = view; }
    void SetSSRQuality(uint32_t quality) { m_ssrQuality = quality; }
    void TogglePlanarReflection(bool enable) { m_usePlanarReflection = enable; }
    void SetDownsample(uint32_t step);
    void EnableGridDebug(bool enable) { m_showGrid = enable; }

    // シーンカラー／深度を外部へ提供
    D3D12_CPU_DESCRIPTOR_HANDLE SceneColorRTV() const { return m_sceneColorRTV; }
    D3D12_CPU_DESCRIPTOR_HANDLE SceneDepthDSV() const { return m_sceneDepthDSV; }
    ID3D12Resource* SceneColorResource() const { return m_sceneColor.Get(); }
    ID3D12Resource* SceneDepthResource() const { return m_sceneDepthBuffer.Get(); }

    // SRV／UAVヒープ取得
    ID3D12DescriptorHeap* DescriptorHeap() const { return m_srvUavHeap.Get(); }
    ID3D12DescriptorHeap* SamplerHeap() const { return m_samplerHeap.Get(); }
    D3D12_GPU_DESCRIPTOR_HANDLE GpuHandle(UINT index) const;
    D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle(UINT index) const;

    // 各種テクスチャを他モジュールへ
    ID3D12Resource* FluidDepthTexture() const { return m_fluidDepth.Get(); }
    ID3D12Resource* FluidNormalTexture() const { return m_fluidNormal.Get(); }
    uint32_t SSRQuality() const { return m_ssrQuality; }

    // SSRからのカラーをSRVとして登録するためのヘルパ
    void UpdateSSRColorSRV(ID3D12Resource* resource);

    // 現在の解像度取得
    UINT Width() const { return m_width; }
    UINT Height() const { return m_height; }

private:
    void CreateHeaps(ID3D12Device* device);
    void CreateConstantBuffers(ID3D12Device* device);
    void CreateRenderTargets(ID3D12Device* device);
    void CreateRootSignatures(ID3D12Device* device);
    void CreatePipelineStates(ID3D12Device* device);
    void CreateSamplers(ID3D12Device* device);

    void UpdateCameraConstants(const Camera& camera);
    void UpdateFluidConstants(const FluidSystem& fluid);

    void Transition(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource, D3D12_RESOURCE_STATES& state, D3D12_RESOURCE_STATES after);

private:
    bool m_initialized = false;
    ID3D12Device* m_device = nullptr;
    UINT m_width = 0;
    UINT m_height = 0;
    uint32_t m_downsample = 1;

    // ディスクリプタヒープ
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_samplerHeap;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_dsvHeap;
    UINT m_srvUavIncrement = 0;
    UINT m_rtvIncrement = 0;
    UINT m_dsvIncrement = 0;

    D3D12_CPU_DESCRIPTOR_HANDLE m_sceneColorRTV{};
    D3D12_CPU_DESCRIPTOR_HANDLE m_fluidThicknessRTV{};
    D3D12_CPU_DESCRIPTOR_HANDLE m_fluidDepthRTV{};
    D3D12_CPU_DESCRIPTOR_HANDLE m_sceneDepthDSV{};
    D3D12_CPU_DESCRIPTOR_HANDLE m_fluidDSV{};

    D3D12_GPU_DESCRIPTOR_HANDLE m_srvUavGpuStart{};
    D3D12_CPU_DESCRIPTOR_HANDLE m_srvUavCpuStart{};

    // 定数バッファ
    std::unique_ptr<ConstantBuffer> m_cameraCB;
    std::unique_ptr<ConstantBuffer> m_fluidCB;

    // テクスチャリソース
    Microsoft::WRL::ComPtr<ID3D12Resource> m_sceneColor;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_sceneDepthBuffer; // typeless depth
    Microsoft::WRL::ComPtr<ID3D12Resource> m_fluidDepth;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_fluidThickness;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_fluidDepthBlur;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_fluidNormal;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_fluidDepthStencil;

    D3D12_RESOURCE_STATES m_sceneColorState = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES m_sceneDepthState = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES m_fluidDepthState = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES m_fluidThicknessState = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES m_fluidDepthBlurState = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES m_fluidNormalState = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES m_fluidDepthStencilState = D3D12_RESOURCE_STATE_DEPTH_WRITE;

    // ルートシグネチャ / PSO
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_depthThicknessRS;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_blurRS;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_normalRS;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_compositeRS;

    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_depthThicknessPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_blurXPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_blurYPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_normalPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_compositePSO;

    // デバッグ / 設定
    FluidDebugView m_debugView = FluidDebugView::Composite;
    uint32_t m_ssrQuality = 1;
    bool m_usePlanarReflection = false;
    bool m_showGrid = false;
};
