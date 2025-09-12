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
        ComPtr<ID3D12Resource>        m_gridCountUpload;    // グリッドカウント転送用
        ComPtr<ID3D12Resource>        m_gridTableUpload;    // グリッドテーブル転送用

	// Compute 用定数バッファ
	ConstantBuffer* m_sphParamCB = nullptr;
	ConstantBuffer* m_viewProjCB = nullptr;

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
	DirectX::XMFLOAT3 m_gridMin{ -1.0f, -1.0f, -1.0f };
	ConstantBuffer* m_cbGridParam;

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
        // グリッドバッファがシェーダリソース状態にあるか
        bool m_gridInSrvState = false;

	// ドラッグ中の粒子インデックスとその距離
	int   m_dragIndex = -1;
	float m_dragDepth = 0.0f;


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

};
