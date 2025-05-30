#include "ParticlePipelineState.h"
#include "Engine.h"
#include <d3dx12.h>
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")

ParticlePipelineState::ParticlePipelineState()
{
	// パイプラインステートの設定
	
	// ラスタライザーの設定
	desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);			// ラスタライザーはデフォルト
	desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;					// カリングはなし
	desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;					// 塗りつぶしはソリッド

	// 深度ステンシルの設定
	desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);		// 深度ステンシルはデフォルト
	desc.DepthStencilState.DepthEnable = TRUE;								// 深度ステンシルを無効
	desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;	// 深度書き込みは無効
	desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;		// 深度比較関数は常に通過

	desc.SampleMask = UINT_MAX;
	desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;	// 描画方法
	desc.NumRenderTargets = 1;												// 描画対象は1
	desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;

	// サンプル数の設定
	desc.SampleDesc.Count = 1;												// サンプラーは1
	desc.SampleDesc.Quality = 0;

	// ブレンドステートの設定
	desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);					// ブレンドステートもデフォルト
	desc.BlendState.RenderTarget[0].BlendEnable = TRUE;						// ブレンドは有効
	desc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;		// ソースブレンドはアルファ
	desc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;	// デストブレンドは逆アルファ
	desc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;			// ブレンド演算は加算
	desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL; // 全ての色成分を書き込み可能
}

bool ParticlePipelineState::IsValid()
{
	return m_IsValid;
}

void ParticlePipelineState::SetInputLayout(D3D12_INPUT_LAYOUT_DESC layout)
{
	desc.InputLayout = layout;
}

void ParticlePipelineState::SetRootSignature(ID3D12RootSignature* rootSignature)
{
	desc.pRootSignature = rootSignature;
}

void ParticlePipelineState::SetVS(std::wstring filePath)
{
	// 頂点シェーダー読み込み
	auto hr = D3DReadFileToBlob(filePath.c_str(), m_pVsBlob.GetAddressOf());
	if (FAILED(hr))
	{
		printf("頂点シェーダーの読み込みに失敗\n");
		return;
	}

	desc.VS = CD3DX12_SHADER_BYTECODE(m_pVsBlob.Get());
}

void ParticlePipelineState::SetPS(std::wstring filePath)
{
	// ピクセルシェーダー読み込み
	auto hr = D3DReadFileToBlob(filePath.c_str(), m_pPSBlob.GetAddressOf());
	if (FAILED(hr))
	{
		printf("ピクセルシェーダーの読み込みに失敗\n");
		return;
	}

	desc.PS = CD3DX12_SHADER_BYTECODE(m_pPSBlob.Get());
}

void ParticlePipelineState::Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE type)
{
	desc.PrimitiveTopologyType = type;
	// パイプラインステートを生成
	auto hr = g_Engine->Device()->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(m_pPipelineState.ReleaseAndGetAddressOf()));
	if (FAILED(hr))
	{
		printf("パイプラインステートの生成に失敗 HRESULT=0x%08X\n", hr);
		return;
	}

	m_IsValid = true;
}

ID3D12PipelineState* ParticlePipelineState::Get()
{
	return m_pPipelineState.Get();
}
