#pragma once
#include "ComPtr.h"
#include <d3dx12.h>
#include <string>

class PipelineState
{
public:
	PipelineState();	// RXgN^łx̐ݒ
	bool IsValid();		// ɐǂԂ

	void SetInputLayout(D3D12_INPUT_LAYOUT_DESC layout);            // ̓CAEgݒ
        void SetRootSignature(ID3D12RootSignature* rootSignature);      // [gVOl`ݒ
        void SetVS(std::wstring filePath);                                                      // VSシェーダーを設定
        void SetPS(std::wstring filePath);                                                      // PSシェーダーを設定
        void SetDepthStencilFormat(DXGI_FORMAT format);                                         // 深度ステンシルフォーマット設定
        void SetBlendState(const D3D12_BLEND_DESC& blend);                                      // ブレンド設定を上書き
        void SetRenderTargetFormat(DXGI_FORMAT format);                                         // RTVフォーマット設定
        void Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE type);                                        // グラフィックスパイプラインを生成
	ID3D12PipelineState* Get();

private:
	bool m_IsValid = false;										// ɐǂ
	D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};				// pCvCXe[g̐ݒ
	ComPtr<ID3D12PipelineState> m_pPipelineState = nullptr;		// pCvCXe[g
	ComPtr<ID3DBlob> m_pVsBlob;									// _VF[_[
	ComPtr<ID3DBlob> m_pPSBlob;									// sNZVF[_[
};

