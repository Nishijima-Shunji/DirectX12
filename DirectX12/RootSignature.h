#pragma once
#include "ComPtr.h"
#include <d3d12.h>

struct ID3D12RootSignature;

class RootSignature
{
public:
	RootSignature();			// コンストラクタでルートシグネチャを生成
	bool IsValid();				// ルートシグネチャの生成に成功したかどうかを返す
	ID3D12RootSignature* Get(); // ルートシグネチャを返す
	bool Init(const D3D12_ROOT_SIGNATURE_DESC& desc);

private:
	bool m_IsValid = false;									// ルートシグネチャの生成に成功したかどうか
	ComPtr<ID3D12RootSignature> m_pRootSignature = nullptr; // ルートシグネチャ
};


