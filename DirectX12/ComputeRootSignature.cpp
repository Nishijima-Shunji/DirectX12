// ComputeRootSignature.cpp
#include "ComputeRootSignature.h"
#include <d3dx12.h>

bool ComputeRootSignature::InitForSPH()
{
    // SPHParams 用 CBV(b0)
    CD3DX12_ROOT_PARAMETER params[2];
    params[0].InitAsConstantBufferView(
        /*register*/ 0,                                    // b0
        /*space   */ 0,                                    // default
        D3D12_SHADER_VISIBILITY_ALL);

    // Particle バッファ用 SRV(t0) + UAV(u0) をまとめて DescriptorTable に
    CD3DX12_DESCRIPTOR_RANGE ranges[2];
    ranges[0].Init(
        D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        /*numDescriptors*/      1,
        /*baseShaderRegister*/  0,   // t0
        /*registerSpace*/       0);
    ranges[1].Init(
        D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        /*numDescriptors*/      1,
        /*baseShaderRegister*/  0,   // u0
        /*registerSpace*/       0);
    params[1].InitAsDescriptorTable(_countof(ranges), ranges, D3D12_SHADER_VISIBILITY_ALL);

    // RootSignatureDesc の設定
    D3D12_ROOT_SIGNATURE_DESC desc{};
	desc.NumParameters = _countof(params);              // パラメータの数
	desc.pParameters = params;                          // パラメータを設定
	desc.NumStaticSamplers = 0;                         // 静的サンプラーは不要
    desc.pStaticSamplers = nullptr; 
	desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;        // 計算用なので特にフラグは不要

    // 4) シリアライズ＆生成
    return this->Init(desc);  // RootSignature::Init(desc) を呼び出す
}
