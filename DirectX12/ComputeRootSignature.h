#pragma once
#include "RootSignature.h"

// GPU ベースの SPH 計算で使用するルートシグネチャ
class ComputeRootSignature : public RootSignature {
public:
    bool InitForSPH() {
        // b0: SPHParams
        // b1: ViewProj
        // t0: inParticles
        // u0: outParticles
        // u1: outMeta
        // u2: gridCount
        // u3: gridTable

        // ルートパラメーターの配列（計7個）
        CD3DX12_ROOT_PARAMETER params[7] = {};

        // CBV b0 (SPH の各種パラメーター)
        params[0].InitAsConstantBufferView(0);

        // CBV b1 (ビュー・プロジェクション行列)
        params[1].InitAsConstantBufferView(1);

        // SRV t0 (入力パーティクル)
        CD3DX12_DESCRIPTOR_RANGE rangeSRV;
        rangeSRV.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
        params[2].InitAsDescriptorTable(1, &rangeSRV, D3D12_SHADER_VISIBILITY_ALL);

        // UAV u0 (更新後パーティクル書き込み先)
        CD3DX12_DESCRIPTOR_RANGE rangeUAV0;
        rangeUAV0.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
        params[3].InitAsDescriptorTable(1, &rangeUAV0, D3D12_SHADER_VISIBILITY_ALL);

        // UAV u1 (メタボール用情報書き込み先)
        CD3DX12_DESCRIPTOR_RANGE rangeUAV1;
        rangeUAV1.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1);
        params[4].InitAsDescriptorTable(1, &rangeUAV1, D3D12_SHADER_VISIBILITY_ALL);

        // UAV u2 (グリッド内粒子数テーブル)
        CD3DX12_DESCRIPTOR_RANGE rangeUAV2;
        rangeUAV2.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2);
        params[5].InitAsDescriptorTable(1, &rangeUAV2, D3D12_SHADER_VISIBILITY_ALL);

        // UAV u3 (グリッド内粒子インデックス)
        CD3DX12_DESCRIPTOR_RANGE rangeUAV3;
        rangeUAV3.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 3);
        params[6].InitAsDescriptorTable(1, &rangeUAV3, D3D12_SHADER_VISIBILITY_ALL);

        // ルートシグネチャを生成
        CD3DX12_ROOT_SIGNATURE_DESC desc;
        desc.Init(_countof(params), params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

        return Init(desc);
    }
};

