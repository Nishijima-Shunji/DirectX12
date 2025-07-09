#pragma once
#include "RootSignature.h"

class ComputeRootSignature : public RootSignature {
public:
    bool InitForSPH()
    {
        // 0: b0, 1: t0, 2: u0, 3: u1
        CD3DX12_ROOT_PARAMETER params[4] = {};

        // CBV b0
        params[0].InitAsConstantBufferView(0);

        // SRV t0 (inParticles)
        CD3DX12_DESCRIPTOR_RANGE rangeSRV;
        rangeSRV.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
        params[1].InitAsDescriptorTable(1, &rangeSRV, D3D12_SHADER_VISIBILITY_ALL);

        // UAV u0 (outParticles)
        CD3DX12_DESCRIPTOR_RANGE rangeUAV0;
        rangeUAV0.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
        params[2].InitAsDescriptorTable(1, &rangeUAV0, D3D12_SHADER_VISIBILITY_ALL);

        // UAV u1 (outMeta)
        CD3DX12_DESCRIPTOR_RANGE rangeUAV1;
        rangeUAV1.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1);
        params[3].InitAsDescriptorTable(1, &rangeUAV1, D3D12_SHADER_VISIBILITY_ALL);

        // ルートシグネイチャの定義
        CD3DX12_ROOT_SIGNATURE_DESC desc;
        desc.Init(
            _countof(params), params,
            0, nullptr,
            D3D12_ROOT_SIGNATURE_FLAG_NONE
        );

        return Init(desc);
    }

};
