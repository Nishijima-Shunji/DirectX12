#pragma once
#include "RootSignature.h"

class ComputeRootSignature : public RootSignature {
public:
    bool InitForSPH()
    {
    // b0: SPHParams
    // b1: ViewProj 行列
    // t0: inParticles
    // u0: outParticles
    // u1: outMeta

        CD3DX12_DESCRIPTOR_RANGE srvRange;
        srvRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0); // t0

        CD3DX12_ROOT_PARAMETER params[5] = {};

        // CBV b0
        params[0].InitAsConstantBufferView(0);

        // CBV b1
        params[1].InitAsConstantBufferView(1);

        // SRV t0 (inParticles)
        CD3DX12_DESCRIPTOR_RANGE rangeSRV;
        rangeSRV.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
        params[2].InitAsDescriptorTable(1, &rangeSRV, D3D12_SHADER_VISIBILITY_ALL);

        // UAV u0 (outParticles)
        CD3DX12_DESCRIPTOR_RANGE rangeUAV0;
        rangeUAV0.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
        params[3].InitAsDescriptorTable(1, &rangeUAV0, D3D12_SHADER_VISIBILITY_ALL);

        // UAV u1 (outMeta)
        CD3DX12_DESCRIPTOR_RANGE rangeUAV1;
        rangeUAV1.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1);
        params[4].InitAsDescriptorTable(1, &rangeUAV1, D3D12_SHADER_VISIBILITY_ALL);

        CD3DX12_ROOT_SIGNATURE_DESC desc;
        desc.Init(_countof(params), params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

        return Init(desc);
    }

};
