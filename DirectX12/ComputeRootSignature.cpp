// ComputeRootSignature.cpp
#include "ComputeRootSignature.h"
#include <d3dx12.h>

bool ComputeRootSignature::InitForSPH()
{
    // SPHParams �p CBV(b0)
    CD3DX12_ROOT_PARAMETER params[2];
    params[0].InitAsConstantBufferView(
        /*register*/ 0,                                    // b0
        /*space   */ 0,                                    // default
        D3D12_SHADER_VISIBILITY_ALL);

    // Particle �o�b�t�@�p SRV(t0) + UAV(u0) ���܂Ƃ߂� DescriptorTable ��
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

    // RootSignatureDesc �̐ݒ�
    D3D12_ROOT_SIGNATURE_DESC desc{};
	desc.NumParameters = _countof(params);              // �p�����[�^�̐�
	desc.pParameters = params;                          // �p�����[�^��ݒ�
	desc.NumStaticSamplers = 0;                         // �ÓI�T���v���[�͕s�v
    desc.pStaticSamplers = nullptr; 
	desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;        // �v�Z�p�Ȃ̂œ��Ƀt���O�͕s�v

    // 4) �V���A���C�Y������
    return this->Init(desc);  // RootSignature::Init(desc) ���Ăяo��
}
