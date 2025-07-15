#pragma once
#include "Comptr.h"
#include "ComputePipelineState.h"
#include "PipelineState.h"
#include <d3d12.h>
#include <wrl.h>
#include <DirectXMath.h>
using Microsoft::WRL::ComPtr;

// CPU��GPU ���L�̗��q���^�f�[�^
struct ParticleMeta {
    DirectX::XMFLOAT3 pos;   // ���[���h��Ԉʒu
    float               r;   // ���a
};

class FluidSystem {
public:
    // ������: �f�o�C�X�ERTV�`���E�ő嗱�q���E�X���b�h�O���[�v��
    void Init(ID3D12Device* device, DXGI_FORMAT rtvFormat,
        UINT maxParticles, UINT threadGroupCount);

    // CPU/GPU �؂�ւ�
    void UseGPU(bool enable) { m_useGpu = enable; }

    // �V�~�����[�V���� (CPU or ComputeShader)
    void Simulate(ID3D12GraphicsCommandList* cmd, float dt);

    // 3D���^�{�[���`��
    void Render(ID3D12GraphicsCommandList* cmd,
        const DirectX::XMFLOAT4X4& invViewProj,
        const DirectX::XMFLOAT3& camPos,
        float isoLevel);

private:
    // CPU ���p�[�e�B�N���z��
    std::vector<ParticleMeta>     m_cpuParticles;

    // GPU �p�o�b�t�@ (SRV/UAV���p)
    ComPtr<ID3D12Resource>        m_particleBuffer;
    ComPtr<ID3D12DescriptorHeap>  m_uavHeap;      // UAV �����q�[�v
    ComPtr<ID3D12DescriptorHeap>  m_graphicsSrvHeap; // SRV �p�q�[�v

    // �R���s���[�g�p�p�C�v���C��
    ComPtr<ID3D12RootSignature>    m_computeRS;
    ComputePipelineState           m_computePS;

    // �`��p�p�C�v���C��
    ComPtr<ID3D12RootSignature>    m_graphicsRS;
    PipelineState* m_graphicsPS;
    ComPtr<ID3D12Resource>         m_graphicsCB;

    UINT m_maxParticles;
    UINT m_threadGroupCount;
    bool m_useGpu = false;
};
