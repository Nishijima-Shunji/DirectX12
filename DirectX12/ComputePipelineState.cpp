#include "ComputePipelineState.h"
#include "Engine.h"
#include <d3dcompiler.h>
#pragma comment(lib, "d3dcompiler.lib")	// hlsl�������^�C���ɃR���p�C�����邽�߂ɕK�v (�ǉ��̈ˑ��t�@�C���Őݒ肷��̂Ɠ���)

ComputePipelineState::ComputePipelineState() {
    // �f�t�H���g�������i�K�v�Ȃ�ς���j
    desc.NodeMask = 0;
}

bool ComputePipelineState::IsValid() const {
    return m_isValid;
}

void ComputePipelineState::SetRootSignature(ID3D12RootSignature* rs) {
    desc.pRootSignature = rs;
}

void ComputePipelineState::SetCS(const std::wstring& filePath) {
    auto hr = D3DReadFileToBlob(filePath.c_str(), m_csBlob.GetAddressOf());
    if (FAILED(hr)) {
        printf("ComputeShader �̓ǂݍ��݂Ɏ��s: %ls\n", filePath.c_str());
        return;
    }
    desc.CS = CD3DX12_SHADER_BYTECODE(m_csBlob.Get());  
}

void ComputePipelineState::Create() {
    HRESULT hr = g_Engine->Device()->CreateComputePipelineState(&desc, IID_PPV_ARGS(m_pPSO.GetAddressOf()));
    if (SUCCEEDED(hr)) m_isValid = true;
    else         printf("Compute PSO �����Ɏ��s HRESULT=0x%08X\n", hr);
}

ID3D12PipelineState* ComputePipelineState::Get() const {
    return m_pPSO.Get();
}