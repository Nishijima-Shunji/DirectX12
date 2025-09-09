	#pragma once
#include <d3d12.h>
#include <dxgi.h>
#include <dxgi1_4.h>
#include "ComPtr.h"
#include "DescriptorHeap.h" 
#include <unordered_map>
#include <string>
#include "Object.h"

#pragma comment(lib, "d3d12.lib") // d3d12���C�u�����������N����
#pragma comment(lib, "dxgi.lib") // dxgi���C�u�����������N����

class Engine
{
public:
	enum { FRAME_BUFFER_COUNT = 2 }; // �_�u���o�b�t�@�����O����̂�2

public:
	bool Init(HWND hwnd, UINT windowWidth, UINT windowHeight); // �G���W��������

	void BeginRender(); // �`��̊J�n����
	void EndRender();	// �`��̏I������
	void Flush();       // GPU �����ׂďI���܂ő҂�

public: // �O����A�N�Z�X���邽�߂�Getter
	ID3D12Device6*				Device();
	ID3D12GraphicsCommandList*	CommandList();
	UINT						CurrentBackBufferIndex();
	UINT						FrameBufferWidth() const; // �t���[���o�b�t�@�̕����擾
	UINT						FrameBufferHeight() const; // �t���[���o�b�t�@�̍������擾
	DescriptorHeap *			CbvSrvUavHeap() const { return m_pCbvSrvUavHeap; } // CBV/SRV/UAV �p DescriptorHeap ���擾
    ID3D12CommandQueue*			CommandQueue() const { return m_pQueue.Get(); } // �R�}���h�L���[���擾
    ID3D12CommandQueue*			ComputeCommandQueue() const { return m_pComputeQueue.Get(); } // �R�}���h�L���[���擾
    ID3D12CommandAllocator*		CommandAllocator(UINT index) const { return m_pAllocator[index].Get(); } // �R�}���h�A���P�[�^�[���擾

public:
	// �I�u�W�F�N�g�𖼑O�œo�^�E�擾���邽�߂̊֐�
	template<typename T>
	void RegisterObj(const std::string& name, T* obj) {
		m_namedObjects[name] = obj;
	}

	template<typename T>
	T* GetObj(const std::string& name) {
		auto it = m_namedObjects.find(name);
		if (it != m_namedObjects.end()) {
			return dynamic_cast<T*>(it->second);  // ���S�ɃL���X�g
		}
		return nullptr;
	}

private: // DirectX12�������Ɏg���֐�
	bool CreateDevice();		// �f�o�C�X�𐶐�
	bool CreateCommandQueue();	// �R�}���h�L���[�𐶐�
	bool CreateSwapChain();		// �X���b�v�`�F�C���𐶐�
	bool CreateCommandList();	// �R�}���h���X�g�ƃR�}���h�A���P�[�^�[�𐶐�
	bool CreateFence();			// �t�F���X�𐶐�
	void CreateViewPort();		// �r���[�|�[�g�𐶐�
	void CreateScissorRect();	// �V�U�[��`�𐶐�

private: // �`��Ɏg��DirectX12�̃I�u�W�F�N�g����
	HWND m_hWnd;
	UINT m_FrameBufferWidth = 0;
	UINT m_FrameBufferHeight = 0;
	UINT m_CurrentBackBufferIndex = 0;

	ComPtr<ID3D12Device6> m_pDevice = nullptr;										// �f�o�C�X
	ComPtr<ID3D12CommandQueue> m_pQueue = nullptr;									// �R�}���h�L���[
	ComPtr<ID3D12CommandQueue> m_pComputeQueue = nullptr;							// �R���s���[�g�V�F�[�_�[�p�R�}���h�L���[
	ComPtr<IDXGISwapChain3> m_pSwapChain = nullptr;									// �X���b�v�`�F�C��
	ComPtr<ID3D12CommandAllocator> m_pAllocator[FRAME_BUFFER_COUNT] = { nullptr };	// �R�}���h�A���P�[���[
	ComPtr<ID3D12GraphicsCommandList> m_pCommandList = nullptr;						// �R�}���h���X�g
	HANDLE m_fenceEvent = nullptr;													// �t�F���X�Ŏg���C�x���g
	ComPtr<ID3D12Fence> m_pFence = nullptr;											// �t�F���X
	UINT64 m_fenceValue[FRAME_BUFFER_COUNT];										// �t�F���X�̒l�i�_�u���o�b�t�@�����O�p��2�j
	D3D12_VIEWPORT m_Viewport;														// �r���[�|�[�g
	D3D12_RECT m_Scissor;															// �V�U�[��`
	std::unordered_map<std::string, Object*> m_namedObjects;						// ���O�t���I�u�W�F�N�g�̃}�b�v

	// CBV/SRV/UAV �p�̃f�B�X�N���v�^�q�[�v�Ǘ��N���X
	DescriptorHeap * m_pCbvSrvUavHeap = nullptr;

private: // �`��Ɏg���I�u�W�F�N�g�Ƃ��̐����֐�����
	bool CreateRenderTarget(); // �����_�[�^�[�Q�b�g�𐶐�
	bool CreateDepthStencil(); // �[�x�X�e���V���o�b�t�@�𐶐�

	UINT m_RtvDescriptorSize = 0; // �����_�[�^�[�Q�b�g�r���[�̃f�B�X�N���v�^�T�C�Y
	ComPtr<ID3D12DescriptorHeap> m_pRtvHeap = nullptr; // �����_�[�^�[�Q�b�g�̃f�B�X�N���v�^�q�[�v
	ComPtr<ID3D12Resource> m_pRenderTargets[FRAME_BUFFER_COUNT] = { nullptr }; // �����_�[�^�[�Q�b�g�i�_�u���o�b�t�@�����O����̂�2�j

	UINT m_DsvDescriptorSize = 0; // �[�x�X�e���V���̃f�B�X�N���v�^�[�T�C�Y
	ComPtr<ID3D12DescriptorHeap> m_pDsvHeap = nullptr; // �[�x�X�e���V���̃f�B�X�N���v�^�q�[�v
	ComPtr<ID3D12Resource> m_pDepthStencilBuffer = nullptr; // �[�x�X�e���V���o�b�t�@�i��������1�ł����j

private: // �`�惋�[�v�Ŏg�p�������
	ID3D12Resource* m_currentRenderTarget = nullptr; // ���݂̃t���[���̃����_�[�^�[�Q�b�g���ꎞ�I�ɕۑ����Ă����֐�
	void WaitRender(); // �`�抮����҂���
};

extern Engine* g_Engine; // �ǂ�����ł��Q�Ƃ������̂ŃO���[�o���ɂ���
