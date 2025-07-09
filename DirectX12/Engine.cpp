#include "Engine.h"
#include <d3d12.h>
#include <stdio.h>
#include <Windows.h>
#include <DirectXTex.h>
#include <d3dx12.h>
#include "DescriptorHeap.h"

Engine* g_Engine;

// デバッグ
template<typename T>
class DX12Object
{
protected:
	ComPtr<T> ptr = nullptr;

	DX12Object() = default;



public:
	T* operator->() { return ptr.Get(); }
	operator T* () { return ptr.Get(); }
	T** operator&() { return ptr.GetAddressOf(); }

	// コピーの禁止
	DX12Object(const DX12Object&) = delete;
	DX12Object& operator=(const DX12Object&) = delete;

};

class DebugLayer : public DX12Object<ID3D12Debug>
{
public:
	bool Create();
	bool Enable();
};

bool DebugLayer::Create()
{
	HRESULT hr = D3D12GetDebugInterface(
		IID_PPV_ARGS(ptr.ReleaseAndGetAddressOf())
	);

	if (FAILED(hr)) {
		return false;
	}

	return true;
}

bool DebugLayer::Enable()
{
	if (!ptr)
		return false;

	ptr->EnableDebugLayer();

	return true;
}





bool Engine::Init(HWND hwnd, UINT windowWidth, UINT windowHeight)
{
	m_FrameBufferWidth = windowWidth;
	m_FrameBufferHeight = windowHeight;
	m_hWnd = hwnd;

	DebugLayer debug;

	debug.Create();
	debug.Enable();


	// =====デバイスの初期化処理=====
	if (!CreateDevice())
	{
		printf("デバイスの生成に失敗");
		return false;
	}

	if (!CreateCommandQueue())
	{
		printf("コマンドキューの生成に失敗");
		return false;
	}

	if (!CreateSwapChain())
	{
		printf("スワップチェインの生成に失敗");
		return false;
	}

	if (!CreateCommandList())
	{
		printf("コマンドリストの生成に失敗");
		return false;
	}

	if (!CreateFence())
	{
		printf("フェンスの生成に失敗");
		return false;
	}

	// ビューポートとシザー矩形を生成
	CreateViewPort();
	CreateScissorRect();

	if (!CreateRenderTarget())
	{
		printf("レンダーターゲットの生成に失敗");
		return false;
	}

	if (!CreateDepthStencil())
	{
		printf("デプスステンシルバッファの生成に失敗\n");
		return false;
	}

	// --- CBV/SRV/UAV ディスクリプタヒープ生成 ---
	 m_pCbvSrvUavHeap = new DescriptorHeap();
	if (!m_pCbvSrvUavHeap/* || !m_pCbvSrvUavHeap->IsValid()*/) {
		printf("CBV/SRV/UAV DescriptorHeap の生成に失敗\n");
		return false;
		
	}

	printf("描画エンジンの初期化に成功\n");
	return true;
}

// コマンドキューを作成
bool Engine::CreateCommandQueue()
{
	// 描画用のコマンドキューを作成
	D3D12_COMMAND_QUEUE_DESC desc = {};
	desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
	desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	desc.NodeMask = 0;

	auto hr = m_pDevice->CreateCommandQueue(&desc, IID_PPV_ARGS(m_pQueue.ReleaseAndGetAddressOf()));


	// コンピュートシェーダー用のコマンドキューを作成
	desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
	desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
	desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

	hr = m_pDevice->CreateCommandQueue(&desc, IID_PPV_ARGS(m_pComputeQueue.ReleaseAndGetAddressOf()));


	return SUCCEEDED(hr);
}

// デバイスを作成
bool Engine::CreateDevice()
{
	auto hr = D3D12CreateDevice(
		nullptr,
		D3D_FEATURE_LEVEL_12_1,
		IID_PPV_ARGS(m_pDevice.ReleaseAndGetAddressOf()
		));

	D3D12_FEATURE_DATA_D3D12_OPTIONS5 features5;
	hr = m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &features5, sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS5));
	if (FAILED(hr) || features5.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED)
	{
		int a = 0;
	}

	return SUCCEEDED(hr);
}

// スワップチェインを作成
bool Engine::CreateSwapChain()
{
	// DXGIファクトリーの生成
	IDXGIFactory4* pFactory = nullptr;
	HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&pFactory));
	if (FAILED(hr))
	{
		return false;
	}

	// スワップチェインの生成
	DXGI_SWAP_CHAIN_DESC desc = {};
	desc.BufferDesc.Width = m_FrameBufferWidth;
	desc.BufferDesc.Height = m_FrameBufferHeight;
	desc.BufferDesc.RefreshRate.Numerator = 60;
	desc.BufferDesc.RefreshRate.Denominator = 1;
	desc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	desc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
	desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	desc.BufferCount = FRAME_BUFFER_COUNT;
	desc.OutputWindow = m_hWnd;
	desc.Windowed = TRUE;
	desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

	// スワップチェインの生成
	IDXGISwapChain* pSwapChain = nullptr;
	hr = pFactory->CreateSwapChain(m_pQueue.Get(), &desc, &pSwapChain);
	if (FAILED(hr))
	{
		pFactory->Release();
		return false;
	}

	// IDXGISwapChain3を取得
	hr = pSwapChain->QueryInterface(IID_PPV_ARGS(m_pSwapChain.ReleaseAndGetAddressOf()));
	if (FAILED(hr))
	{
		pFactory->Release();
		pSwapChain->Release();
		return false;
	}

	// バックバッファ番号を取得
	m_CurrentBackBufferIndex = m_pSwapChain->GetCurrentBackBufferIndex();

	pFactory->Release();
	pSwapChain->Release();
	return true;
}

// コマンドリストを作成(描画命令を格納)
bool Engine::CreateCommandList()
{
	// コマンドアロケーターの作成
	HRESULT hr;
	for (size_t i = 0; i < FRAME_BUFFER_COUNT; i++)
	{
		hr = m_pDevice->CreateCommandAllocator(
			D3D12_COMMAND_LIST_TYPE_DIRECT,
			IID_PPV_ARGS(m_pAllocator[i].ReleaseAndGetAddressOf()));
	}

	if (FAILED(hr))
	{
		return false;
	}

	// コマンドリストの生成
	hr = m_pDevice->CreateCommandList(
		0,
		D3D12_COMMAND_LIST_TYPE_DIRECT,
		m_pAllocator[m_CurrentBackBufferIndex].Get(),
		nullptr,
		IID_PPV_ARGS(&m_pCommandList)
	);

	if (FAILED(hr))
	{
		return false;
	}

	//コマンドリストは開かれている状態で作成されるので、いったん閉じる。
	m_pCommandList->Close();

	return true;
}

// フェンスを作成
bool Engine::CreateFence()
{
	for (auto i = 0u; i < FRAME_BUFFER_COUNT; i++)
	{
		m_fenceValue[i] = 0;
	}

	auto hr = m_pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(m_pFence.ReleaseAndGetAddressOf()));
	if (FAILED(hr))
	{
		return false;
	}

	m_fenceValue[m_CurrentBackBufferIndex]++;

	//同期を行うときのイベントハンドラを作成する。
	m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
	return m_fenceEvent != nullptr;
}

// ビューポートを作成
void Engine::CreateViewPort()
{
	m_Viewport.TopLeftX = 0;
	m_Viewport.TopLeftY = 0;
	m_Viewport.Width = static_cast<float>(m_FrameBufferWidth);
	m_Viewport.Height = static_cast<float>(m_FrameBufferHeight);
	m_Viewport.MinDepth = 0.0f;
	m_Viewport.MaxDepth = 1.0f;
}

// シザー矩形を作成
void Engine::CreateScissorRect()
{
	m_Scissor.left = 0;
	m_Scissor.right = m_FrameBufferWidth;
	m_Scissor.top = 0;
	m_Scissor.bottom = m_FrameBufferHeight;
}

// レンダーターゲットを作成
bool Engine::CreateRenderTarget()
{
	// RTV用のディスクリプタヒープを作成する
	D3D12_DESCRIPTOR_HEAP_DESC desc = {};
	desc.NumDescriptors = FRAME_BUFFER_COUNT;
	desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

	// RTVヒープの作成
	auto hr = m_pDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(m_pRtvHeap.ReleaseAndGetAddressOf()));
	if (FAILED(hr))
	{
		return false;
	}

	// ディスクリプタのサイズを取得
	m_RtvDescriptorSize = m_pDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_pRtvHeap->GetCPUDescriptorHandleForHeapStart();

	// 各バックアップバッファに対してRTVを作成
	for (UINT i = 0; i < FRAME_BUFFER_COUNT; i++)
	{
		m_pSwapChain->GetBuffer(i, IID_PPV_ARGS(m_pRenderTargets[i].ReleaseAndGetAddressOf()));
		m_pDevice->CreateRenderTargetView(m_pRenderTargets[i].Get(), nullptr, rtvHandle);
		rtvHandle.ptr += m_RtvDescriptorSize;
	}

	return true;
}

// 深度ステンシルバッファを作成
bool Engine::CreateDepthStencil()
{
	//DSV用のディスクリプタヒープを作成する
	D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
	heapDesc.NumDescriptors = 1;
	heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
	heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	auto hr = m_pDevice->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_pDsvHeap));
	if (FAILED(hr))
	{
		return false;
	}

	//ディスクリプタのサイズを取得
	m_DsvDescriptorSize = m_pDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);

	D3D12_CLEAR_VALUE dsvClearValue;
	dsvClearValue.Format = DXGI_FORMAT_D32_FLOAT;
	dsvClearValue.DepthStencil.Depth = 1.0f;
	dsvClearValue.DepthStencil.Stencil = 0;

	auto heapProp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
	CD3DX12_RESOURCE_DESC resourceDesc(
		D3D12_RESOURCE_DIMENSION_TEXTURE2D,
		0,
		m_FrameBufferWidth,
		m_FrameBufferHeight,
		1,
		1,
		DXGI_FORMAT_D32_FLOAT,
		1,
		0,
		D3D12_TEXTURE_LAYOUT_UNKNOWN,
		D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE);
	hr = m_pDevice->CreateCommittedResource(
		&heapProp,
		D3D12_HEAP_FLAG_NONE,
		&resourceDesc,
		D3D12_RESOURCE_STATE_DEPTH_WRITE,
		&dsvClearValue,
		IID_PPV_ARGS(m_pDepthStencilBuffer.ReleaseAndGetAddressOf())
	);

	if (FAILED(hr))
	{
		return false;
	}

	//ディスクリプタを作成
	D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = m_pDsvHeap->GetCPUDescriptorHandleForHeapStart();

	m_pDevice->CreateDepthStencilView(m_pDepthStencilBuffer.Get(), nullptr, dsvHandle);

	return true;
}

// 描画開始
void Engine::BeginRender()
{
	// 現在のレンダーターゲットを更新
	m_currentRenderTarget = m_pRenderTargets[m_CurrentBackBufferIndex].Get();

	// コマンドを初期化してためる準備をする（前フレームで使用したのコマンドリストをリセット）
	m_pAllocator[m_CurrentBackBufferIndex]->Reset();
	m_pCommandList->Reset(m_pAllocator[m_CurrentBackBufferIndex].Get(), nullptr);

	// ビューポートとシザー矩形を設定
	m_pCommandList->RSSetViewports(1, &m_Viewport);
	m_pCommandList->RSSetScissorRects(1, &m_Scissor);

	// 現在のフレームのレンダーターゲットビューのディスクリプタヒープの開始アドレスを取得
	auto currentRtvHandle = m_pRtvHeap->GetCPUDescriptorHandleForHeapStart();
	currentRtvHandle.ptr += m_CurrentBackBufferIndex * m_RtvDescriptorSize;

	// 深度ステンシルのディスクリプタヒープの開始アドレス取得
	auto currentDsvHandle = m_pDsvHeap->GetCPUDescriptorHandleForHeapStart();

	// レンダーターゲットが使用可能になるまで待つ
	auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(m_currentRenderTarget, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
	m_pCommandList->ResourceBarrier(1, &barrier);

	// レンダーターゲットを設定
	m_pCommandList->OMSetRenderTargets(1, &currentRtvHandle, FALSE, &currentDsvHandle);

	// レンダーターゲットをクリア
	const float clearColor[] = { 0.25f, 0.25f, 0.25f, 1.0f };
	m_pCommandList->ClearRenderTargetView(currentRtvHandle, clearColor, 0, nullptr);

	// 深度ステンシルビューをクリア
	m_pCommandList->ClearDepthStencilView(currentDsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
}

// 描画待機
void Engine::WaitRender()
{
	//描画終了待ち
	const UINT64 fenceValue = m_fenceValue[m_CurrentBackBufferIndex];
	m_pQueue->Signal(m_pFence.Get(), fenceValue);
	m_fenceValue[m_CurrentBackBufferIndex]++;

	// 次のフレームの描画準備がまだであれば待機する.
	if (m_pFence->GetCompletedValue() < fenceValue)
	{
		// 完了時にイベントを設定.
		auto hr = m_pFence->SetEventOnCompletion(fenceValue, m_fenceEvent);
		if (FAILED(hr))
		{
			return;
		}

		// 待機処理.
		if (WAIT_OBJECT_0 != WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE))
		{
			return;
		}
	}
}

// 描画終了
void Engine::EndRender()
{
	// レンダーターゲットに書き込み終わるまで待つ
	auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(m_currentRenderTarget, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
	m_pCommandList->ResourceBarrier(1, &barrier);

	// コマンドの記録を終了
	m_pCommandList->Close();

	// コマンドを実行
	ID3D12CommandList* ppCmdLists[] = { m_pCommandList.Get() };
	m_pQueue->ExecuteCommandLists(1, ppCmdLists);

	// スワップチェーンを切り替え
	m_pSwapChain->Present(1, 0);

	// 描画完了を待つ
	WaitRender();

	// バックバッファ番号更新
	m_CurrentBackBufferIndex = m_pSwapChain->GetCurrentBackBufferIndex();
}

// GPU完了まで待機
void Engine::Flush()
{
	UINT64 fenceValue = 0;
	m_pQueue->Signal(m_pFence.Get(), ++fenceValue);

	if (m_pFence->GetCompletedValue() < fenceValue)
	{
		m_pFence->SetEventOnCompletion(fenceValue, m_fenceEvent);
		WaitForSingleObject(m_fenceEvent, INFINITE);
	}
}

// デバイス情報を取得用
ID3D12Device6* Engine::Device()
{
	return m_pDevice.Get();
}

// コマンドリスト情報を取得
ID3D12GraphicsCommandList* Engine::CommandList()
{
	return m_pCommandList.Get();
}

// バッファのフレーム番号を取得
UINT Engine::CurrentBackBufferIndex()
{
	return m_CurrentBackBufferIndex;
}

UINT Engine::FrameBufferWidth() const
{
	return m_FrameBufferWidth;
}

UINT Engine::FrameBufferHeight() const
{
	return m_FrameBufferHeight;
}


