	#include "VertexBuffer.h"
	#include "Engine.h"
	#include <d3dx12.h>

	VertexBuffer::VertexBuffer(size_t size, size_t stride, const void* pInitData)
	{

		auto prop = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD); 	// ヒーププロパティ
		auto desc = CD3DX12_RESOURCE_DESC::Buffer(size); 				// リソースの設定

		// リソースを生成
		auto hr = g_Engine->Device()->CreateCommittedResource(
			&prop,
			D3D12_HEAP_FLAG_NONE,
			&desc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(m_pBuffer.GetAddressOf()));

		if (FAILED(hr))
		{
			printf("頂点バッファリソースの生成に失敗");
			return;
		}

		// 頂点バッファビューの設定
		m_View.BufferLocation = m_pBuffer->GetGPUVirtualAddress();
		m_View.SizeInBytes = static_cast<UINT>(size);
		m_View.StrideInBytes = static_cast<UINT>(stride);

		// マッピングする
		if (pInitData != nullptr)
		{
			void* ptr = nullptr;
			hr = m_pBuffer->Map(0, nullptr, &ptr);
			if (FAILED(hr))
			{
				printf("頂点バッファマッピングに失敗");
				return;
			}

			// 頂点データをマッピング先に設定
			memcpy(ptr, pInitData, size);

			// マッピング解除
			m_pBuffer->Unmap(0, nullptr);
		}

		m_IsValid = true;
	}

	D3D12_VERTEX_BUFFER_VIEW VertexBuffer::View() const
	{
		return m_View;
	}

	bool VertexBuffer::IsValid()
	{
		return m_IsValid;
	}

	ID3D12Resource* VertexBuffer::GetResource() const
	{
		return m_pBuffer.Get();
	}
