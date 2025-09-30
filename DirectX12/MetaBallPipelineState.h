#pragma once

#include "ComPtr.h"
#include <d3d12.h>
#include <wrl.h>

//using Microsoft::WRL::ComPtr;

namespace graphics {

    struct MetaBallPipeline
    {
        // ルートシグネチャ生成（失敗時は false を返す）
        static bool CreateRootSignature(ID3D12Device* device,
            ComPtr<ID3D12RootSignature>& outRootSig);

        static bool CreatePipelineState(ID3D12Device* device,
            ID3D12RootSignature* rootSig,
            DXGI_FORMAT rtvFormat,
            DXGI_FORMAT dsvFormat,
            ComPtr<ID3D12PipelineState>& outPSO);
    };

} // namespace graphics
