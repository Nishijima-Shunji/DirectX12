#pragma once

#include "Comptr.h"
#include <d3d12.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

namespace graphics {

    struct MetaBallPipeline
    {
        // èâä˙âª
        static void CreateRootSignature(ID3D12Device* device,
            ComPtr<ID3D12RootSignature>& outRootSig);

        static void CreatePipelineState(ID3D12Device* device,
            ID3D12RootSignature* rootSig,
            DXGI_FORMAT rtvFormat,
            ComPtr<ID3D12PipelineState>& outPSO);
    };

} // namespace graphics
