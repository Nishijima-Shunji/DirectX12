#include "MetaBallPipelineState.h"
#include <d3dcompiler.h>
#include "d3dx12.h"
#include <Windows.h>
#include <filesystem>
#include <vector>
#include <string>
#include <cstdio>

namespace graphics {

    namespace
    {
        // 実行ファイルのあるディレクトリを取得する
        std::filesystem::path GetExecutableDirectory()
        {
            wchar_t path[MAX_PATH] = {};
            DWORD length = GetModuleFileNameW(nullptr, path, MAX_PATH);
            if (length == 0 || length == MAX_PATH)
            {
                return {};
            }
            return std::filesystem::path(path).parent_path();
        }

        // シェーダーファイルを探索して最初に見つかったパスを返す
        std::wstring ResolveShaderPath(const std::wstring& fileName)
        {
            std::vector<std::filesystem::path> searchDirectories;

            // まずは現在の作業ディレクトリ
            searchDirectories.push_back(std::filesystem::current_path());

            // さらに実行ファイルのディレクトリとその親ディレクトリを順番に確認する
            std::filesystem::path exeDir = GetExecutableDirectory();
            for (int i = 0; !exeDir.empty() && i < 5; ++i)
            {
                searchDirectories.push_back(exeDir);
                exeDir = exeDir.parent_path();
            }

            for (const auto& base : searchDirectories)
            {
                std::filesystem::path candidate = base / fileName;
                if (std::filesystem::exists(candidate))
                {
                    return candidate.wstring();
                }
            }

            return L"";
        }

        // CSO 読み込み→失敗時 HLSL コンパイルの順でシェーダーを確保する
        bool LoadOrCompileShader(
            const std::wstring& csoName,
            const std::wstring& hlslName,
            const char* entryPoint,
            const char* shaderModel,
            ComPtr<ID3DBlob>& outBlob)
        {
            std::wstring csoPath = ResolveShaderPath(csoName);
            if (!csoPath.empty())
            {
                HRESULT hr = D3DReadFileToBlob(csoPath.c_str(), &outBlob);
                if (SUCCEEDED(hr))
                {
                    return true;
                }
            }

            std::wstring hlslPath = ResolveShaderPath(hlslName);
            if (hlslPath.empty())
            {
                wprintf(L"FluidSystem: シェーダーファイル %ls が見つかりません\n", hlslName.c_str());
                return false;
            }

            UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
            flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

            ComPtr<ID3DBlob> err;
            HRESULT hr = D3DCompileFromFile(
                hlslPath.c_str(),
                nullptr,
                nullptr,
                entryPoint,
                shaderModel,
                flags,
                0,
                &outBlob,
                &err);

            if (FAILED(hr))
            {
                if (err)
                {
                    printf("FluidSystem: シェーダーコンパイルに失敗しました -> %s\n", static_cast<const char*>(err->GetBufferPointer()));
                }
                else
                {
                    wprintf(L"FluidSystem: シェーダー %ls のコンパイルに失敗しました (HRESULT=0x%08X)\n", hlslPath.c_str(), hr);
                }
                return false;
            }

            return true;
        }
    }

    void MetaBallPipeline::CreateRootSignature(
        ID3D12Device* device,
        ComPtr<ID3D12RootSignature>& outRootSig)
    {
        // SRV(t0) + CBV(b0)
        CD3DX12_DESCRIPTOR_RANGE ranges[1];
        ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);

        CD3DX12_ROOT_PARAMETER params[2];
        params[0].InitAsDescriptorTable(1, &ranges[0], D3D12_SHADER_VISIBILITY_PIXEL);
        params[1].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_PIXEL);

        CD3DX12_ROOT_SIGNATURE_DESC desc(
            _countof(params), params,
            0, nullptr,
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
        );

        ComPtr<ID3DBlob> blob, error;
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error);
        device->CreateRootSignature(
            0,
            blob->GetBufferPointer(),
            blob->GetBufferSize(),
            IID_PPV_ARGS(&outRootSig)
        );
    }

    void MetaBallPipeline::CreatePipelineState(
        ID3D12Device* device,
        ID3D12RootSignature* rootSig,
        DXGI_FORMAT rtvFormat,
        ComPtr<ID3D12PipelineState>& outPSO)
    {
        // メタボール描画用シェーダーを読み込み（失敗時はその場でコンパイルを試みる）
        ComPtr<ID3DBlob> vsBlob;
        if (!LoadOrCompileShader(L"MetaBallVS.cso", L"MetaBallVS.hlsl", "main", "vs_5_0", vsBlob))
        {
            return;
        }

        ComPtr<ID3DBlob> psBlob;
        if (!LoadOrCompileShader(L"MetaBallPS.cso", L"MetaBallPS.hlsl", "main", "ps_5_0", psBlob))
        {
            return;
        }

        // PSO 設定
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = rootSig;
        desc.VS = CD3DX12_SHADER_BYTECODE(vsBlob.Get());
        desc.PS = CD3DX12_SHADER_BYTECODE(psBlob.Get());
        desc.InputLayout = { nullptr, 0 };
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = rtvFormat;
        desc.SampleMask = UINT_MAX;
        desc.SampleDesc.Count = 1;
        desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        desc.BlendState.RenderTarget[0].BlendEnable = TRUE; // 水らしい半透明表現のためにアルファブレンドを有効化
        desc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
        desc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        desc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
        desc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
        desc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
        desc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        desc.DepthStencilState.DepthEnable = FALSE; // スクリーンスペース合成なのでデプスは参照しない

        device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&outPSO));
    }

} // namespace graphics
