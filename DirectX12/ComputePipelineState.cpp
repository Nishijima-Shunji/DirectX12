#include "ComputePipelineState.h"
#include <d3dcompiler.h>
#include "Engine.h"
#include <Windows.h>
#include <filesystem>
#include <vector>

namespace
{
    // 実行ファイル周辺を走査してシェーダーファイルの絶対パスを求める
    std::filesystem::path ResolveShaderPath(const std::wstring& fileName)
    {
        std::vector<std::filesystem::path> searchDirectories;

        // まずは現在の作業ディレクトリ
        searchDirectories.push_back(std::filesystem::current_path());

        // 実行ファイルのディレクトリと親ディレクトリを上へ辿って確認する
        wchar_t exePath[MAX_PATH] = {};
        DWORD length = GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        if (length > 0 && length < MAX_PATH)
        {
            std::filesystem::path dir = std::filesystem::path(exePath).parent_path();
            for (int i = 0; !dir.empty() && i < 5; ++i)
            {
                searchDirectories.push_back(dir);
                dir = dir.parent_path();
            }
        }

        for (const auto& base : searchDirectories)
        {
            std::filesystem::path candidate = base / fileName;
            if (std::filesystem::exists(candidate))
            {
                return candidate;
            }
        }

        return {};
    }
}

// シェーダーの読み込み
void ComputePipelineState::SetCS(const std::wstring& path) {
    std::filesystem::path csoPath = ResolveShaderPath(path);
    HRESULT hr = E_FAIL;
    if (!csoPath.empty())
    {
        hr = D3DReadFileToBlob(csoPath.c_str(), m_csBlob.GetAddressOf());
    }
    if (FAILED(hr)) {
        // Fallback: compile from .hlsl when .cso is missing
        std::wstring hlsl = path;
        size_t pos = hlsl.find_last_of(L'.');
        if (pos != std::wstring::npos) hlsl.replace(pos, std::wstring::npos, L".hlsl");

        std::filesystem::path hlslPath = ResolveShaderPath(hlsl);
        if (hlslPath.empty())
        {
            wprintf(L"CSファイル %ls / %ls が見つかりません\n", path.c_str(), hlsl.c_str());
            return;
        }

        UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
        flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
        ComPtr<ID3DBlob> err;
        hr = D3DCompileFromFile(hlslPath.c_str(), nullptr, nullptr, "CSMain", "cs_5_0", flags, 0,
                               m_csBlob.GetAddressOf(), err.GetAddressOf());
        if (FAILED(hr)) {
            if (err) wprintf(L"CS compile error: %hs\n", (char*)err->GetBufferPointer());
            return;
        }
    }
    desc.CS = CD3DX12_SHADER_BYTECODE(m_csBlob.Get());
}

// 生成
bool ComputePipelineState::Create() {
    if (!m_csBlob) {
        wprintf(L"[Error] Compute shader is not loaded.\n");
        return false;
    }

    // desc.CS は SetCS() で設定済みだが、安全のため再設定しておく
    desc.CS = CD3DX12_SHADER_BYTECODE(m_csBlob.Get());

    HRESULT hr = m_device->CreateComputePipelineState(
        &desc, IID_PPV_ARGS(m_pso.ReleaseAndGetAddressOf()));

    if (FAILED(hr)) {
        wprintf(L"Compute PSO生成失敗 0x%08X\n", hr);
        return false;
    }
    return true;
}