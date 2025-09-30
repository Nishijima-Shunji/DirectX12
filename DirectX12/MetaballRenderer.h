#pragma once

#include <DirectXMath.h>
#include <memory>
#include <vector>
#include "ComPtr.h"

class ConstantBuffer;
class DescriptorHandle;

// =======================================================================================
// レイマーチングでメタボールを描画するためのヘルパークラス
// =======================================================================================
class MetaballRenderer
{
public:
    bool Initialize();
    void Update(float deltaTime);
    void Render();

private:
    struct ParticleState
    {
        DirectX::XMFLOAT3 position;   // 表示用座標
        float              radius;     // 粒子半径
        float              angleSeed;  // 回転アニメ用シード
        float              radialSeed; // 半径変化用シード
        float              heightSeed; // 高さ変化用シード
    };

    struct alignas(16) ParticleGPU
    {
        DirectX::XMFLOAT4 data; // xyz: 位置, w: 半径
    };

    struct alignas(16) MetaConstants
    {
        DirectX::XMFLOAT4X4 invViewProj; // ビュー射影逆行列
        DirectX::XMFLOAT4X4 viewProj;    // ビュー射影行列
        DirectX::XMFLOAT4   cameraIso;   // xyz: カメラ位置, w: アイソ値
        DirectX::XMFLOAT4   params;      // x: ステップ係数, y: 最大距離, z: 粒子数, w: 時間
    };

    bool CreatePipeline();
    bool CreateBuffers();
    void UpdateConstantBuffer();
    void UpdateParticleBuffer();

    std::vector<ParticleState> m_particles;

    ComPtr<ID3D12RootSignature>  m_rootSignature;
    ComPtr<ID3D12PipelineState>  m_pipelineState;
    std::unique_ptr<ConstantBuffer> m_constantBuffer;

    ComPtr<ID3D12Resource> m_particleBuffer;
    ParticleGPU*            m_mappedParticles = nullptr;
    DescriptorHandle*       m_particleSrv = nullptr;

    float m_time = 0.0f;
};
