#pragma once
#include "VertexBuffer.h"
#include "RootSignature.h"
#include "ParticlePipelineState.h"
#include "ConstantBuffer.h"
#include <string>
#include <memory>
#include <DirectXMath.h>
#include "Engine.h"

class FadeController
{
public:
    FadeController();
    void StartFadeOut(float duration = 1.0f);
    void StartFadeIn(float duration = 1.0f);
    void Update(float dt);
    void Render();

    bool IsFading() const { return m_active; }
    bool IsFadeOutComplete() const { return !m_active && m_fadeOut && m_alpha >= 1.0f; }

private:
    bool Init();

    std::unique_ptr<VertexBuffer> m_vertexBuffer;
    std::unique_ptr<RootSignature> m_rootSignature;
    std::unique_ptr<ParticlePipelineState> m_pipelineState;
    std::unique_ptr<ConstantBuffer> m_constantBuffers[Engine::FRAME_BUFFER_COUNT];

    float m_alpha = 0.0f;
    bool  m_active = false;
    bool  m_fadeOut = false;
    float m_duration = 1.0f;
    float m_timer = 0.0f;
};

