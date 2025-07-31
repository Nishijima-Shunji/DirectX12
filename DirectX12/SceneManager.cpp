#include "SceneManager.h"
#include "Scene.h"

void SceneManager::RegisterScene(const std::string& name, SceneFactory factory)
{
    m_Registry[name] = factory;
}

void SceneManager::ChangeScene(const std::string& name)
{
    if (m_SceneStack.empty()) {
        auto it = m_Registry.find(name);
        if (it != m_Registry.end()) {
            m_SceneStack.push(it->second());
        }
    } else {
        m_pendingScene = name;
        m_fade.StartFadeOut();
    }
}

void SceneManager::PushScene(const std::string& name)
{
    auto it = m_Registry.find(name);
    if (it != m_Registry.end()) {
        m_SceneStack.push(it->second());
    }
}

void SceneManager::PopScene()
{
    if (!m_SceneStack.empty()) {
        m_SceneStack.pop();
    }
}

void SceneManager::Update(float deltaTime)
{
    m_fade.Update(deltaTime);

    if (!m_SceneStack.empty()) {
        m_SceneStack.top()->Update(deltaTime);
    }

    if (!m_pendingScene.empty() && m_fade.IsFadeOutComplete()) {
        auto it = m_Registry.find(m_pendingScene);
        if (it != m_Registry.end()) {
            if (!m_SceneStack.empty()) {
                m_SceneStack.pop();
            }
            m_SceneStack.push(it->second());
        }
        m_pendingScene.clear();
        m_fade.StartFadeIn();
    }
}

void SceneManager::Render()
{
    if (!m_SceneStack.empty()) {
        m_SceneStack.top()->Draw();
    }
    m_fade.Render();
}

