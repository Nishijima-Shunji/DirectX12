#include "SceneManager.h"
#include "Scene.h"

void SceneManager::RegisterScene(const std::string& name, SceneFactory factory)
{
    m_Registry[name] = factory;
}

std::unique_ptr<BaseScene> SceneManager::CreateInitializedScene(const std::string& name)
{
    auto it = m_Registry.find(name);
    if (it == m_Registry.end())
    {
        return nullptr;
    }

    auto scene = it->second();
    if (!scene)
    {
        return nullptr;
    }

    // 生成直後に必ず初期化を実行し、失敗した場合は無効として扱う
    if (!scene->Init())
    {
        return nullptr;
    }

    return scene;
}

void SceneManager::ChangeScene(const std::string& name)
{
    if (m_SceneStack.empty()) {
        auto scene = CreateInitializedScene(name);
        if (scene)
        {
            m_SceneStack.push(std::move(scene));
        }
    } else {
        m_pendingScene = name;
        m_fade.StartFadeOut();
    }
}

void SceneManager::PushScene(const std::string& name)
{
    auto scene = CreateInitializedScene(name);
    if (scene)
    {
        m_SceneStack.push(std::move(scene));
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
        auto newScene = CreateInitializedScene(m_pendingScene);
        if (newScene)
        {
            if (!m_SceneStack.empty()) {
                m_SceneStack.pop();
            }
            m_SceneStack.push(std::move(newScene));
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

