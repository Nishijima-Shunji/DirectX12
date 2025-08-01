#pragma once
#include <memory>
#include <map>
#include <string>
#include <stack>
#include <functional>
#include "BaseScene.h"
#include "FadeController.h"

class SceneManager
{
private:
    using SceneFactory = std::function<std::unique_ptr<BaseScene>()>;

    std::map<std::string, SceneFactory> m_Registry;
    std::stack<std::unique_ptr<BaseScene>> m_SceneStack;

    FadeController m_fade;
    std::string m_pendingScene;

public:
    void RegisterScene(const std::string& name, SceneFactory factory);
    void ChangeScene(const std::string& name);
    void PushScene(const std::string& name);
    void PopScene();

    void Update(float deltaTime);
    void Render();
};

