#pragma once
#include "SceneManager.h"

class Game
{
private:
    SceneManager m_SceneManager;
public:
    Game();
    void Update();
    void Render();

    void RegisterScenes(); // ŠeƒV[ƒ““o˜^
    void ChangeScene(const std::string& name);

};

