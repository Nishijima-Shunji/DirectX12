#pragma once
#include "SceneManager.h"
#include "Camera.h"


class Game
{
private:
    SceneManager m_SceneManager;
    Camera m_camera;
public:
    Game();
    void Update(float deltaTime);
    void Render();
    
    Camera* GetCamera() { return &m_camera; }

    void RegisterScenes(); // ŠeƒV[ƒ““o˜^
    void ChangeScene(const std::string& name);

};

