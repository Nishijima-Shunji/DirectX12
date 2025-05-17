// Game.cpp
#include <windows.h>
#include "Game.h"
#include "Engine.h"
#include "TitleScene.h"
#include "Scene.h"
#include "GameScene.h"

// =======================================================================================
//      �R���X�g���N�^
// =======================================================================================
Game::Game()
{
    RegisterScenes();                       // �g�p����V�[����o�^
    m_SceneManager.ChangeScene("Scene");    // �ŏ��̃V�[��
}

// =======================================================================================
//      ���C������
// =======================================================================================
void Game::Update()
{
    m_SceneManager.Update();
}

// =======================================================================================
//      �`�揈��
// =======================================================================================
void Game::Render()
{
    g_Engine->BeginRender();
    m_SceneManager.Render();
    g_Engine->EndRender();
}

// =======================================================================================
//      �V�[���̓o�^�Ăяo��
// =======================================================================================
void Game::RegisterScenes()
{
    
    m_SceneManager.RegisterScene("Scene", [this]() {
        return std::make_unique<Scene>(this); // Game*�n��
        });

    m_SceneManager.RegisterScene("TitleScene", [this]() {
        return std::make_unique<TitleScene>(this); // Game*�n��
        });

    m_SceneManager.RegisterScene("Game", [this]() {
        return std::make_unique<GameScene>(this); // ���l��
        });
}

// =======================================================================================
//      �V�[���̕ύX�Ăяo��
// =======================================================================================
void Game::ChangeScene(const std::string& name)
{
    m_SceneManager.ChangeScene(name);
}