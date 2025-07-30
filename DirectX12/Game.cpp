#include <windows.h>
#include "Game.h"
#include "Engine.h"
#include "TitleScene.h"
#include "Scene.h"
#include "GameScene.h"

// =======================================================================================
//      コンストラクタ
// =======================================================================================
Game::Game()
{
    RegisterScenes();                       // 使用するシーンを登録
    m_SceneManager.ChangeScene("Scene");    // 最初のシーン
}

// =======================================================================================
//      メイン処理
// =======================================================================================
void Game::Update(float deltaTime)
{
    m_SceneManager.Update(deltaTime);
}

// =======================================================================================
//      描画処理
// =======================================================================================
void Game::Render()
{
    //g_Engine->BeginRender();
    m_SceneManager.Render();
    //g_Engine->EndRender();
}

// =======================================================================================
//      シーンの登録呼び出し
// =======================================================================================
void Game::RegisterScenes()
{
    
    m_SceneManager.RegisterScene("Scene", [this]() {
        return std::make_unique<Scene>(this); // Game*渡す
        });

    m_SceneManager.RegisterScene("TitleScene", [this]() {
        return std::make_unique<TitleScene>(this); // Game*渡す
        });

    m_SceneManager.RegisterScene("Game", [this]() {
        return std::make_unique<GameScene>(this); // 同様に
        });
}

// =======================================================================================
//      シーンの変更呼び出し
// =======================================================================================
void Game::ChangeScene(const std::string& name)
{
    m_SceneManager.ChangeScene(name);
}