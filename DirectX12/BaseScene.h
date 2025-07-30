#pragma once

class Game;

class BaseScene
{
protected:
    Game* m_game;
public:
    BaseScene(Game* game) : m_game(game) {};
    virtual ~BaseScene() = default;
    virtual bool Init() { return true; }       // 初期化
    virtual void Exit() {}       // 終了時処理
    virtual void Update(float delta) = 0;
    virtual void Draw() = 0;
};

