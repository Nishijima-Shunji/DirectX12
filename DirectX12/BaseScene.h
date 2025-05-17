#pragma once

class Game;

class BaseScene
{
protected:
    Game* m_game;
public:
    BaseScene(Game* game) : m_game(game) {};
    virtual ~BaseScene() = default;
    virtual bool Init() { return true; }       // ������
    virtual void Exit() {}       // �I��������
    virtual void Update() = 0;
    virtual void Draw() = 0;
};

