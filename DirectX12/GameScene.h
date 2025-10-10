#pragma once
#include "BaseScene.h"
#include "FluidSystem.h"
#include <DirectXMath.h>
#include <memory>

class Camera;
class DebugCube;
namespace GameSceneDetail { class TransparentWalls; }

// ゲームシーン。流体・壁・デバッグオブジェクトの管理のみを行う。
class GameScene : public BaseScene
{
public:
    explicit GameScene(class Game* game);
    ~GameScene() override;

    bool Init() override;
    void Update(float deltaTime) override;
    void Draw() override;

private:
    std::unique_ptr<FluidSystem> m_fluid;        // 流体本体
    std::unique_ptr<DebugCube> m_debugCube;      // デバッグ用キューブ

    std::unique_ptr<GameSceneDetail::TransparentWalls> m_walls; // 透明な壁描画用

    FluidSystem::Bounds m_initialBounds{}; // 初期境界情報
    float m_wallMoveSpeed = 1.5f;          // 壁移動速度
    void HandleWallControl(Camera& camera, float deltaTime);
    void HandleCameraLift(Camera& camera, float deltaTime); // カメラから粒子を巻き上げる入力を監視
    bool RecreateFluid(); // 流体システムを再生成
};
