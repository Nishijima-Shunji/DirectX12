#include "TitleScene.h"
#include "GameScene.h"
#include "SceneManager.h"
#include "Game.h"

#include <windows.h>

TitleScene::TitleScene(Game* game) : BaseScene(game) {
	printf("TitleScene¶¬\n");
}

TitleScene::~TitleScene() {
	printf("TitleScene”jŠü\n");
}

void TitleScene::Update(float deltaTime) {
	if (GetAsyncKeyState(VK_SPACE)) {
		m_game->ChangeScene("Game");
	}
}

void TitleScene::Draw() {

}