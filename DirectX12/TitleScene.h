#pragma once
#include "BaseScene.h"

class TitleScene : public BaseScene
{
private:

public:
	TitleScene(Game* game);
	~TitleScene();
	void Update(float deltaTime) override;
	void Draw() override;
};

