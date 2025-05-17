#pragma once
#include "BaseScene.h"

class TitleScene : public BaseScene
{
private:

public:
	TitleScene(Game* game);
	~TitleScene();
	void Update() override;
	void Draw() override;
};

