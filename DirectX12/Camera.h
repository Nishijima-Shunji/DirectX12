#pragma once
#include <DirectXMath.h>

class Camera
{
private:
	float rotateY = 0.0f;
	float posX = 0.0f;
	float posY = 0.0f;

	DirectX::XMVECTOR eyePos;
	DirectX::XMVECTOR targetPos;
	DirectX::XMVECTOR upward;
	float fov;
	float aspect;

public:
	Camera();
	void Init();
	void Update();

	DirectX::XMVECTOR GetEyePos()		{ return eyePos; };
	DirectX::XMVECTOR GetTargetPos()	{ return targetPos; };
	DirectX::XMVECTOR GetUpward()		{ return upward; };
	float GetFov()						{ return fov; };
	float GetAspect()					{ return aspect; };
};

