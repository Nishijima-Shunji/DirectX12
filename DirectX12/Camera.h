#pragma once
#include <DirectXMath.h>
#include "Object.h"

class Camera : public Object
{
private:
	float rotateY = 0.0f;
	float posX = 0.0f;
	float posY = 0.0f;
	float posZ = 0.0f;


	DirectX::XMVECTOR eyePos;
	DirectX::XMVECTOR targetPos;
	DirectX::XMVECTOR upward;
	float fov;
	float aspect;
	float yaw = 0.0f;   // 左右回転（Y軸）
	float pitch = 0.0f; // 上下回転（X軸）

	DirectX::XMMATRIX viewMatrix;
	DirectX::XMMATRIX projMatrix;

public:
	Camera();
	bool Init();
	void Update(float deltaTime);

	DirectX::XMVECTOR GetEyePos() { return eyePos; };
	DirectX::XMVECTOR GetTargetPos() { return targetPos; };
	DirectX::XMVECTOR GetUpward() { return upward; };
	float GetFov() { return fov; };
	float GetAspect() { return aspect; };
	DirectX::XMMATRIX GetViewMatrix() const { return viewMatrix; }
	DirectX::XMMATRIX GetProjMatrix() const { return projMatrix; }
	DirectX::XMFLOAT4X4 GetInvViewProj() const;
	DirectX::XMFLOAT3   GetPosition();
};

