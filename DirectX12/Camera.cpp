#include "Camera.h"
#include "App.h"
#include "Engine.h"
#include "SharedStruct.h"

using namespace DirectX;

Camera::Camera() {
	Init();
}

bool Camera::Init() {
	// 行列変換
	eyePos = DirectX::XMVectorSet(0.0f, 0.0f, -20.0f, 0.0f);							// 視点の位置
	targetPos = DirectX::XMVectorSet(0.0f, -1.0f, 0.0f, 0.0f);							// 視点を向ける座標
	upward = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);								// 上方向を表すベクトル
	fov = DirectX::XMConvertToRadians(75.0);
	// 視野角
	aspect = static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT);		// アスペクト比

	return 0;
}

void Camera::Update() {
    if (GetAsyncKeyState('A')) posX -= 0.1f;
    if (GetAsyncKeyState('D')) posX += 0.1f;
    if (GetAsyncKeyState('W')) posZ -= 0.1f;
    if (GetAsyncKeyState('S')) posZ += 0.1f;
    if (GetAsyncKeyState('Q')) posY -= 0.1f;
    if (GetAsyncKeyState('E')) posY += 0.1f;

    if (GetAsyncKeyState(VK_LEFT))  rotateY -= 0.02f;
    if (GetAsyncKeyState(VK_RIGHT)) rotateY += 0.02f;

    eyePos = XMVectorSet(posX, posY, posZ, 1.0f);

    XMMATRIX rotMat = XMMatrixRotationY(rotateY);
    XMVECTOR forward = XMVector3TransformNormal(XMVectorSet(0, 0, -1, 0), rotMat);
    targetPos = XMVectorAdd(eyePos, forward);

    upward = XMVectorSet(0, 1, 0, 0);
    viewMatrix = XMMatrixLookAtRH(eyePos, targetPos, upward);
    projMatrix = XMMatrixPerspectiveFovRH(fov, aspect, 0.1f, 1000.0f);
}