#include "Camera.h"
#include "App.h"

Camera::Camera() {
	Init();
}

void Camera::Init() {
	// 行列変換
	eyePos = DirectX::XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f);								// 視点の位置
	targetPos = DirectX::XMVectorSet(0.0f, -1.0f, 0.0f, 0.0f);												// 視点を向ける座標
	upward = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);								// 上方向を表すベクトル
	fov = DirectX::XMConvertToRadians(75.0);											// 視野角
	aspect = static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT);		// アスペクト比
}