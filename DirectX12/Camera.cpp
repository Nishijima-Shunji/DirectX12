#include "Camera.h"
#include "App.h"
#include "Engine.h"
#include "SharedStruct.h"
#include <algorithm>


using namespace DirectX;

Camera::Camera() {
	Init();
}

bool Camera::Init() {
	// 行列変換
	eyePos = DirectX::XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);		// 視点の位置
	targetPos = DirectX::XMVectorSet(0.0f, -1.0f, 0.0f, 0.0f);	// 視点を向ける座標
	upward = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);		// 上方向を表すベクトル
	fov = DirectX::XMConvertToRadians(75.0);
	// 視野角
	aspect = static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT);		// アスペクト比

	return 0;
}

void Camera::Update() {
	// 回転角（ヨー・ピッチ）
	static float yaw = 0.0f;
	static float pitch = 0.0f;
	static bool isDragging = false;
	static POINT lastMouse;

	// マウス右ボタンで回転
	if (GetAsyncKeyState(VK_RBUTTON) & 0x8000) {
		POINT curMouse;
		GetCursorPos(&curMouse);

		if (!isDragging) {
			// 初回だけマウス位置をセット（ドラッグ開始）
			lastMouse = curMouse;
			isDragging = true;
		}

		float dx = static_cast<float>(curMouse.x - lastMouse.x) * 0.005f;
		float dy = static_cast<float>(curMouse.y - lastMouse.y) * 0.005f;

		// マウスの動きに応じてヨー・ピッチを更新（+で反転状態）
		yaw -= dx;
		pitch -= dy;

		pitch = std::clamp(pitch, -XM_PIDIV2 + 0.01f, XM_PIDIV2 - 0.01f);

		lastMouse = curMouse;
	}
	else {
		isDragging = false; // ボタン離したらドラッグ終了
	}

	// 回転行列（ヨー→ピッチ）
	XMMATRIX rotMat = XMMatrixRotationRollPitchYaw(pitch, yaw, 0);
	XMVECTOR forward = XMVector3TransformNormal(XMVectorSet(0, 0, -1, 0), rotMat);
	XMVECTOR right = XMVector3TransformNormal(XMVectorSet(1, 0, 0, 0), rotMat);

	// 移動
	if (GetAsyncKeyState('W') & 0x8000) eyePos += forward * 0.1f;
	if (GetAsyncKeyState('S') & 0x8000) eyePos -= forward * 0.1f;
	if (GetAsyncKeyState('A') & 0x8000) eyePos -= right * 0.1f;
	if (GetAsyncKeyState('D') & 0x8000) eyePos += right * 0.1f;
	if (GetAsyncKeyState('E') & 0x8000) eyePos += XMVectorSet(0, 1, 0, 0) * 0.1f;
	if (GetAsyncKeyState('Q') & 0x8000) eyePos -= XMVectorSet(0, 1, 0, 0) * 0.1f;

	// ターゲット = 前方
	targetPos = XMVectorAdd(eyePos, forward);

	// ビュー・プロジェクション更新
	viewMatrix = XMMatrixLookAtRH(eyePos, targetPos, XMVectorSet(0, 1, 0, 0));
	projMatrix = XMMatrixPerspectiveFovRH(fov, aspect, 0.1f, 1000.0f);

}