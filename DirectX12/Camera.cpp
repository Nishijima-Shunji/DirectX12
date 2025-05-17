#include "Camera.h"
#include "App.h"

Camera::Camera() {
	Init();
}

void Camera::Init() {
	// �s��ϊ�
	eyePos = DirectX::XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f);								// ���_�̈ʒu
	targetPos = DirectX::XMVectorSet(0.0f, -1.0f, 0.0f, 0.0f);												// ���_����������W
	upward = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);								// �������\���x�N�g��
	fov = DirectX::XMConvertToRadians(75.0);											// ����p
	aspect = static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT);		// �A�X�y�N�g��
}