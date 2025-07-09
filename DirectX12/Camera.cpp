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
	// �s��ϊ�
	eyePos = DirectX::XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);		// ���_�̈ʒu
	targetPos = DirectX::XMVectorSet(0.0f, -1.0f, 0.0f, 0.0f);	// ���_����������W
	upward = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);		// �������\���x�N�g��
	fov = DirectX::XMConvertToRadians(75.0);
	// ����p
	aspect = static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT);		// �A�X�y�N�g��

	return 0;
}

void Camera::Update() {
	// ��]�p�i���[�E�s�b�`�j
	static float yaw = 0.0f;
	static float pitch = 0.0f;
	static bool isDragging = false;
	static POINT lastMouse;

	// �}�E�X�E�{�^���ŉ�]
	if (GetAsyncKeyState(VK_RBUTTON) & 0x8000) {
		POINT curMouse;
		GetCursorPos(&curMouse);

		if (!isDragging) {
			// ���񂾂��}�E�X�ʒu���Z�b�g�i�h���b�O�J�n�j
			lastMouse = curMouse;
			isDragging = true;
		}

		float dx = static_cast<float>(curMouse.x - lastMouse.x) * 0.005f;
		float dy = static_cast<float>(curMouse.y - lastMouse.y) * 0.005f;

		// �}�E�X�̓����ɉ����ă��[�E�s�b�`���X�V�i+�Ŕ��]��ԁj
		yaw -= dx;
		pitch -= dy;

		pitch = std::clamp(pitch, -XM_PIDIV2 + 0.01f, XM_PIDIV2 - 0.01f);

		lastMouse = curMouse;
	}
	else {
		isDragging = false; // �{�^����������h���b�O�I��
	}

	// ��]�s��i���[���s�b�`�j
	XMMATRIX rotMat = XMMatrixRotationRollPitchYaw(pitch, yaw, 0);
	XMVECTOR forward = XMVector3TransformNormal(XMVectorSet(0, 0, -1, 0), rotMat);
	XMVECTOR right = XMVector3TransformNormal(XMVectorSet(1, 0, 0, 0), rotMat);

	// �ړ�
	if (GetAsyncKeyState('W') & 0x8000) eyePos += forward * 0.1f;
	if (GetAsyncKeyState('S') & 0x8000) eyePos -= forward * 0.1f;
	if (GetAsyncKeyState('A') & 0x8000) eyePos -= right * 0.1f;
	if (GetAsyncKeyState('D') & 0x8000) eyePos += right * 0.1f;
	if (GetAsyncKeyState('E') & 0x8000) eyePos += XMVectorSet(0, 1, 0, 0) * 0.1f;
	if (GetAsyncKeyState('Q') & 0x8000) eyePos -= XMVectorSet(0, 1, 0, 0) * 0.1f;

	// �^�[�Q�b�g = �O��
	targetPos = XMVectorAdd(eyePos, forward);

	// �r���[�E�v���W�F�N�V�����X�V
	viewMatrix = XMMatrixLookAtRH(eyePos, targetPos, XMVectorSet(0, 1, 0, 0));
	projMatrix = XMMatrixPerspectiveFovRH(fov, aspect, 0.1f, 1000.0f);

}