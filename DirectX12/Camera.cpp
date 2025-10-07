#include "Camera.h"
#include "App.h"
#include "Engine.h"
#include "SharedStruct.h"
#include <algorithm>

using namespace DirectX;

Camera::Camera()
{
        Init();
}

bool Camera::Init()
{
        // 行列初期化（視点と向き）
        eyePos = DirectX::XMVectorSet(0.0f, 0.0f, 2.0f, 0.0f);          // 視点の位置
        targetPos = DirectX::XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);       // 注視点
        upward = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);          // 上方向ベクトル
        fov = DirectX::XMConvertToRadians(75.0);
        aspect = static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT); // アスペクト比
        return 0;
}

void Camera::Update(float /*deltaTime*/)
{
        // カメラ回転（ヨー・ピッチ）は右ドラッグで制御
        static float yaw = 0.0f;
        static float pitch = 0.0f;
        static bool isDragging = false;
        static POINT lastMouse;

        bool rightPressed = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
        if (rightPressed) {
                POINT curMouse;
                GetCursorPos(&curMouse);
                if (!isDragging) {
                        lastMouse = curMouse;
                        isDragging = true;
                }

                float dx = static_cast<float>(curMouse.x - lastMouse.x) * 0.005f;
                float dy = static_cast<float>(curMouse.y - lastMouse.y) * 0.005f;

                yaw -= dx;
                pitch -= dy;
                pitch = std::clamp(pitch, -XM_PIDIV2 + 0.01f, XM_PIDIV2 - 0.01f);

                lastMouse = curMouse;
        } else {
                isDragging = false;
        }

        // 回転行列から前方・右方向ベクトルを算出
        XMMATRIX rotMat = XMMatrixRotationRollPitchYaw(pitch, yaw, 0);
        XMVECTOR forward = XMVector3TransformNormal(XMVectorSet(0, 0, -1, 0), rotMat);
        XMVECTOR right = XMVector3TransformNormal(XMVectorSet(1, 0, 0, 0), rotMat);

        // 右クリック中のみ WASDQE でカメラ移動
        if (rightPressed) {
                if (GetAsyncKeyState('W') & 0x8000) eyePos += forward * 0.05f;
                if (GetAsyncKeyState('S') & 0x8000) eyePos -= forward * 0.05f;
                if (GetAsyncKeyState('A') & 0x8000) eyePos -= right * 0.05f;
                if (GetAsyncKeyState('D') & 0x8000) eyePos += right * 0.05f;
                if (GetAsyncKeyState('E') & 0x8000) eyePos += XMVectorSet(0, 1, 0, 0) * 0.05f;
                if (GetAsyncKeyState('Q') & 0x8000) eyePos -= XMVectorSet(0, 1, 0, 0) * 0.05f;
        }

        // 注視点更新
        targetPos = XMVectorAdd(eyePos, forward);

        // ビュー・プロジェクション更新
        viewMatrix = XMMatrixLookAtRH(eyePos, targetPos, XMVectorSet(0, 1, 0, 0));
        projMatrix = XMMatrixPerspectiveFovRH(fov, aspect, 0.1f, 1000.0f);
}

DirectX::XMFLOAT4X4 Camera::GetInvViewProj() const
{
        XMMATRIX view = GetViewMatrix();
        XMMATRIX proj = GetProjMatrix();
        XMMATRIX viewProj = XMMatrixMultiply(view, proj);

        XMVECTOR det;
        XMMATRIX invVP = XMMatrixInverse(&det, viewProj);

        DirectX::XMFLOAT4X4 out;
        XMStoreFloat4x4(&out, invVP);
        return out;
}

DirectX::XMFLOAT3 Camera::GetPosition()
{
        XMVECTOR eye = GetEyePos();
        DirectX::XMFLOAT3 pos;
        XMStoreFloat3(&pos, eye);
        return pos;
}
