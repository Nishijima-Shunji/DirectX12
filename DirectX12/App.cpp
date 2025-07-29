#include "App.h"
#include "Engine.h"
#include "Scene.h"
#include "Game.h"
#include <dxgidebug.h>
#include <chrono>
#include <string>

#pragma comment(lib, "dxguid.lib")


HINSTANCE g_hInst;
HWND g_hWnd = NULL;

LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wp, LPARAM lp)
{
	switch (msg)
	{
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		break;
	}

	return DefWindowProc(hWnd, msg, wp, lp);
}

void InitWindow(const TCHAR* appName)
{
	g_hInst = GetModuleHandle(nullptr);
	if (g_hInst == nullptr)
	{
		return;
	}

	// ウィンドウの設定
	WNDCLASSEX wc = {};
	wc.cbSize = sizeof(WNDCLASSEX);
	wc.style = CS_HREDRAW | CS_VREDRAW;
	wc.lpfnWndProc = WndProc;
	wc.hIcon = LoadIcon(g_hInst, IDI_APPLICATION);
	wc.hCursor = LoadCursor(g_hInst, IDC_ARROW);
	wc.hbrBackground = GetSysColorBrush(COLOR_BACKGROUND);
	wc.lpszMenuName = nullptr;
	wc.lpszClassName = appName;
	wc.hIconSm = LoadIcon(g_hInst, IDI_APPLICATION);

	// ウィンドウクラスの登録。
	RegisterClassEx(&wc);

	// ウィンドウサイズの設定
	RECT rect = {};
	rect.right = static_cast<LONG>(WINDOW_WIDTH);
	rect.bottom = static_cast<LONG>(WINDOW_HEIGHT);

	// ウィンドウサイズを調整
	auto style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU;
	AdjustWindowRect(&rect, style, FALSE);

	// ウィンドウの生成
	g_hWnd = CreateWindowEx(
		0,
		appName,
		appName,
		style,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		rect.right - rect.left,
		rect.bottom - rect.top,
		nullptr,
		nullptr,
		g_hInst,
		nullptr
	);

	// ウィンドウを表示
	ShowWindow(g_hWnd, SW_SHOWNORMAL);

	// ウィンドウにフォーカスする
	SetFocus(g_hWnd);
}

void MainLoop()
{
	MSG msg = {};
	Game game;

	using clock = std::chrono::high_resolution_clock;
	auto prevTime = clock::now();
	int  frameCount = 0;
	float elapsed = 0.0f;

	while (WM_QUIT != msg.message)
	{
		if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else
		{
			// デルタタイム計算
			auto now = clock::now();
			float dt = std::chrono::duration<float>(now - prevTime).count();
			prevTime = now;

			// ゲーム更新・描画
			g_Engine->BeginRender();
			game.Update(dt);
			game.Render();
			g_Engine->EndRender();


			//// フレームを提示
			//g_Engine->EndRender();
			//g_Engine->m_pSwapChain->Present(1, 0);

			// ─── FPS 計測・表示 ─────────
			frameCount++;
			elapsed += dt;
			if (elapsed >= 1.0f)
			{
				// ウィンドウタイトルに FPS を表示
				wchar_t buf[64];
				swprintf(buf, 64, L"FPS: %d", frameCount);
				SetWindowText(g_hWnd, buf);
				frameCount = 0;
				elapsed = 0.0f;
			}
		}
	}
}

void StartApp(const TCHAR* appName)
{
	// ウィンドウ生成
	InitWindow(appName);

	g_Engine = new Engine();
	if (!g_Engine->Init(g_hWnd, WINDOW_WIDTH, WINDOW_HEIGHT))
	{
		return;
	}
	// メイン処理ループ
	MainLoop();
	IDXGIDebug1* pDebug = nullptr;
	if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&pDebug)))) {
		pDebug->ReportLiveObjects(DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_DETAIL);
		pDebug->Release();
	}
}

