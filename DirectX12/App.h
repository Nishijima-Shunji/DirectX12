#pragma once
#include <Windows.h>

const UINT WINDOW_WIDTH = 1920 / 2;
const UINT WINDOW_HEIGHT = 1080 / 2;

extern HWND g_hWnd;
void StartApp(const TCHAR* appName); // これを呼んだらアプリが実行するようにする
