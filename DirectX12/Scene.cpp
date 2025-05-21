#include <Windows.h>

#include "Scene.h"
#include "Engine.h"
#include "App.h"
#include "Game.h"
#include <d3dx12.h>
#include "SharedStruct.h"
#include "VertexBuffer.h"
#include "ConstantBuffer.h"
#include "RootSignature.h"
#include "PipelineState.h"
#include "IndexBuffer.h"
#include "AssimpLoader.h"
#include "DescriptorHeap.h"
#include "Texture2D.h"

#include <filesystem>



Scene* g_Scene;
using namespace DirectX;
VertexBuffer* vertexBuffer;
ConstantBuffer* constantBuffer[Engine::FRAME_BUFFER_COUNT];
RootSignature* rootSignature;
PipelineState* pipelineState;
IndexBuffer* indexBuffer;
const wchar_t* modelFile = L"assets/korosuke.fbx";

std::vector<Mesh> meshes;					// メッシュの配列
std::vector<VertexBuffer*> vertexBuffers;	// メッシュの数分の頂点バッファ
std::vector<IndexBuffer*> indexBuffers;		// メッシュの数分のインデックスバッファ

namespace fs = std::filesystem;

Scene::Scene(Game* game) : BaseScene(game) {
	Init();
}

std::wstring ReplaceExtension(const std::wstring& origin, const char* ext)
{
	fs::path p = origin.c_str();
	return p.replace_extension(ext).c_str();
}

DescriptorHeap* descriptorHeap;
std::vector< DescriptorHandle*> materialHandles; // テクスチャ用のハンドル一覧

bool Scene::Init()
{
	ImportSettings importSetting =				// 自作の読み込み設定構造体
	{
		modelFile,
		meshes,
		false,
		true // アリシアのモデルは、テクスチャのUVのVだけ反転してるっぽい？ので読み込み時にUV座標を逆転させる
	};

	AssimpLoader loader;
	if (!loader.Load(importSetting))
	{
		return false;
	}

	// メッシュの数だけ頂点バッファを用意する
	vertexBuffers.reserve(meshes.size());
	for (size_t i = 0; i < meshes.size(); i++)
	{
		auto size = sizeof(Vertex) * meshes[i].Vertices.size();
		auto stride = sizeof(Vertex);
		auto vertices = meshes[i].Vertices.data();
		auto pVB = new VertexBuffer(size, stride, vertices);
		if (!pVB->IsValid())
		{
			printf("頂点バッファの生成に失敗\n");
			return false;
		}

		vertexBuffers.push_back(pVB);
	}

	// メッシュの数だけインデックスバッファを用意する
	indexBuffers.reserve(meshes.size());
	for (size_t i = 0; i < meshes.size(); i++)
	{
		auto size = sizeof(uint32_t) * meshes[i].Indices.size();
		auto indices = meshes[i].Indices.data();
		auto pIB = new IndexBuffer(size, indices);
		if (!pIB->IsValid())
		{
			printf("インデックスバッファの生成に失敗\n");
			return false;
		}

		indexBuffers.push_back(pIB);
	}

	// 行列変換
	eyePos = XMVectorSet(0.0f, 0.0f, 5.0f, 0.0f);										// 視点の位置
	targetPos = XMVectorZero();															// 視点を向ける座標
	upward = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);										// 上方向を表すベクトル
	fov = XMConvertToRadians(37.5);														// 視野角
	auto aspect = static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT); // アスペクト比

	for (size_t i = 0; i < Engine::FRAME_BUFFER_COUNT; i++)
	{
		constantBuffer[i] = new ConstantBuffer(sizeof(Transform));
		if (!constantBuffer[i]->IsValid())
		{
			printf("変換行列用定数バッファの生成に失敗\n");
			return false;
		}

		// 変換行列の登録
		auto ptr = constantBuffer[i]->GetPtr<Transform>();
		ptr->World = XMMatrixIdentity();
		ptr->View = XMMatrixLookAtRH(eyePos, targetPos, upward);
		ptr->Proj = XMMatrixPerspectiveFovRH(fov, aspect, 0.3f, 1000.0f);
	}

	// マテリアルの読み込み
	materialHandles.clear();
	descriptorHeap = new DescriptorHeap();
	for (size_t i = 0; i < meshes.size(); i++)
	{
		// テクスチャファイルパスの生成（TGA形式に変換）
		auto texPath = ReplaceExtension(meshes[i].DiffuseMap, "png");

		// テクスチャを読み込む
		auto mainTex = Texture2D::Get(texPath);

		// テクスチャが見つからない場合、デフォルトのテクスチャを使用
		if (!mainTex) {
			printf("テクスチャの読み込みに失敗しました: %ws → デフォルトのテクスチャを使用します\n", texPath.c_str());

			// 仮のデフォルトテクスチャ（例えば白色のテクスチャ）を使用する
			mainTex = Texture2D::Get(L"assets/default.png");

			if (!mainTex) {
				printf("デフォルトのテクスチャも読み込めませんでした。\n");
				continue;  // デフォルトテクスチャも読み込めなければ次のメッシュへ進む
			}
		}

		// テクスチャが無事に取得できた場合、ディスクリプタヒープに登録
		auto handle = descriptorHeap->Register(mainTex);
		if (!handle) {
			printf("Register() failed: handle is nullptr\n");
			continue;  // 登録に失敗した場合は次のメッシュへ進む
		}

		materialHandles.push_back(handle);  // ハンドルを保持
	}

	// ルートシグネチャーの作成
	rootSignature = new RootSignature();
	if (!rootSignature->IsValid())
	{
		printf("ルートシグネチャの生成に失敗\n");
		return false;
	}

	// パイプラインステートの作成
	pipelineState = new PipelineState();
	pipelineState->SetInputLayout(Vertex::InputLayout);
	pipelineState->SetRootSignature(rootSignature->Get());
	pipelineState->SetVS(L"../x64/Debug/SimpleVS.cso");
	pipelineState->SetPS(L"../x64/Debug/SimplePS.cso");
	pipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
	if (!pipelineState->IsValid())
	{
		printf("パイプラインステートの生成に失敗\n");
		return false;
	}

	printf("シーンの初期化に成功\n");
	return true;
}

void Scene::Update()
{
	//rotateY += 0.02f;
	//auto currentIndex = g_Engine->CurrentBackBufferIndex();						// 現在のフレーム番号を取得
	//auto currentTransform = constantBuffer[currentIndex]->GetPtr<Transform>();	// 現在のフレーム番号に対応する定数バッファを取得
	//currentTransform->World = DirectX::XMMatrixRotationY(rotateY);				// Y軸で回転させる

	// カメラ移動
	if (GetAsyncKeyState('A')) {
		posX -= 0.1f;
	}
	if (GetAsyncKeyState('D')) {
		posX += 0.1f;
	}
	if (GetAsyncKeyState('S')) {
		posY -= 0.1f;
	}
	if (GetAsyncKeyState('W')) {
		posY += 0.1f;
	}

	// カメラ位置を更新
	eyePos = XMVectorSet(
		posX,
		posY,
		XMVectorGetZ(eyePos),
		XMVectorGetW(eyePos));

	// 正面を見るようにする
	XMVECTOR forward = XMVectorSet(0, 0, -1, 0); // 右手座標系でカメラ前方は -Z
	targetPos = XMVectorAdd(eyePos, forward);

	// カメラの情報を更新する
	auto currentIndex = g_Engine->CurrentBackBufferIndex();
	auto ptr = constantBuffer[currentIndex]->GetPtr<Transform>();

	ptr->World = XMMatrixIdentity();
	ptr->View = XMMatrixLookAtRH(eyePos, targetPos, upward);

	if (GetAsyncKeyState(VK_SPACE)) {
		m_game->ChangeScene("Game");
	}
}

void Scene::Draw()
{
	auto currentIndex = g_Engine->CurrentBackBufferIndex();
	auto commandList = g_Engine->CommandList();
	auto materialHeap = descriptorHeap->GetHeap(); // ディスクリプタヒープ

	for (size_t i = 0; i < meshes.size(); i++)
	{
		auto vbView = vertexBuffers[i]->View();
		auto ibView = indexBuffers[i]->View();

		commandList->SetGraphicsRootSignature(rootSignature->Get());
		commandList->SetPipelineState(pipelineState->Get());
		commandList->SetGraphicsRootConstantBufferView(0, constantBuffer[currentIndex]->GetAddress());

		commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		commandList->IASetVertexBuffers(0, 1, &vbView);
		commandList->IASetIndexBuffer(&ibView);

		commandList->SetDescriptorHeaps(1, &materialHeap); // 使用するディスクリプタヒープをセット
		commandList->SetGraphicsRootDescriptorTable(1, materialHandles[i]->HandleGPU); // そのメッシュに対応するディスクリプタテーブルをセット

		commandList->DrawIndexedInstanced(meshes[i].Indices.size(), 1, 0, 0, 0);
	}
}

