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

std::vector<Mesh> meshes;					// ���b�V���̔z��
std::vector<VertexBuffer*> vertexBuffers;	// ���b�V���̐����̒��_�o�b�t�@
std::vector<IndexBuffer*> indexBuffers;		// ���b�V���̐����̃C���f�b�N�X�o�b�t�@

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
std::vector< DescriptorHandle*> materialHandles; // �e�N�X�`���p�̃n���h���ꗗ

bool Scene::Init()
{
	camera = new Camera();
	g_Engine->RegisterObj("SceneCamera", camera);

	for (size_t i = 0; i < Engine::FRAME_BUFFER_COUNT; i++) {
		constantBuffer[i] = new ConstantBuffer(sizeof(Transform));
		auto ptr = constantBuffer[i]->GetPtr<Transform>();
		ptr->World = XMMatrixIdentity();
		ptr->View = camera->GetViewMatrix();
		ptr->Proj = camera->GetProjMatrix();
	}

	ImportSettings importSetting =				// ����̓ǂݍ��ݐݒ�\����
	{
		modelFile,
		meshes,
		false,
		true // �A���V�A�̃��f���́A�e�N�X�`����UV��V�������]���Ă���ۂ��H�̂œǂݍ��ݎ���UV���W���t�]������
	};

	AssimpLoader loader;
	if (!loader.Load(importSetting))
	{
		return false;
	}

	// ���b�V���̐��������_�o�b�t�@��p�ӂ���
	vertexBuffers.reserve(meshes.size());
	for (size_t i = 0; i < meshes.size(); i++)
	{
		auto size = sizeof(Vertex) * meshes[i].Vertices.size();
		auto stride = sizeof(Vertex);
		auto vertices = meshes[i].Vertices.data();
		auto pVB = new VertexBuffer(size, stride, vertices);
		if (!pVB->IsValid())
		{
			printf("���_�o�b�t�@�̐����Ɏ��s\n");
			return false;
		}

		vertexBuffers.push_back(pVB);
	}

	// ���b�V���̐������C���f�b�N�X�o�b�t�@��p�ӂ���
	indexBuffers.reserve(meshes.size());
	for (size_t i = 0; i < meshes.size(); i++)
	{
		auto size = sizeof(uint32_t) * meshes[i].Indices.size();
		auto indices = meshes[i].Indices.data();
		auto pIB = new IndexBuffer(size, indices);
		if (!pIB->IsValid())
		{
			printf("�C���f�b�N�X�o�b�t�@�̐����Ɏ��s\n");
			return false;
		}

		indexBuffers.push_back(pIB);
	}

	// �}�e���A���̓ǂݍ���
	materialHandles.clear();
	descriptorHeap = new DescriptorHeap();
	for (size_t i = 0; i < meshes.size(); i++)
	{
		// �e�N�X�`���t�@�C���p�X�̐����iTGA�`���ɕϊ��j
		auto texPath = ReplaceExtension(meshes[i].DiffuseMap, "png");

		// �e�N�X�`����ǂݍ���
		auto mainTex = Texture2D::Get(texPath);

		// �e�N�X�`����������Ȃ��ꍇ�A�f�t�H���g�̃e�N�X�`�����g�p
		if (!mainTex) {
			printf("�e�N�X�`���̓ǂݍ��݂Ɏ��s���܂���: %ws �� �f�t�H���g�̃e�N�X�`�����g�p���܂�\n", texPath.c_str());

			// ���̃f�t�H���g�e�N�X�`���i�Ⴆ�Δ��F�̃e�N�X�`���j���g�p����
			mainTex = Texture2D::Get(L"assets/default.png");

			if (!mainTex) {
				printf("�f�t�H���g�̃e�N�X�`�����ǂݍ��߂܂���ł����B\n");
				continue;  // �f�t�H���g�e�N�X�`�����ǂݍ��߂Ȃ���Ύ��̃��b�V���֐i��
			}
		}

		// �e�N�X�`���������Ɏ擾�ł����ꍇ�A�f�B�X�N���v�^�q�[�v�ɓo�^
		auto handle = descriptorHeap->Register(mainTex);
		if (!handle) {
			printf("Register() failed: handle is nullptr\n");
			continue;  // �o�^�Ɏ��s�����ꍇ�͎��̃��b�V���֐i��
		}

		materialHandles.push_back(handle);  // �n���h����ێ�
	}

	// ���[�g�V�O�l�`���[�̍쐬
	rootSignature = new RootSignature();
	if (!rootSignature->IsValid())
	{
		printf("���[�g�V�O�l�`���̐����Ɏ��s\n");
		return false;
	}

	// �p�C�v���C���X�e�[�g�̍쐬
	pipelineState = new PipelineState();
	pipelineState->SetInputLayout(Vertex::InputLayout);
	pipelineState->SetRootSignature(rootSignature->Get());
	pipelineState->SetVS(L"SimpleVS.cso");
	pipelineState->SetPS(L"SimplePS.cso");
	pipelineState->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
	if (!pipelineState->IsValid())
	{
		printf("�p�C�v���C���X�e�[�g�̐����Ɏ��s\n");
		return false;
	}

	printf("�V�[���̏������ɐ���\n");
	return true;
}

void Scene::Update(float deltaTime)
{
	camera->Update(deltaTime);
	auto currentIndex = g_Engine->CurrentBackBufferIndex();
	auto ptr = constantBuffer[currentIndex]->GetPtr<Transform>();
	ptr->World = XMMatrixIdentity();
	ptr->View = camera->GetViewMatrix();
	ptr->Proj = camera->GetProjMatrix();

	if (GetAsyncKeyState(VK_SPACE)) {
		m_game->ChangeScene("Game");
	}
}

void Scene::Draw()
{
	auto currentIndex = g_Engine->CurrentBackBufferIndex();
	auto commandList = g_Engine->CommandList();
	auto materialHeap = descriptorHeap->GetHeap(); // �f�B�X�N���v�^�q�[�v

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

		commandList->SetDescriptorHeaps(1, &materialHeap); // �g�p����f�B�X�N���v�^�q�[�v���Z�b�g
		commandList->SetGraphicsRootDescriptorTable(1, materialHandles[i]->HandleGPU); // ���̃��b�V���ɑΉ�����f�B�X�N���v�^�e�[�u�����Z�b�g

		commandList->DrawIndexedInstanced(meshes[i].Indices.size(), 1, 0, 0, 0);
	}
}

