#include "SceneManager.h"
#include "Scene.h"

void SceneManager::RegisterScene(const std::string& name, SceneFactory factory)
{
	m_Registry[name] = factory;
}

void SceneManager::ChangeScene(const std::string& name)
{
	auto it = m_Registry.find(name);
	// ’†g‚ª‚ ‚é‚È‚ç
	if (it != m_Registry.end())
	{
		if (!m_SceneStack.empty()) {
			m_SceneStack.pop();
		}
		m_SceneStack.push(it->second());
	}
}

void SceneManager::PushScene(const std::string& name)
{
	auto it = m_Registry.find(name);
	if (it != m_Registry.end()) {
		m_SceneStack.push(it->second());
	}
}


void SceneManager::PopScene()
{
	if (!m_SceneStack.empty()) {
		m_SceneStack.pop();
	}
}


void SceneManager::Update()
{
	if (!m_SceneStack.empty()) {
		m_SceneStack.top()->Update();
	}
}


void SceneManager::Render()
{
	if (!m_SceneStack.empty()) {
		m_SceneStack.top()->Draw();
	}
}
