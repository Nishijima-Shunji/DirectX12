#pragma once
struct IActor {
	virtual ~IActor() = default;
	virtual void Update(float deltaTime) = 0;
	virtual void Render(ID3D12GraphicsCommandList* cmd) = 0;

	bool IsAlive = true;
};