#pragma once
class Object
{
private:

public:
	virtual ~Object() = default;

	virtual bool Init() { return true; };
	virtual void Update(float deltaTime) {};
	virtual void Draw() {};
};

