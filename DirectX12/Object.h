#pragma once
class Object
{
private:

public:
	virtual ~Object() = default;

	virtual bool Init() { return true; };
	virtual void Update() {};
	virtual void Draw() {};
};

