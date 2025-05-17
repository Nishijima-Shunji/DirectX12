#pragma once
class Object
{
private:

public:
	virtual bool Init() { return true; };
	virtual void Update() {};
	virtual void Draw() {};
};

