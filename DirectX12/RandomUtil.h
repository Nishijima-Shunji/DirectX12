#pragma once
#include <random>

inline float RandFloat(float min, float max)
{
    static std::mt19937 gen{ std::random_device{}() };
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

