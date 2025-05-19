#pragma once
#include <vector>
#include <DirectXMath.h>
#include "SharedStruct.h"

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

MeshData CreateLowPolySphere(float radius, int subdivisions);
