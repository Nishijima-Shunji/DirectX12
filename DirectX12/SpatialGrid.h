#pragma once
#include <unordered_map>
#include <vector>
#include <DirectXMath.h>

class SpatialGrid {
public:
    explicit SpatialGrid(float cellSize = 0.1f);

    void Clear();
    void SetCellSize(float size) { m_cellSize = size; }
    float CellSize() const { return m_cellSize; }
    void Insert(size_t index, const DirectX::XMFLOAT3& pos);
    void Query(const DirectX::XMFLOAT3& center, float radius, std::vector<size_t>& results) const;

private:
    struct Int3 {
        int x, y, z;
        bool operator==(const Int3& other) const { return x == other.x && y == other.y && z == other.z; }
    };
    struct Int3Hash {
        size_t operator()(const Int3& k) const {
            size_t h1 = std::hash<int>{}(k.x);
            size_t h2 = std::hash<int>{}(k.y);
            size_t h3 = std::hash<int>{}(k.z);
            return ((h1 ^ (h2 << 1)) >> 1) ^ (h3 << 1);
        }
    };

    float m_cellSize;
    std::unordered_map<Int3, std::vector<size_t>, Int3Hash> m_cells;

    Int3 ToCell(const DirectX::XMFLOAT3& pos) const;
};
