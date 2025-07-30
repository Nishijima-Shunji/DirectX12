#include "SpatialGrid.h"
#include <cmath>

SpatialGrid::SpatialGrid(float cellSize) : m_cellSize(cellSize) {}

void SpatialGrid::Clear() {
    m_cells.clear();
}

SpatialGrid::Int3 SpatialGrid::ToCell(const DirectX::XMFLOAT3& pos) const {
    return Int3{
        static_cast<int>(std::floor(pos.x / m_cellSize)),
        static_cast<int>(std::floor(pos.y / m_cellSize)),
        static_cast<int>(std::floor(pos.z / m_cellSize))
    };
}

void SpatialGrid::Insert(size_t index, const DirectX::XMFLOAT3& pos) {
    m_cells[ToCell(pos)].push_back(index);
}

void SpatialGrid::Query(const DirectX::XMFLOAT3& center, float radius, std::vector<size_t>& results) const {
    int minX = static_cast<int>(std::floor((center.x - radius) / m_cellSize));
    int maxX = static_cast<int>(std::floor((center.x + radius) / m_cellSize));
    int minY = static_cast<int>(std::floor((center.y - radius) / m_cellSize));
    int maxY = static_cast<int>(std::floor((center.y + radius) / m_cellSize));
    int minZ = static_cast<int>(std::floor((center.z - radius) / m_cellSize));
    int maxZ = static_cast<int>(std::floor((center.z + radius) / m_cellSize));
    results.clear();
    for (int x = minX; x <= maxX; ++x) {
        for (int y = minY; y <= maxY; ++y) {
            for (int z = minZ; z <= maxZ; ++z) {
                Int3 key{ x, y, z };
                auto it = m_cells.find(key);
                if (it != m_cells.end()) {
                    results.insert(results.end(), it->second.begin(), it->second.end());
                }
            }
        }
    }
}
