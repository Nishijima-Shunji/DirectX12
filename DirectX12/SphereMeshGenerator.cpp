#include "SphereMeshGenerator.h"
#include <cmath>

MeshData CreateLowPolySphere(float radius, int subdivisions) {
    MeshData mesh;

    // ここに低ポリ球の頂点・インデックス生成ロジックを実装
    // 例：単純な正八面体球やICO球を細分割(subdivision)して作る方法など

    // サンプルとして、簡単な正八面体の頂点
    mesh.vertices = {
        {{0, radius, 0}, {0,1,0}},
        {{-radius, 0, 0}, {-1,0,0}},
        {{0, 0, radius}, {0,0,1}},
        {{radius, 0, 0}, {1,0,0}},
        {{0, 0, -radius}, {0,0,-1}},
        {{0, -radius, 0}, {0,-1,0}},
    };

    // インデックス（面）
    mesh.indices = {
        0, 1, 2,
        0, 2, 3,
        0, 3, 4,
        0, 4, 1,
        5, 2, 1,
        5, 3, 2,
        5, 4, 3,
        5, 1, 4,
    };

    // subdivisionsで細分割も可能（任意）

    return mesh;
}
