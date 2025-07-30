#include "SphereMeshGenerator.h"
#include <cmath>

MeshData CreateLowPolySphere(float radius, int subdivisions) {
    MeshData mesh;

    // �����ɒ�|�����̒��_�E�C���f�b�N�X�������W�b�N������
    // ��F�P���Ȑ����ʑ̋���ICO�����ו���(subdivision)���č����@�Ȃ�

    // �T���v���Ƃ��āA�ȒP�Ȑ����ʑ̂̒��_
    mesh.vertices = {
        {{0, radius, 0}, {0,1,0}},
        {{-radius, 0, 0}, {-1,0,0}},
        {{0, 0, radius}, {0,0,1}},
        {{radius, 0, 0}, {1,0,0}},
        {{0, 0, -radius}, {0,0,-1}},
        {{0, -radius, 0}, {0,-1,0}},
    };

    // �C���f�b�N�X�i�ʁj
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

    // subdivisions�ōו������\�i�C�Ӂj

    return mesh;
}
