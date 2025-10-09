#include "SharedStruct.hlsli"

struct PSIn
{
    float4 position    : SV_POSITION;
    float3 viewCenter  : TEXCOORD0;
    float  radius      : TEXCOORD1;
    float2 localOffset : TEXCOORD2;
};

struct PSOut
{
    float depth     : SV_Target0;
    float thickness : SV_Target1;
};

PSOut main(PSIn input)
{
    PSOut output;
    output.depth = 1000000.0f;     // ※MINブレンドに備えて十分大きな初期値を設定
    output.thickness = 0.0f;

    // 補間されたローカル座標（-1〜1）からスプライト内の位置を求める
    float2 local = input.localOffset;
    float radius = input.radius;

    float r2 = dot(local, local);
    if (r2 > 1.0f)
    {
        discard; // 円外は書き込まない
    }

    float radiusSq = radius * radius;
    float surfaceSq = max(radiusSq - radiusSq * r2, 0.0f);
    float viewOffset = sqrt(surfaceSq);

    // ビュー空間の最近接深度を算出（カメラは +Z 方向を見ている想定）
    float surfaceDepth = input.viewCenter.z - viewOffset;
    surfaceDepth = max(surfaceDepth, nearZ);

    // 厚みはビューレイ方向の通過距離（前面＋背面）
    float thickness = 2.0f * viewOffset;

    output.depth = surfaceDepth;
    output.thickness = max(thickness, 0.0f);

    return output;
}
