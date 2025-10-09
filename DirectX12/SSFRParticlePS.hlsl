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
    float2 spritePos = local * radius;
    float surfaceSq = max(radiusSq - dot(spritePos, spritePos), 0.0f);
    float viewOffset = sqrt(surfaceSq);

    // 板ポリを球として描くための前面深度（球の最近点）を計算する
    float surfaceDepth = input.viewCenter.z - viewOffset;

    // 板ポリが球に見えるように、ビューレイ方向の通過厚みを算出する
    float thickness = 2.0f * viewOffset;

    output.depth = surfaceDepth;
    output.thickness = thickness;

    return output;
}
