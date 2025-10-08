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
    output.depth = 0.0f;
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

    float frontDepth = input.viewCenter.z - viewOffset;
    float backDepth = input.viewCenter.z + viewOffset;
    float clampedFront = max(frontDepth, nearZ);
    float clampedBack = min(backDepth, farZ);
    float thickness = max(clampedBack - clampedFront, 0.0f); // ニア面/ファー面を跨いだ場合は欠損分を差し引く
    thickness = min(thickness, 65504.0f); // R16_FLOAT へ収めるため半精度上限でサチらせる

    if (thickness <= 0.0f)
    {
        discard; // 視錐台に寄与しない場合は書き込まない
    }

    output.depth = min(clampedFront, farZ);
    output.thickness = thickness;

    return output;
}
