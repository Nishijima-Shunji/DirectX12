#include "SharedStruct.hlsli"

RWTexture2D<uint> g_FluidDepthUAV    : register(u0); // R32_UINT に線形深度を格納（float を asuint で詰め込む）
RWTexture2D<uint> g_FluidThicknessUAV : register(u1); // 厚みも同様に float ビット列で蓄積

struct PSIn
{
    float4 position    : SV_POSITION;
    float3 viewCenter  : TEXCOORD0;
    float  radius      : TEXCOORD1;
    float2 localOffset : TEXCOORD2;
};

// float を保持している UAV に対して最小値を書き込むヘルパー
void StoreMinDepth(uint2 pixel, float depth)
{
    uint newBits = asuint(depth);
    uint oldBits = g_FluidDepthUAV[pixel];

    // 既存値が 0（未初期化）の場合は優先的に書き込む
    if (oldBits == 0)
    {
        uint exchanged;
        InterlockedCompareExchange(g_FluidDepthUAV[pixel], 0, newBits, exchanged);
        if (exchanged == 0)
        {
            return;
        }
        oldBits = exchanged;
    }

    // 既存の深度より近い場合のみ更新する（0 は未使用として扱う）
    [loop]
    while (asfloat(oldBits) > depth)
    {
        uint exchanged;
        InterlockedCompareExchange(g_FluidDepthUAV[pixel], oldBits, newBits, exchanged);
        if (exchanged == oldBits)
        {
            break;
        }
        oldBits = exchanged;
    }
}

// 厚みを float として蓄積する（CAS で競合を解消）
void AccumulateThickness(uint2 pixel, float thickness)
{
    uint oldBits = g_FluidThicknessUAV[pixel];

    [loop]
    while (true)
    {
        float oldValue = asfloat(oldBits);
        float newValue = oldValue + thickness;
        uint newBits = asuint(newValue);

        uint exchanged;
        InterlockedCompareExchange(g_FluidThicknessUAV[pixel], oldBits, newBits, exchanged);
        if (exchanged == oldBits)
        {
            break;
        }
        oldBits = exchanged;
    }
}

float4 main(PSIn input) : SV_TARGET
{
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

    uint2 pixel = uint2(input.position.xy);
    StoreMinDepth(pixel, surfaceDepth);
    AccumulateThickness(pixel, thickness);

    // 本シェーダーでは UAV 書き込みのみを行い、RT には出力しない
    return float4(0, 0, 0, 0);
}
