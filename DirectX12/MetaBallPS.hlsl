cbuffer MetaCB : register(b0)
{
    float4x4 invViewProj;
    float3 camPos;
    float isoLevel;
    uint particleCount;
    float3 gridMin;
    uint3 gridDim;
    float radius;
    float pad;
};

struct ParticleMeta
{
    float3 pos; // ワールド空間位置
    float r; // 半径
};

struct VSOutput
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD;
};

static const int MAX_STEP = 16;
static const uint MAX_PARTICLES_PER_CELL = 64;

// リソース定義
StructuredBuffer<ParticleMeta> Particles : register(t0);
StructuredBuffer<uint> GridCounts : register(t1);
StructuredBuffer<uint> GridTable : register(t2);

// MetaBallのフィールド計算
float Field(float3 p, out float3 grad)
{
    float sum = 0;
    grad = float3(0, 0, 0);

    // ワールド座標pからグリッドの3Dセルインデックスを計算
    int3 cell3D = int3((p - gridMin) / radius);

    // 自身のセルと隣接する26セル(3x3x3)を探索
    [loop]
    for (int z = -1; z <= 1; ++z)
    {
        [loop]
        for (int y = -1; y <= 1; ++y)
        {
            [loop]
            for (int x = -1; x <= 1; ++x)
            {
                int3 neighborCell = cell3D + int3(x, y, z);

                // グリッド範囲外はスキップ
                if (any(neighborCell < 0) || any(neighborCell >= gridDim))
                {
                    continue;
                }

                // 3Dセルインデックスを1Dインデックスに変換
                uint cellIdx = neighborCell.x + neighborCell.y * gridDim.x + neighborCell.z * gridDim.x * gridDim.y;
                uint countInCell = GridCounts[cellIdx];
                if (countInCell > 0)
                {
                    uint startIdx = cellIdx * MAX_PARTICLES_PER_CELL;
                    // セル内のパーティクルだけをループ
                    for (uint i = 0; i < countInCell; ++i)
                    {
                        uint particleIdx = GridTable[startIdx + i];
                        
                        // 元のフィールド値と勾配の計算
                        float3 d = p - Particles[particleIdx].pos;
                        float r2 = Particles[particleIdx].r * Particles[particleIdx].r;
                        float denom = dot(d, d) + 1e-6;
                        sum += r2 / denom;
                        grad += (-2 * r2 * d) / pow(denom, 2);
                    }
                }
            }
        }
    }
    
    return sum - isoLevel;
}


float4 main(VSOutput IN) : SV_TARGET
{
    float4 clip = float4(IN.uv * 2 - 1, 0, 1);
    float4 wp = mul(invViewProj, clip);
    wp /= wp.w;
    float3 ro = camPos, rd = normalize(wp.xyz - camPos);
    float3 p = ro;
    float d;
    float3 grad;

    [loop]
    for (int i = 0; i < MAX_STEP; ++i)
    {
        d = Field(p, grad);
        if (abs(d) < 0.001)
            break;
        p += rd * d * 0.2;
    }

    if (abs(d) >= 0.001)
        discard;

    float3 n = normalize(grad);
    float diff = saturate(dot(n, normalize(float3(1, 1, 1))));
    return float4(diff * 0.2, diff * 0.4, diff, 1);
}