cbuffer MetaCB : register(b0)
{
    float4x4 invViewProj;   // ビュー射影逆行列
    float4x4 viewProj;      // ビュー射影行列（深度計算用）
    float4   camRadius;     // xyz: カメラ位置, w: 粒子半径（描画用）
    float4   isoCount;      // x: しきい値, y: 粒子数, z: レイステップ係数, w: 未使用
    float4   gridMinCell;   // xyz: グリッド最小座標, w: セルサイズ
    uint4    gridDimInfo;   // xyz: グリッド寸法, w: セル総数
    float4   waterDeep;     // xyz: 深い水の色, w: 吸収係数
    float4   waterShallow;  // xyz: 浅い水の色, w: 泡検出のしきい値
    float4   shadingParams; // x: 泡強度, y: 反射割合, z: スペキュラパワー, w: 経過時間
};

struct ParticleMeta
{
    float3 pos;  // 粒子のワールド座標
    float  r;    // 個別半径（今回は共通値）
};

struct VSOutput
{
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD;
};

StructuredBuffer<ParticleMeta> Particles : register(t0);
StructuredBuffer<uint>        GridTable : register(t1);
StructuredBuffer<uint>        GridCount : register(t2);

static const uint MAX_PARTICLES_PER_CELL = 64;
static const int  NEIGHBOR_SPAN = 1;

// -----------------------------------------------------------------------------
// グリッドを利用しない（フォールバック）密度評価
// -----------------------------------------------------------------------------
float EvaluateFieldBruteForce(float3 p, out float3 grad)
{
    grad = float3(0, 0, 0);
    float sum = 0.0;
    const uint count = (uint)isoCount.y;
    const float iso = isoCount.x;

    [loop]
    for (uint i = 0; i < count; ++i)
    {
        ParticleMeta meta = Particles[i];
        float3 d = p - meta.pos;
        float  radius = max(meta.r, 1e-4);
        float  r2 = radius * radius;
        float  dist2 = dot(d, d);
        float  cutoff2 = r2 * 6.25; // kernelMul=2.5f を二乗した値
        if (dist2 > cutoff2)
        {
            continue;
        }

        float influence = exp(-dist2 / r2);
        if (influence < 1e-4)
        {
            continue;
        }

        sum += influence;
        grad += (-2.0 * influence / r2) * d;

        if (sum > iso + 0.75)
        {
            break;
        }
    }

    return sum - iso;
}

// -----------------------------------------------------------------------------
// 空間グリッドを用いた密度評価。総参照粒子数を返して空セルの判定に使う。
// -----------------------------------------------------------------------------
float EvaluateFieldGrid(float3 p, out float3 grad, out uint totalSamples)
{
    grad = float3(0, 0, 0);
    totalSamples = 0;
    float sum = 0.0;

    const uint particleCount = (uint)isoCount.y;
    const float iso = isoCount.x;
    const uint totalCells = gridDimInfo.w;
    const float cellSize = gridMinCell.w;

    if (particleCount == 0)
    {
        return -iso;
    }

    const bool gridUsable = (totalCells > 0) && (cellSize > 0.0f) &&
        (gridDimInfo.x > 0) && (gridDimInfo.y > 0) && (gridDimInfo.z > 0);

    if (!gridUsable)
    {
        float value = EvaluateFieldBruteForce(p, grad);
        totalSamples = particleCount;
        return value;
    }

    float3 gridMin = gridMinCell.xyz;
    uint3  gridDim = uint3(gridDimInfo.xyz);
    uint   maxTableIndex = totalCells * MAX_PARTICLES_PER_CELL;

    int3 baseCell = int3(floor((p - gridMin) / cellSize));

    [loop]
    for (int oz = -NEIGHBOR_SPAN; oz <= NEIGHBOR_SPAN; ++oz)
    {
        for (int oy = -NEIGHBOR_SPAN; oy <= NEIGHBOR_SPAN; ++oy)
        {
            for (int ox = -NEIGHBOR_SPAN; ox <= NEIGHBOR_SPAN; ++ox)
            {
                int3 cell = baseCell + int3(ox, oy, oz);
                if (cell.x < 0 || cell.y < 0 || cell.z < 0)
                {
                    continue;
                }
                if (cell.x >= int(gridDim.x) || cell.y >= int(gridDim.y) || cell.z >= int(gridDim.z))
                {
                    continue;
                }

                uint cellIndex = uint(cell.x) + gridDim.x * (uint(cell.y) + gridDim.y * uint(cell.z));
                if (cellIndex >= totalCells)
                {
                    continue;
                }

                uint count = GridCount[cellIndex];
                if (count == 0)
                {
                    continue;
                }

                count = min(count, MAX_PARTICLES_PER_CELL);
                uint baseIndex = cellIndex * MAX_PARTICLES_PER_CELL;
                totalSamples += count;

                [loop]
                for (uint i = 0; i < count; ++i)
                {
                    uint tableIndex = baseIndex + i;
                    if (tableIndex >= maxTableIndex)
                    {
                        break;
                    }

                    uint particleIndex = GridTable[tableIndex];
                    if (particleIndex >= particleCount)
                    {
                        continue;
                    }

                    ParticleMeta meta = Particles[particleIndex];
                    float3 d = p - meta.pos;
                    float  radius = max(meta.r, 1e-4);
                    float  r2 = radius * radius;
                    float  dist2 = dot(d, d);
                    float  cutoff2 = r2 * 6.25; // kernelMul=2.5f の二乗
                    if (dist2 > cutoff2)
                    {
                        continue;
                    }

                    float influence = exp(-dist2 / r2);
                    if (influence < 1e-4)
                    {
                        continue;
                    }

                    sum += influence;
                    grad += (-2.0 * influence / r2) * d;

                    if (sum > iso + 0.75)
                    {
                        return sum - iso;
                    }
                }
            }
        }
    }

    return sum - iso;
}

struct PSOutput
{
    float4 color : SV_TARGET;
    float  depth : SV_DEPTH;
};

PSOutput main(VSOutput IN)
{
    // フルスクリーン三角形のUVからレイを生成
    float4 clip = float4(IN.uv * 2.0 - 1.0, 0.0, 1.0);
    float4 wp = mul(invViewProj, clip);
    wp /= wp.w;

    float3 ro = camRadius.xyz;
    float3 rd = normalize(wp.xyz - ro);

    float3 p = ro;
    float3 grad = float3(0, 0, 0);
    float  field = -isoCount.x;
    bool   hit = false;

    float cellSize = max(gridMinCell.w, 0.01);
    float3 gridExtent = float3(gridDimInfo.xyz) * cellSize;
    float maxDistance = length(gridExtent) + cellSize * 4.0;
    maxDistance = max(maxDistance, 50.0);

    float travelled = 0.0;
    const int MAX_STEP = 80;

    [loop]
    for (int i = 0; i < MAX_STEP && travelled < maxDistance; ++i)
    {
        float3 localGrad;
        uint    sampleCount;
        float   value = EvaluateFieldGrid(p, localGrad, sampleCount);

        if (gridDimInfo.w > 0 && sampleCount == 0)
        {
            // 粒子が存在しないセルは大きくスキップして高速化
            float skip = max(cellSize * 1.25, 0.1);
            p += rd * skip;
            travelled += skip;
            continue;
        }

        grad = localGrad;
        field = value;

        if (abs(value) < 0.01)
        {
            hit = true;
            break;
        }

        float stepLen = max(abs(value) * isoCount.z, 0.02);
        p += rd * (-value) * isoCount.z;
        travelled += stepLen;
    }

    if (!hit)
    {
        discard;
    }

    float3 n = (dot(grad, grad) > 1e-6) ? normalize(grad) : float3(0, 1, 0);
    float3 viewDir = normalize(ro - p);
    float3 lightDir = normalize(float3(-0.35, 0.9, -0.25));

    float diff = saturate(dot(n, lightDir));

    float viewDist = length(ro - p);
    float absorption = exp(-viewDist * waterDeep.w);
    float3 baseWater = lerp(waterDeep.xyz, waterShallow.xyz, absorption);

    float3 envColor = float3(0.05, 0.16, 0.28);

    float3 diffuse = baseWater * (0.25 + diff * 0.75);

    float fresnel = pow(1.0 - saturate(dot(n, viewDir)), 5.0);
    float3 reflection = lerp(baseWater, float3(0.75, 0.9, 1.0), shadingParams.y);

    float curvature = saturate(1.0 - length(grad) * waterShallow.w);
    float foamWave = 0.5 + 0.5 * sin(dot(p.xz, float2(0.8, 0.45)) * 2.2 + shadingParams.w * 1.4);
    float foamMask = saturate((curvature - 0.25) * 2.0) * foamWave;

    float specPower = max(shadingParams.z, 4.0);
    float spec = pow(saturate(dot(reflect(-lightDir, n), viewDir)), specPower);

    float3 color = diffuse;
    color += reflection * fresnel;
    color += spec * 0.35;
    color = lerp(envColor, color, 0.65);
    color += shadingParams.x * foamMask;

    float alpha = saturate(0.35 + absorption * 0.45 + fresnel * 0.35);
    color = lerp(envColor, color, alpha);

    float4 clipPos = mul(viewProj, float4(p, 1.0));
    float invW = (abs(clipPos.w) > 1e-6) ? rcp(clipPos.w) : 0.0;
    float depth = saturate(clipPos.z * invW);

    PSOutput OUT;
    OUT.color = float4(saturate(color), alpha);
    OUT.depth = depth;
    return OUT;
}
