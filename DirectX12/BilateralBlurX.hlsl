Texture2D<float> g_InputDepth : register(t0);
SamplerState g_LinearClamp : register(s0);
RWTexture2D<float> g_OutputDepth : register(u0);

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 coord = dispatchThreadId.xy;
    uint width, height;
    g_InputDepth.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
    {
        return;
    }

    float center = g_InputDepth.Load(int3(coord, 0));
    float sigmaSpatial = 2.0f;
    float sigmaRange = 0.5f;
    float sum = 0.0f;
    float weightSum = 0.0f;

    [unroll]
    for (int offset = -3; offset <= 3; ++offset)
    {
        int sampleX = clamp(int(coord.x) + offset, 0, int(width) - 1);
        uint2 sampleCoord = uint2(sampleX, coord.y);
        float sampleDepth = g_InputDepth.Load(int3(sampleCoord, 0));
        float spatial = exp(- (offset * offset) / (2.0f * sigmaSpatial * sigmaSpatial));
        float range = exp(- pow(sampleDepth - center, 2.0f) / (2.0f * sigmaRange * sigmaRange));
        float weight = spatial * range;
        sum += sampleDepth * weight;
        weightSum += weight;
    }

    g_OutputDepth[coord] = (weightSum > 0.0f) ? (sum / weightSum) : center;
}
