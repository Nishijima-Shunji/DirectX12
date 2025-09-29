Texture2D<float> g_Depth : register(t0);
RWTexture2D<float4> g_Normal : register(u0);

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 coord = dispatchThreadId.xy;
    uint width, height;
    g_Depth.GetDimensions(width, height);
    if (coord.x >= width || coord.y >= height)
    {
        return;
    }

    float center = g_Depth.Load(int3(coord, 0));
    float depthRight = g_Depth.Load(int3(uint2(min(coord.x + 1, width - 1), coord.y), 0));
    float depthUp = g_Depth.Load(int3(uint2(coord.x, min(coord.y + 1, height - 1)), 0));

    float3 dx = float3(1.0f, 0.0f, depthRight - center);
    float3 dy = float3(0.0f, 1.0f, depthUp - center);
    float3 normal = normalize(cross(dx, dy));
    normal = (normal * 0.5f) + 0.5f;

    g_Normal[coord] = float4(normal, 1.0f);
}
