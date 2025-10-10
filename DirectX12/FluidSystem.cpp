#include "FluidSystem.h"
#include "Engine.h"
#include "SphereMeshGenerator.h"
#include "SharedStruct.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

using namespace DirectX;

namespace
{
    inline XMVECTOR LoadFloat3(const XMFLOAT3& v)
    {
        return XMLoadFloat3(&v);
    }

    inline XMFLOAT3 ToFloat3(const XMVECTOR& v)
    {
        XMFLOAT3 out{};
        XMStoreFloat3(&out, v);
        return out;
    }

    inline XMMATRIX LoadFloat3x3(const XMFLOAT3X3& m)
    {
        return XMLoadFloat3x3(&m);
    }

    inline XMFLOAT3X3 ToFloat3x3(const XMMATRIX& m)
    {
        XMFLOAT3X3 out{};
        XMStoreFloat3x3(&out, m);
        return out;
    }

    inline float QuadraticWeight(float x)
    {
        x = std::fabsf(x);
        if (x < 0.5f)
        {
            return 0.75f - x * x;
        }
        if (x < 1.5f)
        {
            float t = 1.5f - x;
            return 0.5f * t * t;
        }
        return 0.0f;
    }

    inline float QuadraticGradient(float x)
    {
        float s = (x >= 0.0f) ? 1.0f : -1.0f;
        x = std::fabsf(x);
        if (x < 0.5f)
        {
            return -2.0f * x * s;
        }
        if (x < 1.5f)
        {
            float t = 1.5f - x;
            return -t * s;
        }
        return 0.0f;
    }

    inline float Clamp(float v, float minV, float maxV)
    {
        return std::max(minV, std::min(maxV, v));
    }

    inline XMFLOAT3 Clamp3(const XMFLOAT3& value, const XMFLOAT3& minV, const XMFLOAT3& maxV)
    {
        return XMFLOAT3(
            Clamp(value.x, minV.x, maxV.x),
            Clamp(value.y, minV.y, maxV.y),
            Clamp(value.z, minV.z, maxV.z));
    }
}

FluidSystem::FluidSystem()
{
}

FluidSystem::~FluidSystem()
{
}

bool FluidSystem::Init(ID3D12Device* device, const Bounds& bounds, size_t particleCount, RenderMode mode)
{
    m_device = device;
    m_bounds = bounds;
    m_renderMode = mode;

    m_gridMin = bounds.min;

    if (!BuildParticles(particleCount))
    {
        return false;
    }
    if (!BuildGrid())
    {
        return false;
    }
    if (!BuildRenderResources())
    {
        return false;
    }
    if (!BuildInstanceBuffer())
    {
        return false;
    }

    RebuildSpatialGrid();

    return true;
}

void FluidSystem::Update(float deltaTime)
{
    StepMLSMPM(deltaTime);
    ApplyCameraLift(deltaTime);
    RebuildSpatialGrid();
    UpdateInstanceBuffer();
}

void FluidSystem::Draw(ID3D12GraphicsCommandList* cmd, const Camera& camera)
{
    if (!cmd)
    {
        return;
    }

    UpdateCameraCB(camera);

    auto* pso = (m_renderMode == RenderMode::InstancedSpheres) ? m_instancedPso.get() : m_ssfrPso.get();
    if (!pso)
    {
        return;
    }

    cmd->SetGraphicsRootSignature(m_rootSignature->Get());
    cmd->SetPipelineState(pso->Get());

    UINT frameIndex = g_Engine->CurrentBackBufferIndex();
    if (m_cameraCB[frameIndex])
    {
        cmd->SetGraphicsRootConstantBufferView(0, m_cameraCB[frameIndex]->GetAddress());
    }

    D3D12_VERTEX_BUFFER_VIEW views[2] = {};
    views[0] = m_meshVB->View();
    views[1] = m_instanceVB->View();
    cmd->IASetVertexBuffers(0, 2, views);

    if (m_meshIB)
    {
        cmd->IASetIndexBuffer(&m_meshIB->View());
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd->DrawIndexedInstanced(m_indexCount, static_cast<UINT>(m_particles.size()), 0, 0, 0);
    }
}

void FluidSystem::AdjustWall(const XMFLOAT3& direction, float amount)
{
    // ※壁の調整: 入力ベクトルで対応面を動かし水槽形状を動的変更する

    XMVECTOR dir = XMVector3Normalize(XMLoadFloat3(&direction));
    XMFLOAT3 dirF{};
    XMStoreFloat3(&dirF, dir);

    XMFLOAT3 min = m_bounds.min;
    XMFLOAT3 max = m_bounds.max;

    // 壁操作は最も寄与する軸のみを採用してシンプルに制御する
    if (std::fabsf(dirF.x) > std::fabsf(dirF.y) && std::fabsf(dirF.x) > std::fabsf(dirF.z))
    {
        if (dirF.x > 0.0f)
        {
            max.x += amount;
        }
        else
        {
            min.x += amount;
        }
    }
    else if (std::fabsf(dirF.y) > std::fabsf(dirF.z))
    {
        if (dirF.y > 0.0f)
        {
            max.y += amount;
        }
        else
        {
            min.y += amount;
        }
    }
    else
    {
        if (dirF.z > 0.0f)
        {
            max.z += amount;
        }
        else
        {
            min.z += amount;
        }
    }

    m_bounds.min = min;
    m_bounds.max = max;
    m_gridMin = m_bounds.min;
}

void FluidSystem::SetCameraLiftRequest(const XMFLOAT3& origin, const XMFLOAT3& direction, float deltaTime)
{
    m_liftRequested = true;
    m_liftRayOrigin = origin;
    m_liftRayDirection = direction;
    m_liftAccumulated += deltaTime;
}

void FluidSystem::ClearCameraLiftRequest()
{
    m_liftRequested = false;
    m_liftAccumulated = 0.0f;
    m_liftTargetIndex = static_cast<size_t>(-1);
}

bool FluidSystem::BuildParticles(size_t particleCount)
{
    m_particles.clear();
    m_particles.reserve(particleCount);

    XMFLOAT3 size{
        m_bounds.max.x - m_bounds.min.x,
        m_bounds.max.y - m_bounds.min.y,
        m_bounds.max.z - m_bounds.min.z };

    const float spacing = m_particleRadius * 2.0f;

    size_t count = 0;
    for (float y = m_bounds.min.y + spacing; y < m_bounds.max.y - spacing && count < particleCount; y += spacing)
    {
        for (float z = m_bounds.min.z + spacing; z < m_bounds.max.z - spacing && count < particleCount; z += spacing)
        {
            for (float x = m_bounds.min.x + spacing; x < m_bounds.max.x - spacing && count < particleCount; x += spacing)
            {
                Particle p{};
                p.position = XMFLOAT3(x, y, z);
                p.velocity = XMFLOAT3(0.0f, 0.0f, 0.0f);
                p.previous = p.position;
                p.mass = 1.0f;
                p.affine = XMFLOAT3X3(
                    0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f);
                m_particles.push_back(p);
                ++count;
            }
        }
    }

    // 粒子数が不足する場合でも柔軟に扱う
    if (m_particles.empty())
    {
        return false;
    }

    return true;
}

bool FluidSystem::BuildGrid()
{
    XMFLOAT3 extent{
        m_bounds.max.x - m_bounds.min.x,
        m_bounds.max.y - m_bounds.min.y,
        m_bounds.max.z - m_bounds.min.z };

    int dimX = static_cast<int>(std::ceil(extent.x / m_cellSize)) + 4;
    int dimY = static_cast<int>(std::ceil(extent.y / m_cellSize)) + 4;
    int dimZ = static_cast<int>(std::ceil(extent.z / m_cellSize)) + 4;

    m_gridDim = XMINT3(dimX, dimY, dimZ);
    m_grid.resize(static_cast<size_t>(dimX * dimY * dimZ));

    return true;
}

bool FluidSystem::BuildRenderResources()
{
    m_rootSignature = std::make_unique<RootSignature>();
    if (!m_rootSignature || !m_rootSignature->IsValid())
    {
        return false;
    }

    m_instancedPso = std::make_unique<PipelineState>();
    m_ssfrPso = std::make_unique<PipelineState>();
    if (!m_instancedPso || !m_ssfrPso)
    {
        return false;
    }

    auto layout = CreateInputLayout();
    m_instancedPso->SetInputLayout(layout);
    m_instancedPso->SetRootSignature(m_rootSignature->Get());
    m_instancedPso->SetVS(L"ParticleVS.cso");
    m_instancedPso->SetPS(L"ParticlePS.cso");
    m_instancedPso->SetDepthStencilFormat(DXGI_FORMAT_D32_FLOAT);
    m_instancedPso->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    if (!m_instancedPso->IsValid())
    {
        return false;
    }

    m_ssfrPso->SetInputLayout(layout);
    m_ssfrPso->SetRootSignature(m_rootSignature->Get());
    m_ssfrPso->SetVS(L"ParticleVS.cso");
    m_ssfrPso->SetPS(L"FluidSSFRPS.cso");
    m_ssfrPso->SetDepthStencilFormat(DXGI_FORMAT_D32_FLOAT);
    m_ssfrPso->Create(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    if (!m_ssfrPso->IsValid())
    {
        return false;
    }

    MeshData sphere = CreateLowPolySphere(1.0f, 2);

        struct MeshVertex
    {
        XMFLOAT3 position;
        XMFLOAT3 normal;
    };

    std::vector<MeshVertex> vertices;
    vertices.reserve(sphere.vertices.size());
    for (auto& v : sphere.vertices)
    {
        MeshVertex mv{};
        mv.position = v.Position;
        mv.normal = v.Normal;
        vertices.push_back(mv);
    }

    m_meshVB = std::make_unique<VertexBuffer>(vertices.size() * sizeof(MeshVertex), sizeof(MeshVertex), vertices.data());
    m_meshIB = std::make_unique<IndexBuffer>(sphere.indices.size() * sizeof(uint32_t), sphere.indices.data());
    m_indexCount = static_cast<UINT>(sphere.indices.size());

    for (UINT i = 0; i < Engine::FRAME_BUFFER_COUNT; ++i)
    {
        m_cameraCB[i] = std::make_unique<ConstantBuffer>(sizeof(CameraConstants));
    }

    return m_meshVB && m_meshVB->IsValid() && m_meshIB && m_meshIB->IsValid();
}

bool FluidSystem::BuildInstanceBuffer()
{
    m_instanceData.resize(m_particles.size());
    m_instanceVB = std::make_unique<VertexBuffer>(m_particles.size() * sizeof(InstanceData), sizeof(InstanceData), nullptr);
    return m_instanceVB && m_instanceVB->IsValid();
}

void FluidSystem::StepMLSMPM(float deltaTime)
{
    // ※MLS-MPM主処理: 粒子から格子へ写像し、格子演算後に粒子へ戻す
    for (auto& node : m_grid)
    {
        node.mass = 0.0f;
        node.velocity = XMFLOAT3(0.0f, 0.0f, 0.0f);
    }

    TransferParticlesToGrid();
    UpdateGrid(deltaTime);
    TransferGridToParticles(deltaTime);
}

void FluidSystem::TransferParticlesToGrid()
{
    // ※粒子→格子: 二次Bスプライン重みで運動量を積算する

    for (const Particle& p : m_particles)
    {
        XMVECTOR pos = XMLoadFloat3(&p.position);
        XMVECTOR rel = XMVectorSubtract(pos, XMLoadFloat3(&m_gridMin));
        XMVECTOR coord = XMVectorScale(rel, 1.0f / m_cellSize);
        coord = XMVectorSubtract(coord, XMVectorReplicate(0.5f));

        XMFLOAT3 coordF{};
        XMStoreFloat3(&coordF, coord);

        int baseX = static_cast<int>(std::floor(coordF.x));
        int baseY = static_cast<int>(std::floor(coordF.y));
        int baseZ = static_cast<int>(std::floor(coordF.z));

        float fx = coordF.x - baseX;
        float fy = coordF.y - baseY;
        float fz = coordF.z - baseZ;

        float wx[3] = {
            QuadraticWeight(fx + 1.0f),
            QuadraticWeight(fx),
            QuadraticWeight(fx - 1.0f)
        };
        float wy[3] = {
            QuadraticWeight(fy + 1.0f),
            QuadraticWeight(fy),
            QuadraticWeight(fy - 1.0f)
        };
        float wz[3] = {
            QuadraticWeight(fz + 1.0f),
            QuadraticWeight(fz),
            QuadraticWeight(fz - 1.0f)
        };

        XMMATRIX C = LoadFloat3x3(p.affine);

        for (int dz = 0; dz < 3; ++dz)
        {
            int gz = baseZ + dz;
            if (gz < 0 || gz >= m_gridDim.z) continue;
            for (int dy = 0; dy < 3; ++dy)
            {
                int gy = baseY + dy;
                if (gy < 0 || gy >= m_gridDim.y) continue;
                for (int dx = 0; dx < 3; ++dx)
                {
                    int gx = baseX + dx;
                    if (gx < 0 || gx >= m_gridDim.x) continue;

                    float weight = wx[dx] * wy[dy] * wz[dz];
                    GridNode& node = GridAt(gx, gy, gz);
                    XMFLOAT3 cellPos = GridCellCenter(gx, gy, gz);
                    XMVECTOR diff = XMVectorSubtract(XMLoadFloat3(&cellPos), pos);
                    XMVECTOR momentum = XMVectorAdd(XMLoadFloat3(&p.velocity), XMVector3Transform(diff, C));

                    XMFLOAT3 contrib{};
                    XMStoreFloat3(&contrib, XMVectorScale(momentum, weight * p.mass));

                    node.velocity.x += contrib.x;
                    node.velocity.y += contrib.y;
                    node.velocity.z += contrib.z;
                    node.mass += weight * p.mass;
                }
            }
        }
    }
}

void FluidSystem::UpdateGrid(float deltaTime)
{
    // ※格子演算: 重力付加と境界処理で安定化する

    for (int z = 0; z < m_gridDim.z; ++z)
    {
        for (int y = 0; y < m_gridDim.y; ++y)
        {
            for (int x = 0; x < m_gridDim.x; ++x)
            {
                GridNode& node = GridAt(x, y, z);
                if (node.mass <= 0.0f)
                {
                    continue;
                }

                float invMass = 1.0f / node.mass;
                node.velocity.x *= invMass;
                node.velocity.y *= invMass;
                node.velocity.z *= invMass;

                node.velocity.x += m_gravity.x * deltaTime;
                node.velocity.y += m_gravity.y * deltaTime;
                node.velocity.z += m_gravity.z * deltaTime;

                XMFLOAT3 cellPos = GridCellCenter(x, y, z);

                // 境界衝突では速度を反転させて流体が箱から漏れないようにする
                if (cellPos.x < m_bounds.min.x + m_cellSize && node.velocity.x < 0.0f) node.velocity.x = 0.0f;
                if (cellPos.x > m_bounds.max.x - m_cellSize && node.velocity.x > 0.0f) node.velocity.x = 0.0f;
                if (cellPos.y < m_bounds.min.y + m_cellSize && node.velocity.y < 0.0f) node.velocity.y = 0.0f;
                if (cellPos.y > m_bounds.max.y - m_cellSize && node.velocity.y > 0.0f) node.velocity.y = 0.0f;
                if (cellPos.z < m_bounds.min.z + m_cellSize && node.velocity.z < 0.0f) node.velocity.z = 0.0f;
                if (cellPos.z > m_bounds.max.z - m_cellSize && node.velocity.z > 0.0f) node.velocity.z = 0.0f;
            }
        }
    }
}

void FluidSystem::TransferGridToParticles(float deltaTime)
{
    // ※格子→粒子: 格子速度を補間して粒子を更新し、C行列も再構築する

    for (Particle& p : m_particles)
    {
        XMVECTOR pos = XMLoadFloat3(&p.position);
        XMVECTOR rel = XMVectorSubtract(pos, XMLoadFloat3(&m_gridMin));
        XMVECTOR coord = XMVectorScale(rel, 1.0f / m_cellSize);
        coord = XMVectorSubtract(coord, XMVectorReplicate(0.5f));

        XMFLOAT3 coordF{};
        XMStoreFloat3(&coordF, coord);

        int baseX = static_cast<int>(std::floor(coordF.x));
        int baseY = static_cast<int>(std::floor(coordF.y));
        int baseZ = static_cast<int>(std::floor(coordF.z));

        float fx = coordF.x - baseX;
        float fy = coordF.y - baseY;
        float fz = coordF.z - baseZ;

        float wx[3] = {
            QuadraticWeight(fx + 1.0f),
            QuadraticWeight(fx),
            QuadraticWeight(fx - 1.0f)
        };
        float wy[3] = {
            QuadraticWeight(fy + 1.0f),
            QuadraticWeight(fy),
            QuadraticWeight(fy - 1.0f)
        };
        float wz[3] = {
            QuadraticWeight(fz + 1.0f),
            QuadraticWeight(fz),
            QuadraticWeight(fz - 1.0f)
        };

        float gxGrad[3] = {
            QuadraticGradient(fx + 1.0f),
            QuadraticGradient(fx),
            QuadraticGradient(fx - 1.0f)
        };
        float gyGrad[3] = {
            QuadraticGradient(fy + 1.0f),
            QuadraticGradient(fy),
            QuadraticGradient(fy - 1.0f)
        };
        float gzGrad[3] = {
            QuadraticGradient(fz + 1.0f),
            QuadraticGradient(fz),
            QuadraticGradient(fz - 1.0f)
        };

        XMVECTOR velocity = XMVectorZero();
        XMMATRIX C = XMMatrixIdentity();
        XMFLOAT3X3 accum{};

        for (int dz = 0; dz < 3; ++dz)
        {
            int gz = baseZ + dz;
            if (gz < 0 || gz >= m_gridDim.z) continue;
            for (int dy = 0; dy < 3; ++dy)
            {
                int gy = baseY + dy;
                if (gy < 0 || gy >= m_gridDim.y) continue;
                for (int dx = 0; dx < 3; ++dx)
                {
                    int gx = baseX + dx;
                    if (gx < 0 || gx >= m_gridDim.x) continue;

                    float weight = wx[dx] * wy[dy] * wz[dz];
                    const GridNode& node = GridAt(gx, gy, gz);
                    if (node.mass <= 0.0f)
                    {
                        continue;
                    }

                    XMVECTOR nodeVel = XMLoadFloat3(&node.velocity);
                    velocity = XMVectorAdd(velocity, XMVectorScale(nodeVel, weight));

                    XMFLOAT3 cellPos = GridCellCenter(gx, gy, gz);
                    XMVECTOR diff = XMVectorSubtract(XMLoadFloat3(&cellPos), pos);

                    float wGradX = gxGrad[dx] * wy[dy] * wz[dz];
                    float wGradY = wx[dx] * gyGrad[dy] * wz[dz];
                    float wGradZ = wx[dx] * wy[dy] * gzGrad[dz];

                    XMFLOAT3 grad(wGradX, wGradY, wGradZ);
                    XMVECTOR gradV = XMLoadFloat3(&grad);
                    gradV = XMVectorScale(gradV, 1.0f / m_cellSize);
                    XMFLOAT3 gradF;
                    XMStoreFloat3(&gradF, gradV);

                    accum._11 += gradF.x * node.velocity.x;
                    accum._12 += gradF.x * node.velocity.y;
                    accum._13 += gradF.x * node.velocity.z;
                    accum._21 += gradF.y * node.velocity.x;
                    accum._22 += gradF.y * node.velocity.y;
                    accum._23 += gradF.y * node.velocity.z;
                    accum._31 += gradF.z * node.velocity.x;
                    accum._32 += gradF.z * node.velocity.y;
                    accum._33 += gradF.z * node.velocity.z;
                }
            }
        }

        XMStoreFloat3(&p.velocity, velocity);
        p.previous = p.position;
        XMFLOAT3 displacement;
        XMStoreFloat3(&displacement, XMVectorScale(velocity, deltaTime));
        p.position.x += displacement.x;
        p.position.y += displacement.y;
        p.position.z += displacement.z;
        p.affine = accum;

        ApplyBounds(p);
    }
}

void FluidSystem::ApplyBounds(Particle& particle)
{
    particle.position = Clamp3(particle.position, m_bounds.min, m_bounds.max);
    if (particle.position.x <= m_bounds.min.x || particle.position.x >= m_bounds.max.x)
    {
        particle.velocity.x *= -0.4f;
    }
    if (particle.position.y <= m_bounds.min.y || particle.position.y >= m_bounds.max.y)
    {
        particle.velocity.y *= -0.4f;
    }
    if (particle.position.z <= m_bounds.min.z || particle.position.z >= m_bounds.max.z)
    {
        particle.velocity.z *= -0.4f;
    }
}

void FluidSystem::RebuildSpatialGrid()
{
    m_neighborGrid.Clear();
    m_neighborGrid.SetCellSize(m_particleRadius * 3.0f);
    for (size_t i = 0; i < m_particles.size(); ++i)
    {
        m_neighborGrid.Insert(i, m_particles[i].position);
    }
}

void FluidSystem::ApplyCameraLift(float deltaTime)
{
    // ※視線巻き上げ: レイ付近の粒子に上昇+渦度を加えて流体を掴む

    if (!m_liftRequested)
    {
        m_liftTargetIndex = static_cast<size_t>(-1);
        return;
    }

    if (m_liftTargetIndex == static_cast<size_t>(-1))
    {
        m_liftTargetIndex = FindRayHitParticle(m_liftRayOrigin, m_liftRayDirection);
    }

    if (m_liftTargetIndex == static_cast<size_t>(-1))
    {
        return;
    }

    const float liftRadius = m_particleRadius * 4.0f;
    const float liftStrength = 12.0f;

    std::vector<size_t> neighbors;
    m_neighborGrid.Query(m_particles[m_liftTargetIndex].position, liftRadius, neighbors);

    XMVECTOR origin = XMLoadFloat3(&m_liftRayOrigin);
    XMVECTOR dir = XMVector3Normalize(XMLoadFloat3(&m_liftRayDirection));

    for (size_t index : neighbors)
    {
        Particle& p = m_particles[index];
        XMVECTOR pos = XMLoadFloat3(&p.position);
        XMVECTOR toRay = XMVectorSubtract(pos, origin);
        float t = XMVectorGetX(XMVector3Dot(toRay, dir));
        XMVECTOR foot = XMVectorAdd(origin, XMVectorScale(dir, t));
        XMVECTOR offset = XMVectorSubtract(pos, foot);
        float distance = XMVectorGetX(XMVector3Length(offset));

        if (distance < liftRadius)
        {
            float weight = 1.0f - (distance / liftRadius);
            XMVECTOR lift = XMVectorSet(0.0f, liftStrength * weight * deltaTime, 0.0f, 0.0f);
            XMVECTOR swirl = XMVector3Cross(dir, offset);
            swirl = XMVectorScale(XMVector3Normalize(swirl), liftStrength * 0.4f * weight * deltaTime);

            XMVECTOR newVel = XMVectorAdd(XMLoadFloat3(&p.velocity), XMVectorAdd(lift, swirl));
            XMStoreFloat3(&p.velocity, newVel);
        }
    }
}

size_t FluidSystem::FindRayHitParticle(const XMFLOAT3& origin, const XMFLOAT3& direction) const
{
    XMVECTOR rayOrigin = XMLoadFloat3(&origin);
    XMVECTOR rayDir = XMVector3Normalize(XMLoadFloat3(&direction));

    float closestT = std::numeric_limits<float>::max();
    size_t closestIndex = static_cast<size_t>(-1);

    for (size_t i = 0; i < m_particles.size(); ++i)
    {
        const Particle& p = m_particles[i];
        XMVECTOR pos = XMLoadFloat3(&p.position);
        XMVECTOR toParticle = XMVectorSubtract(pos, rayOrigin);
        float t = XMVectorGetX(XMVector3Dot(toParticle, rayDir));
        if (t < 0.0f)
        {
            continue;
        }
        XMVECTOR closestPoint = XMVectorAdd(rayOrigin, XMVectorScale(rayDir, t));
        float distSq = XMVectorGetX(XMVector3LengthSq(XMVectorSubtract(closestPoint, pos)));
        if (distSq <= m_particleRadius * m_particleRadius && t < closestT)
        {
            closestT = t;
            closestIndex = i;
        }
    }
    return closestIndex;
}

void FluidSystem::UpdateInstanceBuffer()
{
    // ※描画用インスタンス: MLS-MPM結果をGPUバッファへ即時転送する

    for (size_t i = 0; i < m_particles.size(); ++i)
    {
        m_instanceData[i].position = m_particles[i].position;
        m_instanceData[i].radius = m_particleRadius;
    }

    if (m_instanceVB)
    {
        void* mapped = nullptr;
        if (SUCCEEDED(m_instanceVB->GetResource()->Map(0, nullptr, &mapped)))
        {
            memcpy(mapped, m_instanceData.data(), m_instanceData.size() * sizeof(InstanceData));
            m_instanceVB->GetResource()->Unmap(0, nullptr);
        }
    }
}

void FluidSystem::UpdateCameraCB(const Camera& camera)
{
    UINT frameIndex = g_Engine->CurrentBackBufferIndex();
    auto& cb = m_cameraCB[frameIndex];
    if (!cb)
    {
        return;
    }

    CameraConstants* constants = cb->GetPtr<CameraConstants>();
    XMStoreFloat4x4(&constants->world, XMMatrixTranspose(XMMatrixIdentity()));
    XMStoreFloat4x4(&constants->view, XMMatrixTranspose(camera.GetViewMatrix()));
    XMStoreFloat4x4(&constants->proj, XMMatrixTranspose(camera.GetProjMatrix()));
}

D3D12_INPUT_LAYOUT_DESC FluidSystem::CreateInputLayout()
{
    static D3D12_INPUT_ELEMENT_DESC descs[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "POSITION", 1, DXGI_FORMAT_R32G32B32_FLOAT, 1, 0, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32_FLOAT,       1, 12, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
    };
    D3D12_INPUT_LAYOUT_DESC layout{};
    layout.NumElements = _countof(descs);
    layout.pInputElementDescs = descs;
    return layout;
}

FluidSystem::GridNode& FluidSystem::GridAt(int x, int y, int z)
{
    return m_grid[GridIndex(x, y, z)];
}

const FluidSystem::GridNode& FluidSystem::GridAt(int x, int y, int z) const
{
    return m_grid[GridIndex(x, y, z)];
}

size_t FluidSystem::GridIndex(int x, int y, int z) const
{
    return static_cast<size_t>(x + m_gridDim.x * (y + m_gridDim.y * z));
}

XMFLOAT3 FluidSystem::GridCellCenter(int x, int y, int z) const
{
    return XMFLOAT3(
        m_gridMin.x + (x + 0.5f) * m_cellSize,
        m_gridMin.y + (y + 0.5f) * m_cellSize,
        m_gridMin.z + (z + 0.5f) * m_cellSize);
}

