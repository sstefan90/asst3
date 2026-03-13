implementing "../renderer.slang";

import utils;

public float2 getShiftedTexUV(float2 uv, RendererUniform uniforms, Triangle triangle, float2 delta, float2 tex_uv)
{
    float2 shifted_uv = uv + delta;
    Ray ray = uniforms.camera.generateRay(shifted_uv);
    Optional<RayHitResult> hitResult = triangle.hit(ray);
    float2 result = tex_uv;
    if (hitResult.hasValue)
    {
        float3 hitPoint = ray.origin + hitResult.value.t * ray.direction;
        float3 barycentricCoord = triangle.calculateBarycentricCoord(hitPoint);
        return triangle.getUV(barycentricCoord);
    }

    Optional<RayMeshIntersectionResult> rayMeshIntersectionResult_dx = rayMeshIntersection(ray, uniforms);
    if (rayMeshIntersectionResult_dx.hasValue)
    {
        Triangle hitTriangle = rayMeshIntersectionResult_dx.value.hitTriangle;
        if (hitTriangle.getMaterialId() == triangle.getMaterialId())
        {
            float3 hitPoint = ray.origin + rayMeshIntersectionResult_dx.value.closestHit.t * ray.direction;
            float3 barycentricCoord = hitTriangle.calculateBarycentricCoord(hitPoint);
            return hitTriangle.getUV(barycentricCoord);
        }
    }

    return tex_uv;
}

public float getLevel(float2 uv, RendererUniform uniforms, Triangle triangle, float2 tex_uv)
{
    

    float delta = 1e-6;

    //launch 4 rays to compute the level of detail
    // + dx
    Ray ray_dx = uniforms.camera.generateRay(uv + float2(delta, 0.0));
    // - dx
    Ray ray_minus_dx = uniforms.camera.generateRay(uv - float2(delta, 0.0));
    // + dy
    Ray ray_dy = uniforms.camera.generateRay(uv + float2(0.0, delta));
    // - dy
    Ray ray_minus_dy = uniforms.camera.generateRay(uv - float2(0.0, delta));

    Optional<RayHitResult> res_dx = triangle.hit(ray_dx);
    Optional<RayHitResult> res_minus_dx = triangle.hit(ray_minus_dx);
    Optional<RayHitResult> res_dy = triangle.hit(ray_dy);
    Optional<RayHitResult> res_minus_dy = triangle.hit(ray_minus_dy);

    float2 uv_dx = getTextCoord(triangle, ray_dx, res_dx, tex_uv);
    float2 uv_minus_dx = getTextCoord(triangle, ray_minus_dx, res_minus_dx, tex_uv);
    float2 uv_dy = getTextCoord(triangle, ray_dy, res_dy, tex_uv);
    float2 uv_minus_dy = getTextCoord(triangle, ray_minus_dy, res_minus_dy, tex_uv);

    float2 duv_dx = (uv_dx - uv_minus_dx) / (2.0*delta);
    float2 duv_dy = (uv_dy - uv_minus_dy) / (2.0*delta);

    float px = length(duv_dx);
    float py = length(duv_dy);
    float level = log2(max(px, py));


    return level;
}

float2 getTextCoord(Triangle triangle, Ray ray, Optional<RayHitResult> hitResult, float2 tex_uv){
    if (!hitResult.hasValue) {
        return tex_uv;
    }

    float3 hitPoint = ray.origin + hitResult.value.t * ray.direction;
    float3 barycentricCoord = triangle.calculateBarycentricCoord(hitPoint);
    float2 tex_uv = triangle.getUV(barycentricCoord);
    return tex_uv;
}

public PhysicsBasedParameters getHitPointParameters(float2 uv, RendererUniform uniforms, Triangle triangle, float3 normal, float3 barycentricCoord, float3 hitPoint)
{
    float2 tex_uv = triangle.getUV(barycentricCoord);
    float level = getLevel(uv, uniforms, triangle, tex_uv);
    uint materialId = triangle.materialId + triangle.offset;
    PhysicsBasedMaterial physicsBasedMaterial = uniforms.physicsBasedMaterialBuf[materialId];
    PhysicsBasedParameters parameters = sampleMaterial(physicsBasedMaterial, uniforms.physicsBasedMaterialTextureBuf, tex_uv, level, uniforms.visualizeLevelOfDetail);
    if (uniforms.visualizeBarycentricCoords)
    {
        parameters.albedo = barycentricCoord;
    }
    if (uniforms.visualizeTexUV)
    {
        parameters.albedo = float3(tex_uv, 0.0);
    }
    return parameters;
}
