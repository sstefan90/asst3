implementing "../renderer.slang";

import light;

public struct RayMeshIntersectionResult
{
    public RayHitResult closestHit;
    public Triangle hitTriangle;
}

public Optional<RectangularLight.LightHitResult> rayRectangularLightIntersection(Ray ray, RendererUniform uniforms)
{
    Optional<RectangularLight.LightHitResult> closestLightHitResult = Optional<RectangularLight.LightHitResult>();
    for (int i = 0; i < uniforms.rectangularLightCount; i++)
    {
        Optional<RectangularLight.LightHitResult> lightHitResult = uniforms.rectangularLightBuf[i].hit(ray);
        if (lightHitResult.hasValue)
        {
            if (!closestLightHitResult.hasValue || lightHitResult.value.t < closestLightHitResult.value.t)
            {
                closestLightHitResult = lightHitResult;
            }
        }
    }
    return closestLightHitResult;
}

public Optional<RayMeshIntersectionResult> rayTriangles(Ray ray, StructuredBuffer<Triangle> triangles, uint triangleCount)
{

    // find the normal of the plane defined by the triangle
    Optional<RayHitResult> result;
    Triangle hitTriangle;
    float t_smallest = float.maxValue;
    for (int i = 0; i < triangleCount; i++)
    {
        Optional<RayHitResult> hitResult = triangles[i].hit(ray);
        if (hitResult.hasValue) {
            if (hitResult.value.t < t_smallest)
            {
                t_smallest = hitResult.value.t;
                result = hitResult;
                hitTriangle = triangles[i];
            }
        }
    }

    if (t_smallest == float.maxValue)
    {
        return Optional<RayMeshIntersectionResult>();
    }

    return Optional<RayMeshIntersectionResult>(RayMeshIntersectionResult(result.value, hitTriangle));

}

public Optional<RayMeshIntersectionResult> rayMeshIntersection(Ray ray, RendererUniform uniforms, bool occlusionTest = false)
{
    Optional<RayHitResult> result = Optional<RayHitResult>();

    // IMPORTANT CHANGE: We need to store the hit triangle.
    Optional<RayMeshIntersectionResult> allResult = Optional<RayMeshIntersectionResult>();
    //Optional<Triangle> hitTriangle = Optional<Triangle>();
    // IMPORTANT CHANGE: We need to store the hit triangle.

    // TODO: Student implementation starts here.
    float t_min = float.maxValue;

    if (!uniforms.useBVH) {
    
        allResult = rayTriangles(ray, uniforms.triangleBuf, uniforms.triangleCount);
    }
    else {
        Optional<uniforms.bvh.HitResult> res = Optional<uniforms.bvh.HitResult>();

        if (occlusionTest) {
            res = uniforms.bvh.hitAny(ray);
        }
        else {
            res = uniforms.bvh.hit(ray);
        }

        if (res.hasValue) {
            Triangle hitTriangle = res.value.primitive;
            RayHitResult rayHitResult = res.value.rayHitResult;
            allResult = Optional<RayMeshIntersectionResult>(RayMeshIntersectionResult(rayHitResult, hitTriangle));

        }

    }
    
    // TODO: Student implementation ends here.

    //{
    //    allResult = Optional<RayMeshIntersectionResult>(RayMeshIntersectionResult(result.value, hitTriangle.value));
    //}

    return allResult;
}
