implementing "../brdf.slang";

public struct MirrorBRDF
{
    public float3 albedo;

    public float3 evaluate()
    {
        return albedo;
    }

    public float3 sample(float3 rayOut, float3 normal)
    {
        float3 sampleDirection = rayOut;
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);
        float3x3 localToWorldMatrix = transposeMatrix(worldToLocalMatrix);

        // HINT: First convert the reflected ray direction to the local space,
        // then compute the incident ray direction in the local space,
        // and finally convert it to the world space.

        // TODO: Student implementation starts here.
        normal = normalize(normal); // just in case
        float3 normalLocal = mul(worldToLocalMatrix, normal);
        float3 rayOutLocal = mul(worldToLocalMatrix, rayOut);
        float3 rayReflect = -rayOutLocal + 2*dot(rayOutLocal, normalLocal) * normalLocal;
        sampleDirection = mul(localToWorldMatrix, rayReflect);
        sampleDirection = normalize(sampleDirection);



        // TODO: Student implementation ends here.

        return sampleDirection;
    }

    public float pdf(float3 rayIn, float3 rayOut, float3 normal)
    {
        const float EPSILON = 1e-6;
        float pdf = all(abs(rayIn - rayOut) < EPSILON) ? 1.0 : 0.0;
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);

        // TODO: Student implementation starts here.
        float cosTheta = dot(rayIn, normal);
        return max(cosTheta, 0.0);
    }
}
