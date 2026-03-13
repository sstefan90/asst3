implementing "../brdf.slang";

public struct GlassBRDF
{
    public float3 albedo;
    public float ior;

    public float3 evaluate()
    {
        return albedo;
    }

    public float3 sample(float3 rayOut, float3 normal)
    {
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);
        float3x3 localToWorldMatrix = transposeMatrix(worldToLocalMatrix);
        
        float3 normalLocal = normalize(mul(worldToLocalMatrix, normal));
        float3 rayOutLocal = normalize(mul(worldToLocalMatrix, rayOut));

        float cosTheta = dot(rayOutLocal, normalLocal);
        float3 sampleDirection = float3(0.0);
        
        bool entering = cosTheta > 0.0;
        float n1 = entering ? 1.0 : ior;
        float n2 = entering ? ior : 1.0;
        float eta = n1 / n2;
        
        float3 N = entering ? normalLocal : -normalLocal;
        float cosThetaI = abs(cosTheta);
        
        float sin2ThetaT = eta * eta * (1.0 - cosThetaI * cosThetaI);
        
        float3 rayAns = float3(0.0);
        
        if (sin2ThetaT >= 1.0) 
        {
            rayAns = rayOutLocal - 2.0 * dot(rayOutLocal, N) * N;
        } 
        else {

            float cosThetaT = sqrt(1.0 - sin2ThetaT);
            rayAns = -eta * rayOutLocal + (eta * cosThetaI - cosThetaT) * N;
        }

        rayAns = mul(localToWorldMatrix, rayAns);
        sampleDirection = normalize(rayAns);
        return sampleDirection;
    }

    public float pdf(float3 rayIn, float3 rayOut, float3 normal)
    {
        // For dielectric materials, the PDF is 1.0 for the single direction
        // (either refracted or reflected due to total internal reflection)
        return 1.0;
    }
}
