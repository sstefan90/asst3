implementing "../brdf.slang";

import utils;

public struct LambertianBRDF
{
    public float3 albedo;
    public bool useCosineWeightedSampling;

    public float3 evaluate()
    {
        float3 fr = this.albedo;

        fr = this.albedo * INV_PI;

        return fr;
    }

    public float3 sample(float3 rayOut, float3 normal, float2 randomEpsilon)
    {
        float3 sampleDirection = rayOut;
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);
        float3x3 localToWorldMatrix = transposeMatrix(worldToLocalMatrix);

        // HINT: First convert the reflected ray direction to the local space,
        // then compute the incident ray direction in the local space,
        // and finally convert it to the world space.

        if (this.useCosineWeightedSampling)
        {
            // TODO: Student implementation starts here.
            float2 xy = randomEpsilon;
            float phi = 2.0 * PI * randomEpsilon.x;
            float r = sqrt(randomEpsilon.y);
            float x = r * cos(phi);
            float z = r * sin(phi);
            float y = sqrt(max(0.0, 1.0 - x * x - z * z));

            sampleDirection = float3(x, y, z);

            sampleDirection = normalize(mul(localToWorldMatrix, sampleDirection));

            // TODO: Student implementation ends here.
        }
        else // uniform sampling, use uniform sampling by default
        {
            // sample direction
            float phi = 2.0 * PI * randomEpsilon.x;
            float cosTheta = randomEpsilon.y;
            float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
            float x = sinTheta * cos(phi);
            float z = sinTheta * sin(phi);
            float y = cosTheta;

            sampleDirection.x = x;
            sampleDirection.y = y;
            sampleDirection.z = z;

            sampleDirection = normalize(mul(localToWorldMatrix, sampleDirection));
            //sampleDirection = normalize(mul( sampleDirection, localToWorldMatrix));
        }

        return sampleDirection;
    }

    public float pdf(float3 rayIn, float3 rayOut, float3 normal)
    {
        float EPSILON = 1e-6;
        float pdf = all(abs(rayIn - rayOut) < EPSILON) ? 1.0 : 0.0;
        rayIn = normalize(rayIn);
        rayOut = normalize(rayOut);
        if (this.useCosineWeightedSampling)
        {
            float cosTheta = dot(rayIn, normal);
            if (cosTheta <= EPSILON)
            {
                return float.maxValue;
            }
            pdf = cosTheta * INV_PI;

            
        }
        else // uniform sampling, use uniform sampling by default
        {
            float cosTheta = dot(rayIn, normal);
            if (cosTheta <= EPSILON)
            {
                return float.maxValue;
            }



            pdf =  0.5 * INV_PI;

        }

        return pdf;
    }
}
