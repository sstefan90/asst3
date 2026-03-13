implementing "../light.slang";

import math;
import primitive;
import utils;

public struct RectangularLight
{
    public float3 bottomLeftVertex;
    public float3 bottomEdge;
    public float3 leftEdge;
    public float3 normal;
    public float area;
    public float3 color;
    public float intensity;
    public bool doubleSided;

    public struct LightHitResult
    {
        public float t;
        public float3 normal;
        public float3 radiance;
    }

    public LightSample sample(float3 hitPoint, float2 randomEpsilon)
    {
        float3 randomPointOnLight = this.bottomLeftVertex + this.bottomEdge * randomEpsilon.x + this.leftEdge * randomEpsilon.y;
        float3 toLight = randomPointOnLight - hitPoint;
        float3 lightDirection = normalize(toLight);
        float lightDistance = length(toLight);
        float3 radiance = getRadiance(lightDirection, lightDistance);
        float pdf = getPdf(hitPoint, randomPointOnLight);
        return LightSample(lightDirection, radiance, pdf, lightDistance);
    }

    public float getPdf(float3 hitPoint, float3 pointOnLight)
    {
        const float EPSILON = 1e-6;

        float3 toLight = pointOnLight - hitPoint;
        float distance_2 = dot(toLight, toLight);
        //distance_2 = max(distance_2, 1e-4);
        float3 lightDir = normalize(toLight);
        
        float cos_theta_p = dot(-lightDir, this.normal);
        
        if (cos_theta_p <= 0.0)
        {
            return float.maxValue;
    }

        return distance_2 / (cos_theta_p * this.area);
    }

    

    public float3 getRadiance(float3 lightDirection, float lightDistance)
    {
        return this.color * this.intensity;
    }

    

    public Optional<LightHitResult> hit(Ray ray)
    {
        const float EPSILON = 1e-6;
        
        float denom = dot(ray.direction, this.normal);
        
        if (abs(denom) < EPSILON)
        {
            return Optional<LightHitResult>();
        }
        
        float3 planePoint = this.bottomLeftVertex;
        float t = dot(planePoint - ray.origin, this.normal) / denom;
        
        if (t < ray.tRange.x || t > ray.tRange.y)
        {
            return Optional<LightHitResult>();
        }
        
        float3 hitPoint = ray.origin + t * ray.direction;
        float3 localPoint = hitPoint - this.bottomLeftVertex;
        
        
        float bottomEdgeLengthSq = dot(this.bottomEdge, this.bottomEdge);
        float leftEdgeLengthSq = dot(this.leftEdge, this.leftEdge);
        
        float u = dot(localPoint, this.bottomEdge) / bottomEdgeLengthSq;
        float v = dot(localPoint, this.leftEdge) / leftEdgeLengthSq;

        if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)
        {
            return Optional<LightHitResult>();
        }
        
        float lightDistance = length(hitPoint - ray.origin);
        float3 radiance = getRadiance(ray.direction, lightDistance);
        return Optional<LightHitResult>(LightHitResult(t, this.normal, radiance));
    }
}
