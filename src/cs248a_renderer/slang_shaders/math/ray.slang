implementing "../math.slang";

// A ray represented by an origin, a direction, and a t range along the ray.
public struct Ray
{
    public float3 origin;
    public float3 direction;
    // A point on the ray is given by origin + t * direction
    // tRange represents the valid interval of talong the ray: [tRange.x, tRange.y]
    public float2 tRange;

    public __init(float3 origin, float3 direction, float2 tRange = float2(0.0, float.maxValue))
    {
        this.origin = origin;
        this.direction = direction;
        this.tRange = tRange;
    }

    /**
     * Transform the ray by a 4x4 matrix.
     * @param matrix The transformation matrix.
     * @return The transformed ray.
     */
    public Ray transform(float4x4 matrix)
    {

        float4 origin_new = mul(matrix, float4(origin, 1.0));
        float4 direction_new = mul(matrix, float4(direction, 0.0));

        Ray ray_t;
        ray_t.origin = (origin_new / origin_new.w).xyz;
        ray_t.direction = normalize(direction_new.xyz);
        ray_t.tRange = this.tRange;
        return ray_t;

    }
};

public struct RayHitResult
{
    // The parameter t at which the ray intersects the primitive.
    public float t;
    // The normal of the primitive at the intersection point.
    public float3 normal;
};

public interface IRayCastable
{
    public Optional<RayHitResult> hit(Ray ray);
}
