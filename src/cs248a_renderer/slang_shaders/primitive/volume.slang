implementing "../primitive.slang";
import math;
import texture;

// The size of each marching step.
static float STEP_SIZE = 0.005;
// The density threshold to consider a hit.
static float DENSITY_THRESHOLD = 0.25;
// The delta used for finite difference normal estimation.
static float FINITE_DIFF_DELTA = 0.001;

public struct Volume
{
    public BoundingBox bound;
    public SharedTexture3D tex;
    public float4x4 modelMatrix;
    public float4x4 invModelMatrix;

    public float3 findNormal(float3 point, SharedTexture3DBuffer<float4> buffer) {
        float3 dx = float3(FINITE_DIFF_DELTA, 0.0, 0.0);
        float3 dy = float3(0.0, FINITE_DIFF_DELTA, 0.0);
        float3 dz = float3(0.0, 0.0, FINITE_DIFF_DELTA);

        float3 f1 = float3(0.0);
        float3 f2 = float3(0.0);

        f1[0] = buffer.trilinearSample(tex, getUVW(point + dx))[3];
        f1[1] = buffer.trilinearSample(tex, getUVW(point + dy))[3];
        f1[2] = buffer.trilinearSample(tex, getUVW(point + dz))[3];

        f2[0] = buffer.trilinearSample(tex, getUVW(point - dx))[3];
        f2[1] = buffer.trilinearSample(tex, getUVW(point - dy))[3];
        f2[2] = buffer.trilinearSample(tex, getUVW(point - dz))[3];


        return -normalize(f1 - f2);
    }

    /**
     * Test ray-volume bounding box intersection.
     * @param ray The ray to test against the volume.
     * @param volumeTexBuf The texture buffer contains all the volumetrically represented models in the scene .
     * @return An Optional containing RayHitResult if the ray hits the volume, otherwise an empty Optional.
     */
    public Optional<RayHitResult> hit(Ray ray, SharedTexture3DBuffer<float4> volumeTexBuf)
    {

        float step = 0.0;
        float occupancy = 0.0;
        float3 boxSize = this.bound.pMax - this.bound.pMin;

        float2 bounds = this.bound.hit(ray);
        step = bounds[0];

        if (bounds.x < 0) {
            return Optional<RayHitResult>();
        } 

        float3 curr_point = ray.origin + step * ray.direction;
        while (step < bounds[1]) {

            // test intersection

            float3 uvw = getUVW(curr_point);
            if (any(uvw < 0.0) || any(uvw > 1.0)) {
                step += STEP_SIZE;
                curr_point = ray.origin + step * ray.direction;
                continue;
            }
            occupancy = volumeTexBuf.trilinearSample(tex,uvw )[3];

            if (occupancy >= DENSITY_THRESHOLD) {
                RayHitResult result = { step, findNormal(curr_point, volumeTexBuf) };
                return result;
            }
            step += STEP_SIZE;
            curr_point = ray.origin + step * ray.direction;
        }

        
        //for (float t = 0.0; t < 100.0; t += 0.001) {
        //    float3 p = ray.origin + t * ray.direction;
        //    float3 uvw = getUVW(p);

        //    if (all(uvw >= 0.0) && all(uvw <= 1.0)) {
        //        
        //        RayHitResult result = { t, findNormal(p, volumeTexBuf) };
        //        return result;
        //    }
        //}
        //return none;

        

        return Optional<RayHitResult>();
    }

    /**
     * Get the uvw coordinates of a point in local space.
     * @param p The point in world space.
     * @return The uvw coordinates in [0, 1].
     */
    public float3 getUVW(float3 p)
    {
        // Compute uvw coordinates in [0, 1].
        float3 minBound = this.bound.pMin;
        float3 maxBound = this.bound.pMax;
        return (p - minBound) / (maxBound - minBound);
    }
};
