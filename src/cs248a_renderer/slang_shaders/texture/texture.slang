implementing "../texture.slang";

public struct SharedTexture2DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture2D tex, uint2 texelCoord)
    {
        uint2 texSize = tex.size;
        uint offset = tex.offset;
        uint index = texelCoord.y * texSize.x + texelCoord.x;
        return buffer[offset + index];
    }

    public T pointSample(SharedTexture2D tex, float2 uv)
    {

        uint2 texelCoord = uint2(round(uv * float2(tex.size[0]-1, tex.size[1]-1)));
        T outV = getBufferValue(tex, texelCoord);
        return outV;
    }

    public T bilinearSample(SharedTexture2D tex, float2 uv)
    {
        //T outV = T(0.0);
        // TODO: Student implementation starts here.
        float2 coord_float = uv * float2(tex.size) -0.5;
        int2 voxelCoord = int2(floor(coord_float)); //
        float2 frac =  coord_float - float2(voxelCoord);

        // TODO: Student implementation ends here.

        uint x0 = uint(clamp(voxelCoord.x + 0, 0, int(tex.size[0]-1)));
        uint y0 = uint(clamp(voxelCoord.y + 0, 0, int(tex.size[1]-1)));

        uint x1 = uint(clamp(voxelCoord.x + 1, 0, int(tex.size[0]-1)));
        uint y1 = uint(clamp(voxelCoord.y + 1, 0, int(tex.size[1]-1)));

        
        
        T v00 = getBufferValue(tex, uint2(x0, y0));
        T v10 = getBufferValue(tex, uint2(x1, y0));
        T v01 = getBufferValue(tex, uint2(x0, y1));
        T v11 = getBufferValue(tex, uint2(x1, y1));

        T v0 = v00 * T(1.0 - frac.x) + v10 * T(frac.x);
        T v1 = v01 * T(1.0 - frac.x) +  v11 * T(frac.x);

        T outV = v0 * T(1.0 - frac.y) + v1 * T(frac.y);
        return outV;
    }

    public T trilinearSample(SharedTexture2D[] mipmapLevels, float2 uv, float d, uint totalLevels)
    {
        d = clamp(d, 0.0, float(totalLevels - 1));
        uint level_low = uint(floor(d));
        uint level_high = uint(ceil(d));
        float frac = d - float(level_low);

        T low = bilinearSample(mipmapLevels[level_low], uv);
        T high = bilinearSample(mipmapLevels[level_high], uv);

        return low * T(1.0 - frac) + high * T(frac);
    }
}

public struct SharedTexture2D
{
    public uint2 size;
    public uint offset;
}

public struct SharedTexture3DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture3D tex, uint3 voxelCoord)
    {
        uint3 texSize = tex.size;
        uint offset = tex.offset;
        uint index = voxelCoord.z * texSize.y * texSize.x +
                     voxelCoord.y * texSize.x +
                     voxelCoord.x;
        return buffer[offset + index];
    }

    /**
     * Sample the texture at the given uvw coordinates using nearest neighbor interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     */
    public T pointSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        uint3 voxelCoord = uint3(round(uvw * tex.size));

        return getBufferValue(tex, voxelCoord);

        // TODO: Student implementation ends here.
    }

    public T bilinearSample(T v00, T v10, T v01, T v11, float2 frac) {
        T v0 = v00 * T(1.0 - frac.x) + v10 * T(frac.x);//lerp(v00, v10, frac.x);
        T v1 = v01 * T(1.0 - frac.x) +  v11 * T(frac.x);

        return v0 * T(1 - frac.y) + v1 * T(frac.y);

    }

    /**
     * Sample the texture at the given uvw coordinates using trilinear interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     **/
    public T trilinearSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        float3 coord_float = uvw * float3(tex.size);
        uint3 voxelCoord = uint3(floor(coord_float)); //
        float3 frac =  coord_float - float3(voxelCoord);
        T result;

        T v000 = getBufferValue(tex, voxelCoord + int3(0, 0, 0));
        T v100 = getBufferValue(tex, voxelCoord + int3(1, 0, 0));
        T v010 = getBufferValue(tex, voxelCoord + int3(0, 1, 0));
        T v110 = getBufferValue(tex, voxelCoord + int3(1, 1, 0));
        T v001 = getBufferValue(tex, voxelCoord + int3(0, 0, 1));
        T v101 = getBufferValue(tex, voxelCoord + int3(1, 0, 1));
        T v011 = getBufferValue(tex, voxelCoord + int3(0, 1, 1));
        T v111 = getBufferValue(tex, voxelCoord + int3(1, 1, 1));

        T bot = bilinearSample(v000, v100, v010, v110, frac.xy);
        T top = bilinearSample(v001, v101, v011, v111, frac.xy);

        


        
        return bot * T(1 - frac.z) + top * T(frac.z);

        // TODO: Student implementation ends here.
    }
}

public struct SharedTexture3D
{
    public uint3 size;
    public uint offset;
}

