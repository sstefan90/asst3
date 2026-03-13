implementing "../model.slang";
import math;

public struct BVHNode
{
    public BoundingBox bound;
    public int left;
    public int right;
    public int primLeft;
    public int primRight;
    public int depth;

    public __init(BoundingBox bound, int left = -1, int right = -1, int primLeft = -1, int primRight = -1, int depth = -1)
    {
        this.bound = bound;
        this.left = left;
        this.right = right;
        this.primLeft = primLeft;
        this.primRight = primRight;
        this.depth = depth;
    }

    public float2 hit(Ray ray)
    {
        return this.bound.hit(ray);
    }
}

public struct BVH<T>
    where T : IRayCastable
{
    public struct HitResult
    {
        public T primitive;
        public RayHitResult rayHitResult;
    }

    // Slang buffer of BVHNode copied from Python.
    public RWStructuredBuffer<BVHNode> nodes;
    // Maximum number of nodes in the BVH, in this case, it'll be the size of the nodes buffer.
    public uint maxNodes;
    // Slang buffer of primitives represented by the BVH.
    public RWStructuredBuffer<T> primitives;
    // Number of primitives in the primitives buffer.
    public uint numPrimitives;

    public __init(RWStructuredBuffer<BVHNode> nodes, RWStructuredBuffer<T> primitives, uint numPrimitives, uint maxNodes)
    {
        this.nodes = nodes;
        this.primitives = primitives;
        this.numPrimitives = numPrimitives;
        this.maxNodes = maxNodes;
    }

    struct HitStack<int N>
    {
        uint[N] nodeIndices;
        uint top;

        __init()
        {
            this.top = 0;
        }

        [mutating]
        void push(uint nodeIndex)
        {
            this.nodeIndices[this.top] = nodeIndex;
            this.top += 1;
        }

        [mutating]
        uint pop()
        {
            this.top -= 1;
            return this.nodeIndices[this.top];
        }

        bool isEmpty()
        {
            return this.top == 0;
        }

        uint size()
        {
            return this.top;
        }
    }

    // BVH traversal to find the closest hit.
    public Optional<HitResult> hit(Ray ray)
    {
        // Initialize traversal.
        Optional<HitResult> hitResult = Optional<HitResult>();
        HitStack<64> stack;

        stack.push(0);
        float t_min = float.maxValue;
        while (!stack.isEmpty()) {
            uint node_idx = stack.pop();
            BVHNode node = nodes[node_idx];

            float2 box_hit = node.bound.hit(ray);

            if ((box_hit[0] == -1) || (box_hit[0] > t_min)) {
                continue;
            }

            // check if it's a leaf
            if (node.left == -1) {
                for (uint i = node.primLeft; i < node.primRight; i++) {
                    T prim = primitives[i];
                    Optional<RayHitResult> result = prim.hit(ray);

                    if (result.hasValue) {
                        RayHitResult res = result.value;

                        if (res.t < t_min) {
                            t_min = res.t;

                            HitResult candidate;
                            candidate.primitive = prim;
                            candidate.rayHitResult = res;
                            hitResult = Optional<HitResult>(candidate);
                        }

                    }
                }
            }

            else {
                if (node.right != -1) stack.push(node.right);
                if (node.left != -1) stack.push(node.left);
            }
        }

        return hitResult;

    }



    // BVH traversal to find ANY hit!
    public Optional<HitResult> hitAny(Ray ray)
    {
        // Initialize traversal.
        Optional<HitResult> hitResult = Optional<HitResult>();
        HitStack<64> stack;

        stack.push(0);
        float t_min = float.maxValue;
        while (!stack.isEmpty()) {
            uint node_idx = stack.pop();
            BVHNode node = nodes[node_idx];

            float2 box_hit = node.bound.hit(ray);

            if ((box_hit[0] == -1) || (box_hit[0] > t_min)) {
                continue;
            }

            // check if it's a leaf
            if (node.left == -1) {
                for (uint i = node.primLeft; i < node.primRight; i++) {
                    T prim = primitives[i];
                    Optional<RayHitResult> result = prim.hit(ray);

                    if (result.hasValue) {
                        RayHitResult res = result.value;
                        HitResult candidate;
                        candidate.primitive = prim;
                        candidate.rayHitResult = res;
                        hitResult = Optional<HitResult>(candidate);

                        return hitResult;
                        }

                    }
                }

            else {
                if (node.right != -1) stack.push(node.right);
                if (node.left != -1) stack.push(node.left);
                }
        }

        return hitResult;

    }
}
