#include "intersections.h"
#include "stack.h"

__device__ int bvhCountHits(    
    const Geom &mesh,
    Ray r
) 
{
    Ray r_ws = r;   

    r.origin = glm::vec3(mesh.inverseTransform * glm::vec4(r.origin, 1.0f));
    r.direction = glm::vec3(mesh.inverseTransform * glm::vec4(r.direction, 0.0f));

    r.inv_direction.x = __frcp_rn(r.direction.x);
    r.inv_direction.y = __frcp_rn(r.direction.y);
    r.inv_direction.z = __frcp_rn(r.direction.z);

    r.sign.x = (r.inv_direction.x < 0.0f) ? 1 : 0;
    r.sign.y = (r.inv_direction.y < 0.0f) ? 1 : 0;
    r.sign.z = (r.inv_direction.z < 0.0f) ? 1 : 0;

    // at this point, r is now in object space

    BVHNode* bvh = mesh.mesh.bvh.dev_bvh;

    int hitCount = 0;
    
    Stack dfs_stack;
    dfs_stack.init();
    dfs_stack.push(0);
    hitCount++;

    while (!dfs_stack.isEmpty()) {
        int bvh_idx = dfs_stack.pop();

        if (!bvh[bvh_idx].isLeaf) {
            int left = bvh[bvh_idx].left_child;
            int right = bvh[bvh_idx].left_child + 1;

            float tmin_left, tmin_right;
            bool hitLeft = bvh[left].isLeaf || bvh[left].box.RayBoxInterection(r, tmin_left);
            bool hitRight = bvh[right].isLeaf || bvh[right].box.RayBoxInterection(r, tmin_right);

            if (hitLeft && hitRight) {
                dfs_stack.push(tmin_left <= tmin_right ? right : left);
                dfs_stack.push(tmin_left <= tmin_right ? left : right);
                hitCount += 2;
            }
            else if (hitLeft) {
                dfs_stack.push(left);
                hitCount++;
            }
            else if (hitRight) {
                dfs_stack.push(right);
                hitCount++;
            }
        }
    }

    return hitCount;
}
