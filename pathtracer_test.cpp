#include <malloc.h>

#include "kt_bvh.h"

#define FAST_OBJ_IMPLEMENTATION
#include "fast_obj.h"

int main(int argc, char** _argv)
{
	fastObjMesh* obj_mesh = fast_obj_read("models/kitten.obj");

	// Build linear index array
	uint32_t* pos_indices = (uint32_t*)malloc(sizeof(uint32_t) * obj_mesh->face_count * 3);

	uint32_t* pnext = pos_indices;

	for (uint32_t i = 0; i < obj_mesh->face_count; ++i)
	{
		assert(obj_mesh->face_vertices[i] == 3);
		*pnext++ = obj_mesh->indices[i * 3].p;
		*pnext++ = obj_mesh->indices[i * 3 + 1].p;
		*pnext++ = obj_mesh->indices[i * 3 + 2].p;
	}

	kt_bvh::TriMesh tri_mesh;
	tri_mesh.set_indices(pos_indices, obj_mesh->face_count);
	tri_mesh.set_vertices(obj_mesh->positions, sizeof(float[3]), obj_mesh->position_count);

	kt_bvh::IntermediateBVH2* bvh2 = kt_bvh::bvh2_build_intermediate(&tri_mesh, 1, kt_bvh::BVH2BuildDesc::defult_desc());

	uint32_t const num_nodes = kt_bvh::bvh2_intermediate_num_nodes(bvh2);
	kt_bvh::BVH2Node* bvh2nodes = (kt_bvh::BVH2Node*)malloc(num_nodes * sizeof(kt_bvh::BVH2Node));

	kt_bvh::bvh2_intermediate_to_flat(bvh2, bvh2nodes, num_nodes);



	kt_bvh::bvh2_free_intermediate(bvh2);
	free(pos_indices);
}