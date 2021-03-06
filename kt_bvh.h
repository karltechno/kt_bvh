#pragma once
#include <stdint.h>

#define KT_BVH_COMPILER_MSVC		(0)
#define KT_BVH_COMPILER_CLANG		(0)
#define KT_BVH_COMPILER_GCC			(0)
#define KT_BVH_SSE					(0)

#if defined(__clang__)
	#undef KT_BVH_COMPILER_CLANG
	#define KT_BVH_COMPILER_CLANG	(1)
#elif defined(__GNUG__)
	#undef KT_BVH_COMPILER_GCC	
	#define KT_BVH_COMPILER_GCC		(1)
#elif defined(_MSC_VER)
	#undef KT_BVH_COMPILER_MSVC
	#define KT_BVH_COMPILER_MSVC	(1)
#else
	#error Compiler not supported.
#endif

#if defined(__SSE__) || defined(_M_X64)
	#undef KT_BVH_SSE
	#define KT_BVH_SSE (1)
#endif

#if (KT_BVH_COMPILER_CLANG || KT_BVH_COMPILER_GCC)
	#define KT_BVH_UNREACHABLE __builtin_unreachable();
#elif KT_BVH_COMPILER_MSVC
	#define KT_BVH_UNREACHABLE __assume(0);
#else
	#error Compiler not supported.
#endif

#ifndef KT_BVH_ASSERT
	#include <assert.h>
	#define KT_BVH_ASSERT(_expr) assert((_expr))
#endif

namespace kt_bvh
{

struct AllocHooks
{
	using AllocFn = void*(*)(void* _ctx, size_t _size);
	using FreeFn  = void(*)(void* _ctx, void* _ptr);

	AllocFn alloc_fn;
	FreeFn free_fn;
	void* ctx;
};

struct TriMesh
{
	enum class IndexType : uint32_t
	{
		UnIndexed,
		U16,
		U32
	};

	void set_vertices(float const* _vertices, uint32_t _vertex_stride, uint32_t _num_vertices)
	{
		vertices = _vertices;
		num_vertices = _num_vertices;
		vertex_stride = _vertex_stride;
	}

	void set_unindexed()
	{
		indices_u16 = nullptr;
		num_idx_prims = 0;
		index_type = IndexType::UnIndexed;
	}

	void set_indices(uint32_t const* _index_buffer, uint32_t _num_prims)
	{
		indices_u32 = _index_buffer;
		num_idx_prims = _num_prims;
		index_type = IndexType::U32;
	}

	void set_indices(uint16_t const* _index_buffer, uint32_t _num_prims)
	{
		indices_u16 = _index_buffer;
		num_idx_prims = _num_prims;
		index_type = IndexType::U16;
	}

	uint32_t total_prims() const
	{
		return index_type == IndexType::UnIndexed ? num_vertices / 3 : num_idx_prims;
	}

	// Vertex buffer, containing position (float3) data.
	float const* vertices;

	// Stride between float3 position elements in vertex buffer.
	uint32_t vertex_stride;

	// Total number of vertices.
	uint32_t num_vertices;

	// Index buffer.
	union
	{
		uint16_t const* indices_u16;
		uint32_t const* indices_u32;
	};

	// Data type of index buffer.
	IndexType index_type;

	// Number of primitives in index buffer, if one is used.
	uint32_t num_idx_prims;
};

struct PrimitiveID
{
	uint32_t mesh_idx;
	uint32_t mesh_prim_idx;
};

struct BVH2Node
{
	bool is_leaf() const
	{
		return num_prims_in_leaf != 0;
	}

	float aabb_min[3];
	float aabb_max[3]; 

	uint32_t right_child_or_prim_offset;
	uint16_t num_prims_in_leaf;
	uint16_t split_axis;
};

struct BVH4Node
{
	bool is_child_empty(uint32_t _idx) const
	{
		return children[_idx] != UINT32_MAX;
	}

	bool is_child_leaf(uint32_t _idx) const
	{
		return num_prims_in_leaf[_idx] > 0;
	}

	float aabb_min_soa[3][4];
	float aabb_max_soa[3][4];

	uint32_t children[4];
    uint16_t num_prims_in_leaf[4];
    uint8_t split_axis[3];

    uint32_t _pad_;
};

enum class BVHBuildType
{
	MedianSplit,
	TopDownBinnedSAH
};

enum class BVHWidth
{
	BVH2,
	BVH4,
};

struct BVHBuildDesc
{
	static uint32_t const c_max_branching_factor = 4;

	static uint32_t const c_default_min_prims_per_leaf = 4;
	static uint32_t const c_default_max_prims_per_leaf = 8;
    static uint32_t const c_default_sah_buckets = 16;

	static uint32_t const c_max_sah_buckets = 32;

	uint32_t get_branching_factor() const
	{
		switch (width)
		{
			case BVHWidth::BVH2: return 2;
			case BVHWidth::BVH4: return 4;
		}

		KT_BVH_ASSERT(false); KT_BVH_UNREACHABLE;
	}

	// BVH build algorithm.
	BVHBuildType type = BVHBuildType::TopDownBinnedSAH;

	// Threshold of primitive to force leaf creation.
	uint32_t min_leaf_prims = c_default_min_prims_per_leaf;

	// Maximum amount of primitives per leaf, nodes will be further split to accommodate.
	uint32_t max_leaf_prims = c_default_max_prims_per_leaf;

	// Number of surface area heuristic binning buckets.
	uint32_t sah_buckets = c_default_sah_buckets;

	// Width of BVH.
	BVHWidth width = BVHWidth::BVH2;

	// Estimated cost of traversal for surface area heuristic (relative to intersection cost).
	float sah_traversal_cost = 0.85f;

    // Should SAH check all axis, or just major axis of AABB.
    bool sah_exhaustive_axis_test = true;
};

struct IntermediateBVHNode
{
	bool is_leaf() const
	{
		return num_children == 0;
	}

	// AABB of this node.
	float aabb_min[3];
	float aabb_max[3];

	// Child nodes (leaf or interior).
	IntermediateBVHNode* children[BVHBuildDesc::c_max_branching_factor];
	
	// SAH split axis for child nodes.
	uint32_t split_axis[BVHBuildDesc::c_max_branching_factor - 1];

	// Number of child nodes.
	uint32_t num_children;

	// Offset and count into primitive ID array.
	uint32_t leaf_prim_offset;
	uint32_t leaf_num_prims;
};

struct BVHBuildResult
{
	// Root of BVH tree.
	IntermediateBVHNode const* root;

	// Total amount of intermediate nodes in tree.
	uint32_t total_nodes;

	// Total amount of leaf nodes in tree.
	uint32_t total_leaf_nodes;

	// Total amount of interior nodes in tree.
	uint32_t total_interior_nodes;

	// Maximum depth of tree.
	uint32_t max_depth;

	// Primitive ID array that intermediate nodes point to.
	PrimitiveID const* prim_id_array;
	uint32_t prim_id_array_size;
};

struct IntermediateBVH;

// Build an intermediate representation of a BVH tree for the supplied triangle meshes.
IntermediateBVH* bvh_build_intermediate(TriMesh const* _meshes, uint32_t _num_meshes, BVHBuildDesc const& _build_desc, AllocHooks* alloc_hooks = nullptr);

// Free internal structures from intermediate BVH build.
void bvh_free_intermediate(IntermediateBVH* _intermediate_bvh);

// Get resulting BVH info.
BVHBuildResult bvh_build_result(IntermediateBVH* _intermediate_bvh);

// Write out flat representation of intermediate BVH2 tree. _node_cap must be >= BVHBuildResult::total_nodes
bool bvh2_intermediate_to_flat(IntermediateBVH const* _bvh2, BVH2Node* o_nodes, uint32_t _node_cap);

// Write out flat representation of intermediate BVH4 tree. _node_cap must be >= BVHBuildResult::total_interior_nodes
bool bvh4_intermediate_to_flat(IntermediateBVH const* _bvh4, BVH4Node* o_nodes, uint32_t _node_cap);

} // namespace kt_bvh