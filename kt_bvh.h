#pragma once
#include <stdint.h>

#define KT_BVH_COMPILER_MSVC		(0)
#define KT_BVH_COMPILER_CLANG		(0)
#define KT_BVH_COMPILER_GCC			(0)

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
	#error Compiler could not be detected.
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


enum class BVH2BuildType
{
	MedianSplit,
	TopDownBinnedSAH
};

struct BVH2BuildDesc
{
	static uint32_t const c_default_min_prims_per_leaf = 4;
	static uint32_t const c_default_max_prims_per_leaf = 8;

	static uint32_t const c_sah_max_buckets = 32;

	static BVH2BuildDesc default_desc()
	{
		BVH2BuildDesc ret;
		ret.set_binned_sah(0.8f);
		return ret;
	}

	void set_binned_sah(float _traversal_cost, uint32_t _num_buckets = 16, uint32_t _min_prims_per_leaf = c_default_min_prims_per_leaf, uint32_t _max_prims_per_leaf = c_default_max_prims_per_leaf)
	{
		type = BVH2BuildType::TopDownBinnedSAH;
		sah_buckets = _num_buckets > c_sah_max_buckets ? c_sah_max_buckets : _num_buckets;
		sah_traversal_cost = _traversal_cost;
		min_leaf_prims = _min_prims_per_leaf;
		max_leaf_prims = _max_prims_per_leaf;
	}

	void set_median_split(uint32_t _min_prims_per_leaf = c_default_min_prims_per_leaf, uint32_t _max_prims_per_leaf = c_default_max_prims_per_leaf)
	{
		type = BVH2BuildType::MedianSplit;
		min_leaf_prims = _min_prims_per_leaf;
		max_leaf_prims = _max_prims_per_leaf;
	}

	// BVH build algorithm.
	BVH2BuildType type;

	// Threshold of primitive to force leaf creation.
	uint32_t min_leaf_prims;

	// Maximum amount of primitives per leaf, nodes will be further split to accommodate.
	uint32_t max_leaf_prims;

	// Number of surface area heuristic binning buckets.
	uint32_t sah_buckets;

	// Estimated cost of traversal for surface area heuristic (relative to intersection cost).
	float sah_traversal_cost;
};

struct IntermediateBVH2;

// Build an intermediate representation of a BVH2 tree for the supplied triangle meshes.
IntermediateBVH2* bvh2_build_intermediate(TriMesh const* _meshes, uint32_t _num_meshes, BVH2BuildDesc const& _build_desc, AllocHooks* alloc_hooks = nullptr);

// Free internal structures from intermediate BVH2 build.
void bvh2_free_intermediate(IntermediateBVH2* _intermediate_bvh);

// Get the primitive ID array built with the tree. The BVH nodes point to offsets inside of this array.
// User should either copy or convert this to their own format for intersection.
void bvh2_get_primitive_id_array(IntermediateBVH2 const* _bvh2, PrimitiveID const** o_prim_array, uint32_t* o_prim_array_size);

// Return required amount of nodes in BVH2 tree. 
uint32_t bvh2_intermediate_num_nodes(IntermediateBVH2 const* _bvh2);

// Max depth of BVH2 tree.
uint32_t bvh2_depth(IntermediateBVH2 const* _bvh2);

// Write out flat representation of intermediate BVH2 tree. _node_cap must be >= bvh2_intermediate_num_nodes()
bool bvh2_intermediate_to_flat(IntermediateBVH2 const* _bvh2, BVH2Node* o_nodes, uint32_t _node_cap);

} // namespace kt_bvh