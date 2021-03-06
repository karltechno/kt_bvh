#include "kt_bvh.h"

#include <malloc.h> // malloc
#include <string.h> // memset, memcpy
#include <float.h> // FLT_MAX
#include <math.h> // NAN

#if KT_BVH_SSE
	#include <xmmintrin.h>
#endif

#define KT_BVH_ALLOCA(_size) ::alloca(_size)

#if KT_BVH_COMPILER_MSVC
	#pragma warning(push)
	#pragma warning(disable: 4201) // nonstandard extension used : nameless struct/union
#endif

namespace kt_bvh
{

static const uint32_t c_invalid_split = UINT32_MAX;

template <typename T>
static T min(T _lhs, T _rhs)
{
	return _lhs < _rhs ? _lhs : _rhs;
}

template <typename T>
static T max(T _lhs, T _rhs)
{
	return _lhs > _rhs ? _lhs : _rhs;
}

template <typename T>
static void swap(T& _lhs, T& _rhs)
{
	T temp = _lhs;
	_lhs = _rhs;
	_rhs = temp;
}



template <typename T>
static bool is_pow2(T _v)
{
	return (_v & (_v - 1)) == 0;
}

template <typename T>
static T align_pow2(T _v, uint32_t _pow2_align)
{
	KT_BVH_ASSERT(is_pow2(_pow2_align));
	return (_v + _pow2_align - 1) & ~(T(_pow2_align) - 1);
}

static AllocHooks malloc_hooks()
{
	AllocHooks hooks;
	hooks.alloc_fn = [](void*, size_t _size) -> void* { return malloc(_size); };
	hooks.free_fn = [](void*, void* _ptr) -> void { return free(_ptr); };
	hooks.ctx = nullptr;
	return hooks;
}

struct alignas(16) Vec3
{
	union
	{
#if KT_BVH_SSE
		__m128 xmm;
#endif
		float data[4];

		struct
		{
			float x;
			float y;
			float z;
			float w;
		};
	};
};

static Vec3 operator-(Vec3 const& _lhs, Vec3 const& _rhs)
{
#if KT_BVH_SSE
	return Vec3{ _mm_sub_ps(_lhs.xmm, _rhs.xmm) };
#else
	return Vec3{ _lhs.x - _rhs.x, _lhs.y - _rhs.y, _lhs.z - _rhs.z };
#endif
}

static Vec3 operator+(Vec3 const& _lhs, Vec3 const& _rhs)
{
#if KT_BVH_SSE
	return Vec3{ _mm_add_ps(_lhs.xmm, _rhs.xmm) };
#else
	return Vec3{ _lhs.x + _rhs.x, _lhs.y + _rhs.y, _lhs.z + _rhs.z };
#endif
}

static Vec3 operator/(Vec3 const& _lhs, Vec3 const& _rhs)
{
#if KT_BVH_SSE
	return Vec3{ _mm_div_ps(_lhs.xmm, _rhs.xmm) };
#else
	return Vec3{ _lhs.x + _rhs.x, _lhs.y + _rhs.y, _lhs.z + _rhs.z };
#endif
}

static Vec3 operator*(Vec3 const& _lhs, Vec3 const& _rhs)
{
#if KT_BVH_SSE
	return Vec3{ _mm_mul_ps(_lhs.xmm, _rhs.xmm) };
#else
	return Vec3{ _lhs.x + _rhs.x, _lhs.y + _rhs.y, _lhs.z + _rhs.z };
#endif
}

static Vec3 operator*(Vec3 const& _v, float _scalar)
{
#if KT_BVH_SSE
	return Vec3{ _mm_mul_ps(_v.xmm, _mm_set1_ps(_scalar)) };
#else
	return Vec3{ _v.x * _scalar, _v.y * _scalar, _v.z * _scalar };
#endif
}

static Vec3 min(Vec3 const& _lhs, Vec3 const& _rhs)
{
#if KT_BVH_SSE
	return Vec3{ _mm_min_ps(_lhs.xmm, _rhs.xmm) };
#else
	Vec3 r;
	r.x = min(_lhs.x, _rhs.x);
	r.y = min(_lhs.y, _rhs.y);
	r.z = min(_lhs.z, _rhs.z);
	return r;
#endif
}

static Vec3 max(Vec3 const& _lhs, Vec3 const& _rhs)
{
#if KT_BVH_SSE
	return Vec3{ _mm_max_ps(_lhs.xmm, _rhs.xmm) };
#else
	Vec3 r;
	r.x = max(_lhs.x, _rhs.x);
	r.y = max(_lhs.y, _rhs.y);
	r.z = max(_lhs.z, _rhs.z);
	return r;
#endif
}

static Vec3 vec3_splat(float _f)
{
#if KT_BVH_SSE
	return Vec3{ _mm_set1_ps(_f) };
#else
	return Vec3{ _f, _f, _f, _f };
#endif
}

struct AABB
{
	Vec3 min;
	Vec3 max;
};

static AABB aabb_union(AABB const& _lhs, AABB const& _rhs)
{
	return AABB{ min(_lhs.min, _rhs.min), max(_lhs.max, _rhs.max) };
}

static AABB aabb_expand(AABB const& _aabb, Vec3 const& _v)
{
	return AABB{ min(_aabb.min, _v), max(_aabb.max, _v) };
}

static Vec3 aabb_center(AABB const& _aabb)
{
	return _aabb.min * 0.5f + _aabb.max * 0.5f;
}

static AABB aabb_invalid()
{
	return AABB{ vec3_splat(FLT_MAX), vec3_splat(-FLT_MAX) };
}

static AABB aabb_init(float const* _min, float const* _max)
{
	AABB r;
	memcpy(r.min.data, _min, sizeof(float[3]));
	memcpy(r.max.data, _max, sizeof(float[3]));
	return r;
}

static float aabb_half_surface_area(AABB const& _aabb)
{
	Vec3 const diag = _aabb.max - _aabb.min;
	return (diag.x * diag.y + diag.y * diag.z + diag.z * diag.x);
}

static uint32_t aabb_major_axis(AABB const& _aabb)
{
	Vec3 const aabb_range = _aabb.max - _aabb.min;

	uint32_t split_axis = 0;

	if (aabb_range.y > aabb_range.x)
	{
		split_axis = 1;
	}
	if (aabb_range.z > aabb_range.data[split_axis])
	{
		split_axis = 2;
	}

	return split_axis;
}


struct MemArena
{
	struct ChunkHeader
	{
		ChunkHeader* next;
		uintptr_t cur_ptr;
		uintptr_t end_ptr;
	};

	static size_t const c_chunkSize = 4 * 1024 * 1024;

	void init(AllocHooks const& _hooks)
	{
		hooks = _hooks;
		head = nullptr;
	}

	void new_chunk(size_t _min_size)
	{
		size_t const chunk_size = max(_min_size + sizeof(ChunkHeader), c_chunkSize);
		uint8_t* new_mem = (uint8_t*)hooks.alloc_fn(hooks.ctx, chunk_size);

		ChunkHeader* new_chunk = (ChunkHeader*)new_mem;
		new_chunk->cur_ptr = uintptr_t(new_mem + sizeof(ChunkHeader));
		new_chunk->end_ptr = new_chunk->cur_ptr + (chunk_size - sizeof(ChunkHeader));
		new_chunk->next = head;
		head = new_chunk;
	}


	void* alloc(size_t _size, uint32_t _align = 8)
	{
		if (head)
		{
			head->cur_ptr = min(head->end_ptr, align_pow2(head->cur_ptr, _align));
		}

		if (!head || head->end_ptr - head->cur_ptr < _size)
		{
			new_chunk(_size + _align - 1);
			head->cur_ptr = align_pow2(head->cur_ptr, _align);
			KT_BVH_ASSERT(head->cur_ptr < head->end_ptr);
		}

		KT_BVH_ASSERT(head->end_ptr - head->cur_ptr >= _size);
		void* ret = (void*)head->cur_ptr;
		head->cur_ptr += _size;
		return ret;
	}

	template <typename T>
	T* alloc_array_uninitialized(uint32_t _count)
	{
		return (T*)alloc(sizeof(T) * _count, alignof(T));
	}

	template <typename T>
	T* alloc_uninitialized()
	{
		return (T*)alloc(sizeof(T), alignof(T));
	}

	template <typename T>
	T* alloc_zeroed()
	{
		T* ptr = (T*)alloc(sizeof(T), alignof(T));
		memset(ptr, 0, sizeof(T));
		return ptr;
	}

	void free_all()
	{
		ChunkHeader* next = head;
		while (next)
		{
			ChunkHeader* to_free = next;
			next = to_free->next;
			hooks.free_fn(hooks.ctx, to_free);
		}
	}


	AllocHooks hooks;
	ChunkHeader* head;
};

template <typename T>
struct dyn_pod_array
{
	dyn_pod_array() = default;

	dyn_pod_array(dyn_pod_array const&) = delete;
	dyn_pod_array& operator=(dyn_pod_array const&) = delete;

	void init(MemArena* _arena)
	{
		arena = _arena;
	}

	void ensure_cap(uint32_t _req_cap)
	{
		if (cap < _req_cap)
		{
			uint32_t const amortized_grow = cap + cap / 2;
			uint32_t const new_cap = amortized_grow < _req_cap ? _req_cap : amortized_grow;

			T* new_mem = (T*)arena->alloc(sizeof(T) * new_cap);

			if (mem)
			{
				memcpy(new_mem, mem, size * sizeof(T));
			}

			cap = new_cap;
			mem = new_mem;
		}
	}

	void append(T const& _v)
	{
		*append() = _v;
	}

	T* append()
	{
		ensure_cap(size + 1);
		return &mem[size++];
	}

	T* append_n(uint32_t _n)
	{
		ensure_cap(size + _n);
		T* ptr = mem + size;
		size += _n;
		return ptr;
	}

	T* begin()
	{
		return mem;
	}

	T* end()
	{
		return mem + size;
	}

	T& operator[](uint32_t _idx)
	{
		KT_BVH_ASSERT(_idx < size);
		return mem[_idx];
	}

	T* mem = nullptr;
	uint32_t size = 0;
	uint32_t cap = 0;

	MemArena* arena;
};

struct IntermediateBVH
{
	MemArena arena;
	IntermediateBVHNode *root;
	uint32_t total_nodes;
	uint32_t total_leaf_nodes;

	dyn_pod_array<PrimitiveID> bvh_prim_id_list;
	uint32_t max_depth;

	BVHWidth width;
};

struct IntermediatePrimitive
{
	AABB aabb;
	Vec3 origin;
};

struct BVHBuilderContext
{
	MemArena& arena()
	{
		return bvh->arena;
	}

	IntermediateBVHNode* new_node(uint32_t _depth)
	{
		bvh->max_depth = max(bvh->max_depth, _depth);
		IntermediateBVHNode* node = arena().alloc_zeroed<IntermediateBVHNode>();
		++bvh->total_nodes;

		return node;
	}

	IntermediateBVH* bvh;

	IntermediatePrimitive* primitive_info;
	PrimitiveID* prim_ids;
	uint32_t total_primitives;

	TriMesh const* meshes;
	uint32_t num_meshes;

	BVHBuildDesc build_desc;
};

static void mesh_get_prim_idx_u16(TriMesh const& _mesh, uint32_t _prim_idx, Vec3* o_v0, Vec3* o_v1, Vec3* o_v2)
{
	KT_BVH_ASSERT(_mesh.index_type == TriMesh::IndexType::U16);
	KT_BVH_ASSERT(_prim_idx < _mesh.total_prims());
	uint16_t const i0 = _mesh.indices_u16[_prim_idx * 3 + 0];
	uint16_t const i1 = _mesh.indices_u16[_prim_idx * 3 + 1];
	uint16_t const i2 = _mesh.indices_u16[_prim_idx * 3 + 2];
	KT_BVH_ASSERT(i0 < _mesh.num_idx_prims);
	KT_BVH_ASSERT(i1 < _mesh.num_idx_prims);
	KT_BVH_ASSERT(i2 < _mesh.num_idx_prims);
	memcpy(o_v0, _mesh.vertices + _mesh.vertex_stride * i0, sizeof(float[3]));
	memcpy(o_v1, _mesh.vertices + _mesh.vertex_stride * i1, sizeof(float[3]));
	memcpy(o_v2, _mesh.vertices + _mesh.vertex_stride * i2, sizeof(float[3]));
}

static void mesh_get_prim_idx_u32(TriMesh const& _mesh, uint32_t _prim_idx, Vec3* o_v0, Vec3* o_v1, Vec3* o_v2)
{
	KT_BVH_ASSERT(_mesh.index_type == TriMesh::IndexType::U32);
	KT_BVH_ASSERT(_prim_idx < _mesh.total_prims());
	uint32_t const i0 = _mesh.indices_u32[_prim_idx * 3 + 0];
	uint32_t const i1 = _mesh.indices_u32[_prim_idx * 3 + 1];
	uint32_t const i2 = _mesh.indices_u32[_prim_idx * 3 + 2];
	KT_BVH_ASSERT(i0 < _mesh.num_idx_prims);
	KT_BVH_ASSERT(i1 < _mesh.num_idx_prims);
	KT_BVH_ASSERT(i2 < _mesh.num_idx_prims);
	memcpy(o_v0, (uint8_t*)_mesh.vertices + _mesh.vertex_stride * i0, sizeof(float[3]));
	memcpy(o_v1, (uint8_t*)_mesh.vertices + _mesh.vertex_stride * i1, sizeof(float[3]));
	memcpy(o_v2, (uint8_t*)_mesh.vertices + _mesh.vertex_stride * i2, sizeof(float[3]));
}

static void mesh_get_prim_unindexed(TriMesh const& _mesh, uint32_t _prim_idx, Vec3* o_v0, Vec3* o_v1, Vec3* o_v2)
{
	KT_BVH_ASSERT(_mesh.index_type == TriMesh::IndexType::UnIndexed);
	memcpy(o_v0, (uint8_t*)_mesh.vertices + _mesh.vertex_stride * (_prim_idx * 3 + 0), sizeof(float[3]));
	memcpy(o_v1, (uint8_t*)_mesh.vertices + _mesh.vertex_stride * (_prim_idx * 3 + 1), sizeof(float[3]));
	memcpy(o_v2, (uint8_t*)_mesh.vertices + _mesh.vertex_stride * (_prim_idx * 3 + 2), sizeof(float[3]));
}

/*
static void mesh_get_prim(TriMesh const& _mesh, uint32_t _prim_idx, Vec3* o_v0, Vec3* o_v1, Vec3* o_v2)
{
	switch (_mesh.index_type)
	{
		case TriMesh::IndexType::U16: mesh_get_prim_idx_u16(_mesh, _prim_idx, o_v0, o_v1, o_v2); break;
		case TriMesh::IndexType::U32: mesh_get_prim_idx_u32(_mesh, _prim_idx, o_v0, o_v1, o_v2); break;
		case TriMesh::IndexType::UnIndexed: mesh_get_prim_unindexed(_mesh, _prim_idx, o_v0, o_v1, o_v2); break;
	}
}
*/



template <void (GetPrimT)(TriMesh const&, uint32_t, Vec3*, Vec3*, Vec3*)>
static void build_prim_info_impl(TriMesh const& _mesh, uint32_t _mesh_idx, IntermediatePrimitive* _prim_arr, PrimitiveID* _prim_id_arr, AABB* o_enclosing_aabb, AABB* o_centroid_aabb)
{
	AABB centroid_aabb = *o_centroid_aabb;
	AABB enclosing_aabb = *o_centroid_aabb;

	for (uint32_t i = 0; i < _mesh.total_prims(); ++i)
	{
		Vec3 tri[3];
		GetPrimT(_mesh, i, &tri[0], &tri[1], &tri[2]);
		_prim_arr->aabb = AABB{ tri[0], tri[0] };
		_prim_arr->aabb = aabb_expand(_prim_arr->aabb, tri[1]);
		_prim_arr->aabb = aabb_expand(_prim_arr->aabb, tri[2]);
		_prim_arr->origin = aabb_center(_prim_arr->aabb);
		_prim_id_arr->mesh_idx = _mesh_idx;
		_prim_id_arr->mesh_prim_idx = i;

		enclosing_aabb = aabb_union(enclosing_aabb, _prim_arr->aabb);
		centroid_aabb = aabb_expand(centroid_aabb, _prim_arr->origin);

		++_prim_arr;
		++_prim_id_arr;
	}

	*o_centroid_aabb = centroid_aabb;
	*o_enclosing_aabb = enclosing_aabb;
}

static void build_prim_info(BVHBuilderContext& _ctx, AABB* o_enclosing_aabb, AABB* o_centroid_aabb)
{
	uint32_t prim_idx = 0;
	for (uint32_t mesh_idx = 0; mesh_idx < _ctx.num_meshes; ++mesh_idx)
	{
		TriMesh const& mesh = _ctx.meshes[mesh_idx];
		switch (mesh.index_type)
		{
			case TriMesh::IndexType::U16: build_prim_info_impl<mesh_get_prim_idx_u16>(mesh, mesh_idx, _ctx.primitive_info + prim_idx, _ctx.prim_ids + prim_idx, o_enclosing_aabb, o_centroid_aabb); break;
			case TriMesh::IndexType::U32: build_prim_info_impl<mesh_get_prim_idx_u32>(mesh, mesh_idx, _ctx.primitive_info + prim_idx, _ctx.prim_ids + prim_idx, o_enclosing_aabb, o_centroid_aabb); break;
			case TriMesh::IndexType::UnIndexed: build_prim_info_impl<mesh_get_prim_unindexed>(mesh, mesh_idx, _ctx.primitive_info + prim_idx, _ctx.prim_ids + prim_idx, o_enclosing_aabb, o_centroid_aabb); break;
		}

		prim_idx += mesh.total_prims();
		KT_BVH_ASSERT(prim_idx <= _ctx.total_primitives);
	}

	KT_BVH_ASSERT(prim_idx == _ctx.total_primitives);
}

// Partition array such that all passing the predicate come first.
template <typename PredT>
static uint32_t partition_prim_id(BVHBuilderContext& _ctx, uint32_t _begin, uint32_t _end, PredT _pred)
{
	IntermediatePrimitive* prims = _ctx.primitive_info;
	PrimitiveID* ids = _ctx.prim_ids;

	uint32_t swap_idx = _begin;

	while (swap_idx != _end && _pred(prims[swap_idx]))
	{
		++swap_idx;
	}

	if (swap_idx != _end)
	{
		uint32_t cur_idx = swap_idx + 1;

		while (cur_idx != _end)
		{
			if (_pred(prims[cur_idx]))
			{
				swap(prims[cur_idx], prims[swap_idx]);
				swap(ids[cur_idx], ids[swap_idx]);
				++swap_idx;
			}
			++cur_idx;
		}
	}

	return swap_idx;
}

static void node_copy_aabb(IntermediateBVHNode* _node, AABB const& _aabb)
{
	memcpy(_node->aabb_min, _aabb.min.data, sizeof(float[3]));
	memcpy(_node->aabb_max, _aabb.max.data, sizeof(float[3]));
}

struct SAHSplitResult
{
	bool is_valid() const
	{
		return axis != UINT32_MAX;
	}

	float best_sah_cost() const
	{
		return min(sah_cost, sah_leaf_cost);
	}

	uint32_t axis = UINT32_MAX;
	uint32_t split_idx;

	float sah_cost = FLT_MAX;
	float sah_leaf_cost = FLT_MAX;
};

struct PreSplitIntermediateBVHNode
{
	uint32_t num_prims() const
	{
		KT_BVH_ASSERT(prim_end >= prim_begin);
		return prim_end - prim_begin;
	}

	AABB enclosing_aabb;
	AABB centroid_aabb;

	uint32_t prim_begin;
	uint32_t prim_end;
};

static IntermediateBVHNode* build_leaf_node(BVHBuilderContext& _ctx, uint32_t _depth, uint32_t _prim_begin, uint32_t _prim_end)
{
	KT_BVH_ASSERT(_prim_end > _prim_begin);

	uint32_t const nprims = _prim_end - _prim_begin;

	if (nprims <= _ctx.build_desc.max_leaf_prims)
	{
		IntermediateBVHNode* node = _ctx.new_node(_depth);
		++_ctx.bvh->total_leaf_nodes;
		node->split_axis[0] = 0;

		AABB aabb = aabb_invalid();
		node->leaf_prim_offset = _ctx.bvh->bvh_prim_id_list.size;
		node->leaf_num_prims = nprims;
		node->num_children = 0;
		PrimitiveID* leaf_prims = _ctx.bvh->bvh_prim_id_list.append_n(nprims);

		for (uint32_t i = _prim_begin; i < _prim_end; ++i)
		{
			aabb = aabb_union(aabb, _ctx.primitive_info[i].aabb);
			*leaf_prims++ = _ctx.prim_ids[i];
		}

		node_copy_aabb(node, aabb);

		return node;
	}

	PreSplitIntermediateBVHNode presplit_nodes[BVHBuildDesc::c_max_branching_factor] = {};
	presplit_nodes[0].prim_begin = _prim_begin;
	presplit_nodes[0].prim_end = _prim_end;

	uint32_t num_children = 1;
	uint32_t const branching_factor = _ctx.build_desc.get_branching_factor();
	KT_BVH_ASSERT(branching_factor <= BVHBuildDesc::c_max_branching_factor);

	// Iteratively split large leaf into N smaller leafs, then recurse again if necessary.
	do
	{
		uint32_t biggest_leaf_idx = UINT32_MAX;
		uint32_t most_prims = 0;

		for (uint32_t i = 0; i < num_children; ++i)
		{
			uint32_t const child_nprims = presplit_nodes[i].num_prims();

			if (child_nprims > _ctx.build_desc.max_leaf_prims
				&& presplit_nodes[i].num_prims() > most_prims)
			{
				most_prims = presplit_nodes[i].num_prims();
				biggest_leaf_idx = i;
			}
		}

		if (biggest_leaf_idx == UINT32_MAX)
		{
			// At this point all leafs fit so exit.
			break;
		}

		PreSplitIntermediateBVHNode& to_split = presplit_nodes[biggest_leaf_idx];

		uint32_t const split_mid = (to_split.prim_begin + to_split.prim_end) / 2;

		PreSplitIntermediateBVHNode new_splits[2];
		new_splits[0].prim_begin = to_split.prim_begin;
		new_splits[0].prim_end = split_mid;
		new_splits[1].prim_begin = split_mid;
		new_splits[1].prim_end = to_split.prim_end;

		swap(presplit_nodes[num_children - 1], presplit_nodes[biggest_leaf_idx]);
		presplit_nodes[num_children - 1] = new_splits[0];
		presplit_nodes[num_children] = new_splits[1];
		++num_children;

	} while (num_children < branching_factor);

	IntermediateBVHNode* subtree_root = _ctx.new_node(_depth);
	subtree_root->num_children = num_children;

	AABB enclosing_aabb = aabb_invalid();

	for (uint32_t i = 0; i < num_children; ++i)
	{
		IntermediateBVHNode* child = build_leaf_node(_ctx, _depth + 1, presplit_nodes[i].prim_begin, presplit_nodes[i].prim_end);
		subtree_root->children[i] = child;
		enclosing_aabb = aabb_union(enclosing_aabb, aabb_init(child->aabb_min, child->aabb_max));
	}

	node_copy_aabb(subtree_root, enclosing_aabb);
	return subtree_root;
}

struct SAHBucketingInfo
{
	SAHBucketingInfo()
	{
		for (AABB& aabb : bucket_bounds)
		{
			aabb = aabb_invalid();
		}
	}

	AABB bucket_bounds[BVHBuildDesc::c_max_sah_buckets];
	uint32_t bucket_num_prims[BVHBuildDesc::c_max_sah_buckets] = {};

	AABB forward_split_bounds[BVHBuildDesc::c_max_sah_buckets - 1];
	uint32_t forward_prim_count[BVHBuildDesc::c_max_sah_buckets - 1] = {};
};

static SAHSplitResult eval_sah_split
(
	BVHBuilderContext& _ctx,
	AABB const& _enclosing_aabb,
	AABB const& _centroid_aabb,
	uint32_t _prim_begin,
	uint32_t _prim_end
)
{
	SAHSplitResult best_split_result;
	SAHBucketingInfo axis_buckets[3];

	uint32_t const num_buckets = _ctx.build_desc.sah_buckets;
	uint32_t const num_splits = num_buckets - 1;

	Vec3 const split_axis_len = (_centroid_aabb.max - _centroid_aabb.min);

	Vec3 const project_dim_constant = vec3_splat(float(num_buckets)) / split_axis_len;
	Vec3 const bucket_max = vec3_splat(float(num_buckets - 1));

	for (uint32_t prim_idx = _prim_begin; prim_idx < _prim_end; ++prim_idx)
	{
		IntermediatePrimitive const& prim = _ctx.primitive_info[prim_idx];
		Vec3 const bucket3 = min(bucket_max, (prim.origin - _centroid_aabb.min) * project_dim_constant);

		uint32_t const bucket0 = uint32_t(bucket3.x);
		uint32_t const bucket1 = uint32_t(bucket3.y);
		uint32_t const bucket2 = uint32_t(bucket3.z);

		axis_buckets[0].bucket_num_prims[bucket0]++;
		axis_buckets[1].bucket_num_prims[bucket1]++;
		axis_buckets[2].bucket_num_prims[bucket2]++;

		axis_buckets[0].bucket_bounds[bucket0] = aabb_union(prim.aabb, axis_buckets[0].bucket_bounds[bucket0]);
		axis_buckets[1].bucket_bounds[bucket1] = aabb_union(prim.aabb, axis_buckets[1].bucket_bounds[bucket1]);
		axis_buckets[2].bucket_bounds[bucket2] = aabb_union(prim.aabb, axis_buckets[2].bucket_bounds[bucket2]);
	}


	for (uint32_t split_axis = 0; split_axis < 3; ++split_axis)
	{
		if (split_axis_len.data[split_axis] <= 0.0001f)
		{
			continue;
		}

		SAHBucketingInfo& bucket_info = axis_buckets[split_axis];

		bucket_info.forward_split_bounds[0] = bucket_info.bucket_bounds[0];
		bucket_info.forward_prim_count[0] = bucket_info.bucket_num_prims[0];

		for (int32_t i = 1; i < int32_t(num_splits); ++i)
		{
			bucket_info.forward_split_bounds[i] = aabb_union(bucket_info.forward_split_bounds[i - 1], bucket_info.bucket_bounds[i]);
			bucket_info.forward_prim_count[i] = bucket_info.forward_prim_count[i - 1] + bucket_info.bucket_num_prims[i];
		}

		AABB incremental_reverse_aabb = aabb_invalid();
		uint32_t incremental_reverse_prim_count = 0;
		float const root_surface_area = aabb_half_surface_area(_enclosing_aabb);

		uint32_t best_split = UINT32_MAX;
		float best_cost = FLT_MAX;

		float const traversal_cost = _ctx.build_desc.sah_traversal_cost;

		uint32_t const nprims = _prim_end - _prim_begin;

		for (int32_t i = int32_t(num_splits) - 1; i >= 0; --i)
		{
			incremental_reverse_aabb = aabb_union(incremental_reverse_aabb, bucket_info.bucket_bounds[i + 1]);
			incremental_reverse_prim_count += bucket_info.bucket_num_prims[i + 1];

			uint32_t const countA = bucket_info.forward_prim_count[i];
			uint32_t const countB = incremental_reverse_prim_count;

			KT_BVH_ASSERT(countA + countB == nprims);

			if (countA == 0 || countB == 0)
			{
				continue;
			}

			AABB const& forward_bounds = bucket_info.forward_split_bounds[i];

			float const sa = aabb_half_surface_area(forward_bounds);
			float const sb = aabb_half_surface_area(incremental_reverse_aabb);
			float const cost = traversal_cost + (sa * countA + sb * countB) / root_surface_area;

			if (cost < best_cost)
			{
				best_cost = cost;
				best_split = i;
			}
		}

		if (best_split == UINT32_MAX)
		{
			continue;
		}

		SAHSplitResult result;

		result.axis = split_axis;
		result.split_idx = best_split;
		result.sah_cost = best_cost;
		result.sah_leaf_cost = float(nprims);

		if (result.is_valid() && result.best_sah_cost() < best_split_result.best_sah_cost())
		{
			best_split_result = result;
		}
	}

	return best_split_result;
}

uint32_t exec_sah_split(BVHBuilderContext& _ctx, SAHSplitResult const& _result, AABB const& _centroid_aabb, uint32_t _prim_begin, uint32_t _prim_end)
{
	uint32_t const nbuckets = _ctx.build_desc.sah_buckets;
	float const split_axis_len = (_centroid_aabb.max - _centroid_aabb.min).data[_result.axis];
	float const project_dim_constant = nbuckets / split_axis_len;

	float const bucket_max = float(nbuckets - 1);

	uint32_t const mid_idx = partition_prim_id(_ctx, _prim_begin, _prim_end,
										   [&_result, nbuckets, &_centroid_aabb, project_dim_constant, bucket_max](IntermediatePrimitive const& _prim) -> bool
	{
		float const project_to_dim = (_prim.origin - _centroid_aabb.min).data[_result.axis] * project_dim_constant;
		uint32_t const bucket = uint32_t(min(bucket_max, project_to_dim));
		return bucket <= _result.split_idx;
	});

	return mid_idx;
}

static uint32_t split_node
(
	BVHBuilderContext& _ctx,
	AABB const& _enclosing_aabb,
	AABB const& _centroid_aabb,
	uint32_t _prim_begin,
	uint32_t _prim_end,
	uint32_t* o_split_axis
)
{
	uint32_t middle_prim_split_idx = c_invalid_split;

	switch (_ctx.build_desc.type)
	{
		case BVHBuildType::MedianSplit:
		{
			uint32_t const axis = aabb_major_axis(_centroid_aabb);
			float const midpoint = (_centroid_aabb.max * 0.5f + _centroid_aabb.min * 0.5f).data[axis];
			middle_prim_split_idx = partition_prim_id(_ctx, _prim_begin, _prim_end,
													   [midpoint, axis](IntermediatePrimitive const& _val) { return _val.origin.data[axis] < midpoint; });
			*o_split_axis = axis;
			if (middle_prim_split_idx == _prim_begin || middle_prim_split_idx == _prim_end)
			{
				// TODO: Sort?
				return (_prim_begin + _prim_end) / 2;
			}
			return middle_prim_split_idx;
		};

		case BVHBuildType::TopDownBinnedSAH:
		{
			SAHSplitResult const object_split = eval_sah_split(_ctx, _enclosing_aabb, _centroid_aabb, _prim_begin, _prim_end);

			if (!object_split.is_valid())
			{
				return c_invalid_split;
			}

			*o_split_axis = object_split.axis;

			middle_prim_split_idx = exec_sah_split(_ctx, object_split, _centroid_aabb, _prim_begin, _prim_end);
			KT_BVH_ASSERT(middle_prim_split_idx > _prim_begin && middle_prim_split_idx < _prim_end);
			return middle_prim_split_idx;
		} break;

	}

	KT_BVH_ASSERT(false); KT_BVH_UNREACHABLE;
}


void calc_enclosing_and_centroid_aabb(BVHBuilderContext& _ctx, uint32_t _prim_begin, uint32_t _prim_end, AABB* o_enclosing, AABB* o_centroid)
{
	AABB enclosing_aabb = aabb_invalid();
	AABB centroid_aabb = aabb_invalid();

	for (uint32_t i = _prim_begin; i < _prim_end; ++i)
	{
		enclosing_aabb = aabb_union(enclosing_aabb, _ctx.primitive_info[i].aabb);
		centroid_aabb = aabb_expand(centroid_aabb, _ctx.primitive_info[i].origin);
	}

	*o_centroid = centroid_aabb;
	*o_enclosing = enclosing_aabb;
}

static IntermediateBVHNode* build_bvhn_recursive(BVHBuilderContext& _ctx, AABB const& _enclosing_aabb, AABB const& _centroid_aabb, uint32_t _depth, uint32_t _prim_begin, uint32_t _prim_end)
{
	KT_BVH_ASSERT(_prim_begin < _prim_end);

	_ctx.bvh->max_depth = max(_ctx.bvh->max_depth, _depth);

	uint32_t const nprims = _prim_end - _prim_begin;

	if (nprims <= _ctx.build_desc.min_leaf_prims)
	{
		return build_leaf_node(_ctx, _depth, _prim_begin, _prim_end);
	}

	uint32_t split_axis[BVHBuildDesc::c_max_branching_factor - 1];

	uint32_t const middle_prim_split_idx = split_node(_ctx, _enclosing_aabb, _centroid_aabb, _prim_begin, _prim_end, &split_axis[0]);
	if (middle_prim_split_idx == c_invalid_split)
	{
		// No valid split, make leaf directly.
		return build_leaf_node(_ctx, _depth, _prim_begin, _prim_end);
	}
	KT_BVH_ASSERT(_prim_begin < middle_prim_split_idx && _prim_end > middle_prim_split_idx);

	PreSplitIntermediateBVHNode presplit_nodes[BVHBuildDesc::c_max_branching_factor];
	uint32_t num_children = 2;

	presplit_nodes[0].prim_begin = _prim_begin;
	presplit_nodes[0].prim_end = middle_prim_split_idx;
	calc_enclosing_and_centroid_aabb(_ctx, _prim_begin, middle_prim_split_idx, &presplit_nodes[0].enclosing_aabb, &presplit_nodes[0].centroid_aabb);

	presplit_nodes[1].prim_begin = middle_prim_split_idx;
	presplit_nodes[1].prim_end = _prim_end;
	calc_enclosing_and_centroid_aabb(_ctx, middle_prim_split_idx, _prim_end, &presplit_nodes[1].enclosing_aabb, &presplit_nodes[1].centroid_aabb);

	// Now iteratively split till we reach our limit.
	uint32_t const branching_factor = _ctx.build_desc.get_branching_factor();
	KT_BVH_ASSERT(branching_factor <= BVHBuildDesc::c_max_branching_factor);

	while (num_children < branching_factor)
	{
		uint32_t best_leaf_idx = UINT32_MAX;
		float largest_surface_area = -FLT_MAX;

		for (uint32_t i = 0; i < num_children; ++i)
		{
			PreSplitIntermediateBVHNode const& split_node = presplit_nodes[i];
			float const split_surface_area = aabb_half_surface_area(split_node.enclosing_aabb);

			if (split_node.num_prims() > _ctx.build_desc.min_leaf_prims
				&& split_surface_area > largest_surface_area)
			{
				largest_surface_area = split_surface_area;
				best_leaf_idx = i;
			}
		}

		if (best_leaf_idx == UINT32_MAX)
		{
			// At this point all leafs fit so exit.
			break;
		}

		PreSplitIntermediateBVHNode& to_split = presplit_nodes[best_leaf_idx];
		uint32_t child_split_axis;
		uint32_t const child_middle_prim_split_idx = split_node(_ctx, to_split.enclosing_aabb, to_split.centroid_aabb, to_split.prim_begin, to_split.prim_end, &child_split_axis);

		if (child_middle_prim_split_idx == c_invalid_split)
		{
			break;
		}

		KT_BVH_ASSERT(num_children <= BVHBuildDesc::c_max_branching_factor);

		// Shuffle up to make room for this split and retain split_axis ordering (if we are splitting in the middle of the array).
		for (uint32_t split_insert = num_children - 1; split_insert > best_leaf_idx; --split_insert)
		{
			presplit_nodes[split_insert + 1] = presplit_nodes[split_insert];
			split_axis[split_insert] = split_axis[split_insert - 1];
		}


		uint32_t const split_begin = to_split.prim_begin;
		uint32_t const split_end = to_split.prim_end;

		KT_BVH_ASSERT(child_middle_prim_split_idx > split_begin && child_middle_prim_split_idx < split_end);

		presplit_nodes[best_leaf_idx].prim_begin = split_begin;
		presplit_nodes[best_leaf_idx].prim_end = child_middle_prim_split_idx;
		presplit_nodes[best_leaf_idx + 1].prim_begin = child_middle_prim_split_idx;
		presplit_nodes[best_leaf_idx + 1].prim_end = split_end;

		for (uint32_t leftright = 0; leftright < 2; ++leftright)
		{
			PreSplitIntermediateBVHNode& presplit = presplit_nodes[best_leaf_idx + leftright];
			calc_enclosing_and_centroid_aabb(_ctx, presplit.prim_begin, presplit.prim_end, &presplit.enclosing_aabb, &presplit.centroid_aabb);
		}

		split_axis[best_leaf_idx] = child_split_axis;

		++num_children;
	};

	IntermediateBVHNode* subtree_root = _ctx.new_node(_depth);
	static_assert(sizeof(subtree_root->split_axis) >= sizeof(split_axis), "Split axis arrays mismatch");
	memcpy(subtree_root->split_axis, split_axis, sizeof(split_axis));

	subtree_root->num_children = num_children;

	AABB subtree_aabb = aabb_invalid();

	for (uint32_t i = 0; i < num_children; ++i)
	{
		PreSplitIntermediateBVHNode const& split = presplit_nodes[i];
		IntermediateBVHNode* child = build_bvhn_recursive(_ctx, split.enclosing_aabb, split.centroid_aabb, _depth + 1, split.prim_begin, split.prim_end);
		subtree_root->children[i] = child;
		subtree_aabb = aabb_union(subtree_aabb, aabb_init(child->aabb_min, child->aabb_max));
	}

	node_copy_aabb(subtree_root, subtree_aabb);
	return subtree_root;
}

IntermediateBVH* bvh_build_intermediate(TriMesh const* _meshes, uint32_t _num_meshes, BVHBuildDesc const& _build_desc, AllocHooks* alloc_hooks)
{
	uint32_t total_prims = 0;
	for (uint32_t i = 0; i < _num_meshes; ++i)
	{
		total_prims += _meshes[i].total_prims();
	}

	if (!total_prims)
	{
		return nullptr;
	}

	AllocHooks hooks = alloc_hooks ? *alloc_hooks : malloc_hooks();

	IntermediateBVH* bvh_intermediate = nullptr;

	{
		MemArena arena;
		arena.init(hooks);
		bvh_intermediate = arena.alloc_zeroed<IntermediateBVH>();
		bvh_intermediate->arena = arena;
	}

	bvh_intermediate->width = _build_desc.width;

	BVHBuilderContext builder = {};

	builder.meshes = _meshes;
	builder.num_meshes = _num_meshes;
	builder.total_primitives = total_prims;
	builder.bvh = bvh_intermediate;
	builder.primitive_info = builder.arena().alloc_array_uninitialized<IntermediatePrimitive>(total_prims);
	builder.prim_ids = builder.arena().alloc_array_uninitialized<PrimitiveID>(total_prims);
	builder.build_desc = _build_desc;

	bvh_intermediate->bvh_prim_id_list.init(&bvh_intermediate->arena);

	bvh_intermediate->bvh_prim_id_list.ensure_cap(total_prims); // fixed capacity here, if we do spatial splits may need to increase.

	AABB enclosing_aabb, centroid_aabb;
	build_prim_info(builder, &enclosing_aabb, &centroid_aabb);

	builder.bvh->root = build_bvhn_recursive(builder, enclosing_aabb, centroid_aabb, 0, 0, total_prims);

	return bvh_intermediate;
}

void bvh_free_intermediate(IntermediateBVH* _intermediate_bvh)
{
	if (!_intermediate_bvh)
	{
		return;
	}

	// Copy arena as intermediate structure is allocated from it.
	MemArena arena = _intermediate_bvh->arena;
	arena.free_all();
}

BVHBuildResult bvh_build_result(IntermediateBVH* _intermediate_bvh)
{
	BVHBuildResult res = {};
	res.root = _intermediate_bvh->root;
	res.total_nodes = _intermediate_bvh->total_nodes;
	res.max_depth = _intermediate_bvh->max_depth;
	res.prim_id_array = _intermediate_bvh->bvh_prim_id_list.mem;
	res.prim_id_array_size = _intermediate_bvh->bvh_prim_id_list.size;
	res.total_leaf_nodes = _intermediate_bvh->total_leaf_nodes;
	res.total_interior_nodes = _intermediate_bvh->total_nodes - _intermediate_bvh->total_leaf_nodes;
	return res;
}

struct BVH2FlatWriterCtx
{
	BVH2Node* nodes;
	uint32_t cur_idx;
	uint32_t total_nodes;
};

void bvh2_write_flat_depth_first(BVH2FlatWriterCtx* _ctx, IntermediateBVHNode* _node)
{
	KT_BVH_ASSERT(_ctx->cur_idx < _ctx->total_nodes);

	BVH2Node* flatnode = &_ctx->nodes[_ctx->cur_idx++];
	memcpy(flatnode->aabb_min, _node->aabb_min, sizeof(float[3]));
	memcpy(flatnode->aabb_max, _node->aabb_max, sizeof(float[3]));
	flatnode->split_axis = uint16_t(_node->split_axis[0]);

	if (_node->is_leaf())
	{
		KT_BVH_ASSERT(_node->leaf_num_prims <= UINT16_MAX);
		flatnode->num_prims_in_leaf = uint16_t(_node->leaf_num_prims);
		flatnode->right_child_or_prim_offset = _node->leaf_prim_offset;
	}
	else
	{
		flatnode->num_prims_in_leaf = 0;
		bvh2_write_flat_depth_first(_ctx, _node->children[0]);
		flatnode->right_child_or_prim_offset = _ctx->cur_idx;
		bvh2_write_flat_depth_first(_ctx, _node->children[1]);
	}
}

bool bvh2_intermediate_to_flat(IntermediateBVH const* _bvh2, BVH2Node* o_nodes, uint32_t _node_cap)
{
	if (_node_cap < _bvh2->total_nodes || _bvh2->width != BVHWidth::BVH2)
	{
		return false;
	}

	BVH2FlatWriterCtx writer;
	writer.cur_idx = 0;
	writer.nodes = o_nodes;
	writer.total_nodes = _bvh2->total_nodes;;
	bvh2_write_flat_depth_first(&writer, _bvh2->root);

	return true;
}

struct BVH4FlatWriterCtx
{
	BVH4Node* nodes;
	uint32_t cur_idx;
	uint32_t total_nodes;
};

uint32_t bvh4_write_flat_recursive(BVH4FlatWriterCtx* _ctx, IntermediateBVHNode* _node)
{
	KT_BVH_ASSERT(_ctx->cur_idx < _ctx->total_nodes);
	KT_BVH_ASSERT(!_node->is_leaf());

	uint32_t const node_idx = _ctx->cur_idx++;

	BVH4Node* flatnode = &_ctx->nodes[node_idx];

	for (uint32_t i = 0; i < _node->num_children - 1; ++i)
	{
		flatnode->split_axis[i] = uint8_t(_node->split_axis[i]);
	}

	for (uint32_t i = 0; i < _node->num_children; ++i)
	{
		IntermediateBVHNode* child = _node->children[i];
		// Write SoA AABB.
		for (uint32_t j = 0; j < 3; ++j)
		{
			flatnode->aabb_min_soa[j][i] = child->aabb_min[j];
			flatnode->aabb_max_soa[j][i] = child->aabb_max[j];
		}

		if (child->is_leaf())
		{
			KT_BVH_ASSERT(child->leaf_num_prims <= UINT16_MAX);
			flatnode->num_prims_in_leaf[i] = uint16_t(child->leaf_num_prims);
			flatnode->children[i] = child->leaf_prim_offset;
		}
		else
		{
			flatnode->num_prims_in_leaf[i] = 0;
			flatnode->children[i] = bvh4_write_flat_recursive(_ctx, child);
		}
	}

	for (uint32_t i = _node->num_children; i < 4; ++i)
	{
		// Empty node sentinel.
		flatnode->children[i] = UINT32_MAX;

		for (uint32_t j = 0; j < 4; ++j)
		{
			flatnode->aabb_max_soa[j][i] = NAN;
			flatnode->aabb_min_soa[j][i] = NAN;
		}
	}

	return node_idx;
}

bool bvh4_intermediate_to_flat(IntermediateBVH const* _bvh4, BVH4Node* o_nodes, uint32_t _node_cap)
{
	uint32_t const interior_node_count = _bvh4->total_nodes - _bvh4->total_leaf_nodes;
	if (_node_cap < interior_node_count || _bvh4->width != BVHWidth::BVH4)
	{
		return false;
	}

	BVH4FlatWriterCtx writer;
	writer.cur_idx = 0;
	writer.nodes = o_nodes;
	writer.total_nodes = interior_node_count;
	bvh4_write_flat_recursive(&writer, _bvh4->root);

	return true;
}

} // namespace kt_bvh


#if KT_BVH_COMPILER_MSVC
	#pragma warning(pop)
#endif