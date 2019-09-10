#include "kt_bvh.h"

#include <malloc.h> // malloc
#include <string.h> // memset, memcpy
#include <float.h> // FLT_MAX

#define KT_BVH_ALLOCA(_size) ::alloca(_size)

namespace kt_bvh
{

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

// Partition array such that all passing the predicate come first.
template <typename T, typename PredT>
static T* partition(T* _begin, T* _end, PredT _pred)
{
	T* swap_entry = _begin;
	while (swap_entry != _end && _pred(*swap_entry))
	{
		++swap_entry;
	}

	if (swap_entry == _end)
	{
		return swap_entry;
	}

	_begin = swap_entry + 1;
	
	while (_begin != _end)
	{
		if (_pred(*_begin))
		{
			swap(*_begin, *swap_entry);
			++swap_entry;
		}
		++_begin;
	}

	return swap_entry;
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

struct Vec3
{
	union
	{
		float data[3];

		struct  
		{
			float x;
			float y;
			float z;
		};
	};
};

static Vec3 operator-(Vec3 const& _lhs, Vec3 const& _rhs)
{
	return Vec3 {_lhs.x - _rhs.x, _lhs.y - _rhs.y, _lhs.z - _rhs.z};
}

static Vec3 operator+(Vec3 const& _lhs, Vec3 const& _rhs)
{
	return Vec3{ _lhs.x + _rhs.x, _lhs.y + _rhs.y, _lhs.z + _rhs.z };
}

static Vec3 operator*(Vec3 const& _v, float _scalar)
{
	return Vec3{ _v.x * _scalar, _v.y * _scalar, _v.z * _scalar};
}

static Vec3 operator/(Vec3 const& _v, float _scalar)
{
	float const rcp = 1.0f / _scalar;
	return Vec3{ _v.x * rcp, _v.y * rcp, _v.z * rcp };
}

static Vec3 min(Vec3 const& _lhs, Vec3 const& _rhs)
{
	Vec3 r;
	r.x = min(_lhs.x, _rhs.x);
	r.y = min(_lhs.y, _rhs.y);
	r.z = min(_lhs.z, _rhs.z);
	return r;
}

static Vec3 max(Vec3 const& _lhs, Vec3 const& _rhs)
{
	Vec3 r;
	r.x = max(_lhs.x, _rhs.x);
	r.y = max(_lhs.y, _rhs.y);
	r.z = max(_lhs.z, _rhs.z);
	return r;
}

static Vec3 vec3_splat(float _f)
{
	return Vec3{ _f, _f, _f };
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

static float aabb_surface_area(AABB const& _aabb)
{
	Vec3 const diag = _aabb.max - _aabb.min;
	return (diag.x * diag.y + diag.x * diag.z + diag.y * diag.z) * 2.0f;
}

struct MemArena
{
	struct ChunkHeader
	{
		ChunkHeader* next;
		uintptr_t cur_ptr;
		uintptr_t end_ptr;
	};
	
	static size_t const c_chunkSize = 1024 * 1024;

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

struct IntermediateBVH2Node
{
	bool is_leaf() const
	{
		return children[0] == nullptr;
	}

	AABB aabb;
	IntermediateBVH2Node* children[2];
	uint32_t split_axis;

	uint32_t leaf_prim_offset;
	uint32_t leaf_num_prims;
};

struct IntermediateBVH2
{
	MemArena arena;
	IntermediateBVH2Node *root;
	uint32_t total_nodes;

	dyn_pod_array<PrimitiveID> bvh_prim_id_list;
	uint32_t max_depth;
};

struct IntermediatePrimitive
{
	AABB aabb;
	Vec3 origin;
	PrimitiveID prim_id;
};

struct BVH2BuilderContext
{
	MemArena& arena()
	{
		return bvh2->arena;
	}

	IntermediateBVH2* bvh2;

	IntermediatePrimitive* primitive_info;
	uint32_t total_primitives;

	TriMesh const* meshes;
	uint32_t num_meshes;

	BVH2BuildDesc build_desc;
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

static void mesh_get_prim(TriMesh const& _mesh, uint32_t _prim_idx, Vec3* o_v0, Vec3* o_v1, Vec3* o_v2)
{
	switch (_mesh.index_type)
	{
		case TriMesh::IndexType::U16: mesh_get_prim_idx_u16(_mesh, _prim_idx, o_v0, o_v1, o_v2); break;
		case TriMesh::IndexType::U32: mesh_get_prim_idx_u32(_mesh, _prim_idx, o_v0, o_v1, o_v2); break;
		case TriMesh::IndexType::UnIndexed: mesh_get_prim_unindexed(_mesh, _prim_idx, o_v0, o_v1, o_v2); break;
	}
}

template <void (GetPrimT)(TriMesh const&, uint32_t, Vec3*, Vec3*, Vec3*)>
static void build_prim_info_impl(BVH2BuilderContext& _ctx, TriMesh const& _mesh, uint32_t _mesh_idx, IntermediatePrimitive* _prim_arr)
{
	for (uint32_t i = 0; i < _mesh.total_prims(); ++i)
	{
		Vec3 tri[3];
		GetPrimT(_mesh, i, &tri[0], &tri[1], &tri[2]);
		_prim_arr->aabb = AABB{ tri[0], tri[0] };
		_prim_arr->aabb = aabb_expand(_prim_arr->aabb, tri[1]);
		_prim_arr->aabb = aabb_expand(_prim_arr->aabb, tri[2]);
		_prim_arr->origin = aabb_center(_prim_arr->aabb);
		_prim_arr->prim_id.mesh_idx = _mesh_idx;
		_prim_arr->prim_id.mesh_prim_idx = i;
		++_prim_arr;
	}
}

static void build_prim_info(BVH2BuilderContext& _ctx)
{
	uint32_t prim_idx = 0;
	for (uint32_t mesh_idx = 0; mesh_idx < _ctx.num_meshes; ++mesh_idx)
	{
		TriMesh const& mesh = _ctx.meshes[mesh_idx];
		switch (mesh.index_type)
		{
			case TriMesh::IndexType::U16: build_prim_info_impl<mesh_get_prim_idx_u16>(_ctx, mesh, mesh_idx, _ctx.primitive_info + prim_idx); break;
			case TriMesh::IndexType::U32: build_prim_info_impl<mesh_get_prim_idx_u32>(_ctx, mesh, mesh_idx, _ctx.primitive_info + prim_idx); break;
			case TriMesh::IndexType::UnIndexed: build_prim_info_impl<mesh_get_prim_unindexed>(_ctx, mesh, mesh_idx, _ctx.primitive_info + prim_idx); break;
		}

		prim_idx += mesh.total_prims();
		KT_BVH_ASSERT(prim_idx <= _ctx.total_primitives);
	}

	KT_BVH_ASSERT(prim_idx == _ctx.total_primitives);
}

static void build_bvh2_leaf_node(BVH2BuilderContext& _ctx, IntermediateBVH2Node* _node, uint32_t _prim_begin, uint32_t _prim_end)
{
	KT_BVH_ASSERT(_prim_end > _prim_begin);
	uint32_t const nprims = _prim_end - _prim_begin;

	_node->leaf_prim_offset = _ctx.bvh2->bvh_prim_id_list.size;
	_node->leaf_num_prims = nprims;
	PrimitiveID* leaf_prims = _ctx.bvh2->bvh_prim_id_list.append_n(nprims);

	for (uint32_t i = _prim_begin; i < _prim_end; ++i)
	{
		*leaf_prims++ = _ctx.primitive_info[i].prim_id;
	}
}

struct SAHBucket
{
	AABB bounds = aabb_invalid();
	uint32_t num_prims = 0;
};

struct SAHBucketingInfo
{
	SAHBucketingInfo(uint32_t _num_buckets)
		: num_buckets(_num_buckets)
	{
		KT_BVH_ASSERT(_num_buckets < BVH2BuildDesc::c_sah_max_buckets);
	}

	uint32_t num_buckets;

	SAHBucket buckets[BVH2BuildDesc::c_sah_max_buckets];

	float split_costs[BVH2BuildDesc::c_sah_max_buckets - 1];
	AABB forward_split_bounds[BVH2BuildDesc::c_sah_max_buckets - 1];
	AABB backward_split_bounds[BVH2BuildDesc::c_sah_max_buckets - 1];

	uint32_t forward_prim_count[BVH2BuildDesc::c_sah_max_buckets - 1] = {};
	uint32_t backward_prim_count[BVH2BuildDesc::c_sah_max_buckets - 1] = {};
};

static uint32_t build_bvh2_split_sah(BVH2BuilderContext& _ctx, IntermediateBVH2Node* _node, uint32_t _split_axis, uint32_t _prim_begin, uint32_t _prim_end)
{
	SAHBucketingInfo bucket_info(_ctx.build_desc.sah_buckets);

	float const rcpAabbDim = 1.0f / (_node->aabb.max - _node->aabb.min).data[_split_axis];

	for (uint32_t prim_idx = _prim_begin; prim_idx < _prim_end; ++prim_idx)
	{
		IntermediatePrimitive const& prim = _ctx.primitive_info[prim_idx];
		float const project_to_dim = (prim.origin - _node->aabb.min).data[_split_axis] * rcpAabbDim;
		uint32_t const bucket = min(bucket_info.num_buckets - 1u, uint32_t(project_to_dim * bucket_info.num_buckets));
		bucket_info.buckets[bucket].num_prims++;
		bucket_info.buckets[bucket].bounds = aabb_union(prim.aabb, bucket_info.buckets[bucket].bounds);
	}

	uint32_t const num_splits = bucket_info.num_buckets - 1;

	bucket_info.forward_split_bounds[0] = bucket_info.buckets[0].bounds;
	bucket_info.forward_prim_count[0] = bucket_info.buckets[0].num_prims;

	for (int32_t i = 1; i < int32_t(num_splits); ++i)
	{
		bucket_info.forward_split_bounds[i] = aabb_union(bucket_info.forward_split_bounds[i - 1], bucket_info.buckets[i].bounds);
		bucket_info.forward_prim_count[i] = bucket_info.forward_prim_count[i - 1] + bucket_info.buckets[i].num_prims;
	}

	bucket_info.backward_split_bounds[num_splits - 1] = bucket_info.buckets[num_splits].bounds;
	bucket_info.backward_prim_count[num_splits - 1] = bucket_info.buckets[num_splits].num_prims;

	for (int32_t i = int32_t(num_splits) - 2; i >= 0; --i)
	{
		bucket_info.backward_split_bounds[i] = aabb_union(bucket_info.backward_split_bounds[i + 1], bucket_info.buckets[i + 1].bounds);
		bucket_info.backward_prim_count[i] = bucket_info.backward_prim_count[i + 1] + bucket_info.buckets[i + 1].num_prims;
	}

	float const traversal_cost = _ctx.build_desc.sah_traversal_cost;

	float const root_surface_area = aabb_surface_area(_node->aabb);

	for (uint32_t i = 0; i < num_splits; ++i)
	{
		uint32_t const countA = bucket_info.forward_prim_count[i];
		uint32_t const countB = bucket_info.backward_prim_count[i];

		float const sa = countA ? aabb_surface_area(bucket_info.forward_split_bounds[i]) : 0.0f;
		float const sb = countB ? aabb_surface_area(bucket_info.backward_split_bounds[i]) : 0.0f;
		bucket_info.split_costs[i] = traversal_cost + (sa * countA + sb * countB) / root_surface_area;
	}

	uint32_t best_bucket = 0;
	float best_cost = bucket_info.split_costs[0];
	for (uint32_t i = 1; i < num_splits; ++i)
	{
		if (bucket_info.split_costs[i] < best_cost)
		{
			best_cost = bucket_info.split_costs[i];
			best_bucket = i;
		}
	}

	float const leaf_cost = float(_prim_end - _prim_begin);
	uint32_t const nprims = _prim_end - _prim_begin;

	if (leaf_cost <= best_cost && nprims <= _ctx.build_desc.max_prims_per_leaf)
	{
		build_bvh2_leaf_node(_ctx, _node, _prim_begin, _prim_end);
		return UINT32_MAX;
	}

	IntermediatePrimitive* mid = partition(_ctx.primitive_info + _prim_begin, _ctx.primitive_info + _prim_end, 
	[rcpAabbDim, _split_axis, _node, &bucket_info, best_bucket](IntermediatePrimitive const& _prim) -> bool
	{
		float const project_to_dim = (_prim.origin - _node->aabb.min).data[_split_axis] * rcpAabbDim;
		uint32_t const bucket = min(bucket_info.num_buckets - 1u, uint32_t(project_to_dim * bucket_info.num_buckets));
		return bucket <= best_bucket;
	});

	uint32_t const mid_idx = uint32_t(mid - _ctx.primitive_info);
	return mid_idx;
}

static IntermediateBVH2Node* build_bvh2_recursive(BVH2BuilderContext& _ctx, uint32_t _depth, uint32_t _prim_begin, uint32_t _prim_end)
{
	KT_BVH_ASSERT(_prim_begin < _prim_end);

	_ctx.bvh2->max_depth = max(_ctx.bvh2->max_depth, _depth);

	IntermediateBVH2Node* node = _ctx.arena().alloc_uninitialized<IntermediateBVH2Node>();
	memset(node, 0, sizeof(IntermediateBVH2Node));
	_ctx.bvh2->total_nodes++;

	node->aabb = aabb_invalid();
	for (uint32_t i = _prim_begin; i < _prim_end; ++i)
	{
		node->aabb = aabb_union(node->aabb, _ctx.primitive_info[i].aabb);
	}

	Vec3 const aabb_range = node->aabb.max - node->aabb.min;
	uint32_t const nprims = _prim_end - _prim_begin;

	if (nprims == 1)
	{
		build_bvh2_leaf_node(_ctx, node, _prim_begin, _prim_end);
		return node;
	}

	uint32_t split_axis = 0;

	if (aabb_range.y > aabb_range.x)
	{
		split_axis = 1;
	}
	if (aabb_range.z > aabb_range.data[split_axis])
	{
		split_axis = 2;
	}

	node->split_axis = split_axis;

	switch (_ctx.build_desc.type)
	{
		case BVH2BuildType::TopDownBinnedSAH:
		{
			uint32_t const middle_idx = build_bvh2_split_sah(_ctx, node, split_axis, _prim_begin, _prim_end);
			if (middle_idx == UINT32_MAX)
			{
				return node;
			}

			if (_prim_begin != middle_idx && _prim_end != middle_idx)
			{
				node->children[0] = build_bvh2_recursive(_ctx, _depth + 1, _prim_begin, middle_idx);
				node->children[1] = build_bvh2_recursive(_ctx, _depth + 1, middle_idx, _prim_end);
				break;
			}

		} // fallthrough if binned SAH failed

		case BVH2BuildType::MedianSplit:
		{
			if (nprims <= _ctx.build_desc.max_prims_per_leaf)
			{
				build_bvh2_leaf_node(_ctx, node, _prim_begin, _prim_end);
				return node;
			}

			float const median = aabb_center(node->aabb).data[split_axis];

			IntermediatePrimitive* midPrim = partition(_ctx.primitive_info + _prim_begin, _ctx.primitive_info + _prim_end,
					  [median, split_axis](IntermediatePrimitive const& _val) { return _val.origin.data[split_axis] < median; });

			uint32_t middle_idx;

			middle_idx = uint32_t(midPrim - _ctx.primitive_info);
			if (middle_idx == _prim_begin || middle_idx == _prim_end)
			{
				// Obviously terrible since prims aren't sorted, fixme/remove.
				middle_idx = (_prim_begin + _prim_end) / 2;
			}

			node->children[0] = build_bvh2_recursive(_ctx, _depth + 1, _prim_begin, middle_idx);
			node->children[1] = build_bvh2_recursive(_ctx, _depth + 1, middle_idx, _prim_end);
		} break;
	}

	return node;
}

IntermediateBVH2* bvh2_build_intermediate(TriMesh const* _meshes, uint32_t _num_meshes, BVH2BuildDesc const& _build_desc, AllocHooks* alloc_hooks)
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

	IntermediateBVH2* bvh_intermediate = (IntermediateBVH2*)hooks.alloc_fn(hooks.ctx, sizeof(IntermediateBVH2));
	memset(bvh_intermediate, 0, sizeof(IntermediateBVH2));
	bvh_intermediate->arena.init(hooks);

	BVH2BuilderContext builder = {};

	builder.meshes = _meshes;
	builder.num_meshes = _num_meshes;
	builder.total_primitives = total_prims;
	builder.bvh2 = bvh_intermediate;
	builder.primitive_info = builder.arena().alloc_array_uninitialized<IntermediatePrimitive>(total_prims);
	builder.build_desc = _build_desc;

	bvh_intermediate->bvh_prim_id_list.init(&bvh_intermediate->arena);

	uint32_t const conservative_leaf_prim_cap = 5 * (total_prims / _build_desc.max_prims_per_leaf) / 4;
	bvh_intermediate->bvh_prim_id_list.ensure_cap(conservative_leaf_prim_cap);

	build_prim_info(builder);

	builder.bvh2->root = build_bvh2_recursive(builder, 0, 0, total_prims);

	return bvh_intermediate;
}

void bvh2_free_intermediate(IntermediateBVH2* _intermediate_bvh)
{
	if (!_intermediate_bvh)
	{
		return;
	}

	_intermediate_bvh->arena.free_all();
	_intermediate_bvh->arena.hooks.free_fn(_intermediate_bvh->arena.hooks.ctx, _intermediate_bvh);
}

void bvh2_get_primitive_id_array(IntermediateBVH2 const* _bvh2, PrimitiveID const** o_prim_array, uint32_t* o_prim_array_size)
{
	*o_prim_array = _bvh2->bvh_prim_id_list.mem;
	*o_prim_array_size = _bvh2->bvh_prim_id_list.size;
}

uint32_t bvh2_intermediate_num_nodes(IntermediateBVH2 const* _bvh2)
{
	return _bvh2->total_nodes;
}

uint32_t bvh2_depth(IntermediateBVH2 const* _bvh2)
{
	return _bvh2->max_depth;
}

struct BVH2FlatWriterCtx
{
	BVH2Node* nodes;
	uint32_t cur_idx;
	uint32_t total_nodes;
};

void bvh2_write_flat_depth_first(BVH2FlatWriterCtx* _ctx, IntermediateBVH2Node* _node)
{
	KT_BVH_ASSERT(_ctx->cur_idx < _ctx->total_nodes);

	BVH2Node* flatnode = &_ctx->nodes[_ctx->cur_idx++];
	memcpy(flatnode->aabb_min, &_node->aabb.min, sizeof(Vec3));
	memcpy(flatnode->aabb_max, &_node->aabb.max, sizeof(Vec3));
	flatnode->split_axis = uint16_t(_node->split_axis);
	
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

bool bvh2_intermediate_to_flat(IntermediateBVH2 const* _bvh2, BVH2Node* o_nodes, uint32_t _node_cap)
{
	if (_node_cap < _bvh2->total_nodes)
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

} // namespace kt_bvh