#include <malloc.h>
#include <xmmintrin.h>

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif

#include <math.h>
#include <float.h>
#include <chrono>
#include <utility>

#include "kt_bvh.h"

#define FAST_OBJ_IMPLEMENTATION
#include "fast_obj.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

bool g_bvh4 = true;

struct TimeAccumulator
{
	char const* name;
	uint64_t nanos;
	uint32_t invocations;
};

static TimeAccumulator s_bvhBuildTime = { "bvh_build", 0, 0 };
static TimeAccumulator s_bvhTraverseTime = { "bvh_traverse", 0, 0 };
static TimeAccumulator s_intersectTime = { "tri_intersect", 0, 0 };

#ifdef _MSC_VER
	extern "C" unsigned char _BitScanForward(unsigned long * _Index, unsigned long _Mask);
	#pragma intrinsic(_BitScanForward)
#endif

static uint32_t find_first_set_lsb(uint32_t _v)
{
#ifdef _MSC_VER
	unsigned long idx;
	return ::_BitScanForward(&idx, _v) ? idx : 32;
#else
	return __builtin_ctz(_v);
#endif
}

struct ScopedPerfTimer
{
	ScopedPerfTimer(TimeAccumulator* _accum)
		: accum(_accum)
	{
		begin = std::chrono::high_resolution_clock::now();
	}

	~ScopedPerfTimer()
	{
		auto now = std::chrono::high_resolution_clock::now();
		accum->nanos += std::chrono::duration_cast<std::chrono::nanoseconds>(now - begin).count();
		accum->invocations++;
	}

	TimeAccumulator* accum;
	std::chrono::high_resolution_clock::time_point begin;
};

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
	return Vec3{ _lhs.x - _rhs.x, _lhs.y - _rhs.y, _lhs.z - _rhs.z };
}

static Vec3 operator+(Vec3 const& _lhs, Vec3 const& _rhs)
{
	return Vec3{ _lhs.x + _rhs.x, _lhs.y + _rhs.y, _lhs.z + _rhs.z };
}

static Vec3 operator*(Vec3 const& _v, float _scalar)
{
	return Vec3{ _v.x * _scalar, _v.y * _scalar, _v.z * _scalar };
}

static Vec3 operator/(Vec3 const& _v, float _scalar)
{
	float const rcp = 1.0f / _scalar;
	return Vec3{ _v.x * rcp, _v.y * rcp, _v.z * rcp };
}

static Vec3 vec3_splat(float _f)
{
	return Vec3{ _f, _f, _f };
}

static float vec3_dot(Vec3 const& _lhs, Vec3 const& _rhs)
{
	return _lhs.x * _rhs.x + _lhs.y * _rhs.y + _lhs.z * _rhs.z;
}

static Vec3 vec3_cross(Vec3 const& _lhs, Vec3 const& _rhs)
{
	return Vec3{ _lhs.y * _rhs.z - _lhs.z * _rhs.y, _lhs.z * _rhs.x - _lhs.x * _rhs.z, _lhs.x * _rhs.y - _lhs.y * _rhs.x };
}

static Vec3 vec3_norm(Vec3 const& _vec)
{
	return _vec / sqrtf(vec3_dot(_vec, _vec));
}

static uint32_t const c_width = 1920;
static uint32_t const c_height = 1080;


struct TracerCtx
{
	~TracerCtx()
	{
		fast_obj_destroy(mesh);
		free(bvh4);
		free(pos_indices);
		free(prim_id_buf);
		free(image);
	}

	fastObjMesh* mesh = nullptr;
	uint32_t* pos_indices = nullptr;
	uint32_t* prim_id_buf = nullptr;
	uint8_t* image = nullptr;

	union 
	{
		kt_bvh::BVH4Node* bvh4 = nullptr;
		kt_bvh::BVH2Node* bvh2;
	};

};

struct Ray
{
	void set(Vec3 const& _o, Vec3 const& _d)
	{
		o = _o;
		d = _d;
		rcp_d = Vec3{1.0f / _d.x, 1.0f / _d.y, 1.0f / _d.z};

		for (uint32_t i = 0; i < 3; ++i)
		{
			rcp_d_x4[i] = _mm_set1_ps(rcp_d.data[i]);
			o_x4[i] = _mm_set1_ps(o.data[i]);
		}
	}

	Vec3 o;
	Vec3 d;
	Vec3 rcp_d;

	__m128 o_x4[3];
	__m128 rcp_d_x4[3];
};

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

bool intersect_ray_aabb(Ray const& _ray, float const* _aabb_min, float const* _aabb_max, float* o_tmin)
{
	float tmin = -INFINITY;
	float tmax = INFINITY;

	for (uint32_t i = 0; i < 3; ++i)
	{
		float const t0 = (_aabb_min[i] - _ray.o.data[i]) * _ray.rcp_d.data[i];
		float const t1 = (_aabb_max[i] - _ray.o.data[i]) * _ray.rcp_d.data[i];
		tmin = max(tmin, min(t0, t1));
		tmax = min(tmax, max(t0, t1));
	}

	if (tmin <= tmax)
	{
		*o_tmin = tmin;
		return true;
	}

	return false;
}

__m128 intersect_ray_aabb_x4_soa(Ray const& _ray, float const* _aabb_min_x4_soa, float const* _aabb_max_x4_soa, __m128* o_tmin)
{
	__m128 tmin = _mm_set1_ps(-INFINITY);
	__m128 tmax = _mm_set1_ps(INFINITY);

	for (uint32_t i = 0; i < 3; ++i)
	{
		__m128 const ro = _ray.o_x4[i];
		__m128 const rcp_d = _ray.rcp_d_x4[i];

		__m128 const aabb_min = _mm_load_ps(_aabb_min_x4_soa + 4 * i);
		__m128 const aabb_max = _mm_load_ps(_aabb_max_x4_soa + 4 * i);

		__m128 const t0 = _mm_mul_ps(_mm_sub_ps(aabb_min, ro), rcp_d);
		__m128 const t1 = _mm_mul_ps(_mm_sub_ps(aabb_max, ro), rcp_d);

		tmin = _mm_max_ps(tmin, _mm_min_ps(t0, t1));
		tmax = _mm_min_ps(tmax, _mm_max_ps(t0, t1));
	}

	*o_tmin = tmin;

	return _mm_cmple_ps(tmin, tmax);
}


bool intersect_ray_tri(Ray const& _ray, Vec3 const& _v0, Vec3 const& _v1, Vec3 const& _v2, float* o_t, float* o_u, float* o_v)
{
	Vec3 const v01 = _v1 - _v0;
	Vec3 const v02 = _v2 - _v0;

	Vec3 const pvec = vec3_cross(_ray.d, v02);
	float const det = vec3_dot(pvec, v01);

	if (det < 0.000001f)
	{
		return false;
	}

	float const idet = 1.0f / det;

	Vec3 const tvec = _ray.o - _v0;
	float const u = vec3_dot(tvec, pvec) * idet;
	if (u < 0.0f || u > 1.0f) return false;

	Vec3 const qvec = vec3_cross(tvec, v01);
	float const v = vec3_dot(_ray.d, qvec) * idet;
	if (v < 0.0f || u + v > 1.0f) return false;

	*o_t = vec3_dot(v02, qvec) * idet;
	*o_v = v;
	*o_u = u;
	return true;
}

struct Camera
{
	void init(Vec3 const& _origin, Vec3 const& _lookAt, float _fov, float _aspect, float _screen_dist)
	{
		static Vec3 const world_up = Vec3{ 0.0f, 1.0f, 0.0f };

		Vec3 const fwd = vec3_norm(_lookAt - _origin);
		Vec3 const right = vec3_norm(vec3_cross(fwd, world_up));
		Vec3 const up = vec3_norm(vec3_cross(right, fwd));
	
		basis[0] = right;
		basis[1] = up;
		basis[2] = fwd;

		float const heightOverTwo = tanf(_fov * float(M_PI) / 360.0f);
		float const widthOverTwo = heightOverTwo * _aspect;

		origin = _origin;

		screen_bottom_left_corner = origin - basis[0] * widthOverTwo - basis[1] * heightOverTwo + basis[2] * _screen_dist;

		screen_u = basis[0] * widthOverTwo * 2.0f;
		screen_v = basis[1] * heightOverTwo * 2.0f;
	}

	Ray get_ray(float _u, float _v)
	{
		Ray r;
		Vec3 const point_on_screen = screen_bottom_left_corner + screen_u * _u + screen_v * _v;
		r.set(origin, vec3_norm(point_on_screen - origin));
		return r;
	}

	Vec3 origin;

	Vec3 screen_bottom_left_corner;
	Vec3 screen_u;
	Vec3 screen_v;

	Vec3 basis[3];
};

struct Intersection
{
	uint32_t prim_idx = UINT32_MAX;
	float u, v;
	float t = FLT_MAX;
};


Intersection trace_bvh4(TracerCtx const& _ctx, Ray const& _ray)
{
	Intersection result;

	uint32_t stack[128];
	uint32_t stack_size = 0;

	ScopedPerfTimer isectTime(&s_bvhTraverseTime);

	uint32_t const ray_dir_is_neg[3] = { _ray.d.x < 0.0f, _ray.d.y < 0.0f, _ray.d.z < 0.0f };
	uint32_t node_idx = 0;

	do
	{
		kt_bvh::BVH4Node const& node = _ctx.bvh4[node_idx];
		__m128 aabb_tmin;
		__m128 const mask = intersect_ray_aabb_x4_soa(_ray, (float*)node.aabb_min_soa, (float*)node.aabb_max_soa, &aabb_tmin);

		uint32_t leaves_to_visit = _mm_movemask_ps(_mm_and_ps(mask, _mm_cmpge_ps(_mm_set1_ps(result.t), aabb_tmin)));

		uint32_t sortedIndices[4] = { 0, 1, 2, 3 };

		uint32_t did_sort = 0;
		if (!ray_dir_is_neg[node.split_axis[1]])
		{
			swap(sortedIndices[0], sortedIndices[2]);
			swap(sortedIndices[1], sortedIndices[3]);
			did_sort = 1;
		}

		if (!ray_dir_is_neg[node.split_axis[2 * (did_sort ^ 1)]])
		{
			swap(sortedIndices[2], sortedIndices[3]);
		}

		if (!ray_dir_is_neg[node.split_axis[2 * did_sort]])
		{
			swap(sortedIndices[0], sortedIndices[1]);
		}

		for (uint32_t i = 0; i < 4; ++i)
		{
			uint32_t const idx = sortedIndices[i];
			if (((1 << idx) & leaves_to_visit))
			{
				if (node.is_child_leaf(idx))
				{
					ScopedPerfTimer isectTime(&s_intersectTime);

					uint32_t* prims = _ctx.prim_id_buf + node.children[idx];
					for (uint32_t i = 0; i < node.num_prims_in_leaf[idx]; ++i)
					{
						uint32_t const face_idx = prims[i];
						Vec3 const& v0 = *(((Vec3*)_ctx.mesh->positions) + _ctx.mesh->indices[face_idx * 3].p);
						Vec3 const& v1 = *(((Vec3*)_ctx.mesh->positions) + _ctx.mesh->indices[face_idx * 3 + 1].p);
						Vec3 const& v2 = *(((Vec3*)_ctx.mesh->positions) + _ctx.mesh->indices[face_idx * 3 + 2].p);

						float local_t, local_u, local_v;
						if (intersect_ray_tri(_ray, v0, v1, v2, &local_t, &local_u, &local_v) && local_t < result.t)
						{
							result.t = local_t;
							result.u = local_u;
							result.v = local_v;
							result.prim_idx = face_idx;
						}
					}
				}
				else
				{
					assert(stack_size < sizeof(stack) / sizeof(*stack));
					assert(node.children[i] != UINT32_MAX);
					stack[stack_size++] = node.children[idx];
				}
			}
		}

		if (!stack_size)
		{
			break;
		}

		node_idx = stack[--stack_size];
	} while (true);

	return result;
}

Intersection trace_bvh2(TracerCtx const& _ctx, Ray const& _ray)
{
	Intersection result;

	uint32_t stack[128];
	uint32_t stack_size = 0;
	uint32_t node_idx = 0;

	ScopedPerfTimer isectTime(&s_bvhTraverseTime);

	uint32_t const ray_dir_is_neg[3] = { _ray.d.x < 0.0f, _ray.d.y < 0.0f, _ray.d.z < 0.0f };

	do
	{
		kt_bvh::BVH2Node const& node = _ctx.bvh2[node_idx];
		float aaab_tmin;
		if (intersect_ray_aabb(_ray, node.aabb_min, node.aabb_max, &aaab_tmin) && result.t >= aaab_tmin)
		{
			if (node.is_leaf())
			{
				ScopedPerfTimer isectTime(&s_intersectTime);

				uint32_t* prims = _ctx.prim_id_buf + node.right_child_or_prim_offset;
				for (uint32_t i = 0; i < node.num_prims_in_leaf; ++i)
				{
					uint32_t const face_idx = prims[i];
					Vec3 const& v0 = *(((Vec3*)_ctx.mesh->positions) + _ctx.mesh->indices[face_idx * 3].p);
					Vec3 const& v1 = *(((Vec3*)_ctx.mesh->positions) + _ctx.mesh->indices[face_idx * 3 + 1].p);
					Vec3 const& v2 = *(((Vec3*)_ctx.mesh->positions) + _ctx.mesh->indices[face_idx * 3 + 2].p);

					float local_t, local_u, local_v;
					if (intersect_ray_tri(_ray, v0, v1, v2, &local_t, &local_u, &local_v) && local_t < result.t)
					{
						result.t = local_t;
						result.u = local_u;
						result.v = local_v;
						result.prim_idx = face_idx;
					}
				}
			}
			else
			{
				assert(stack_size < (sizeof(stack) / sizeof(*stack)));
				if (ray_dir_is_neg[node.split_axis])
				{
					stack[stack_size++] = node_idx + 1;
					node_idx = node.right_child_or_prim_offset;
				}
				else
				{
					node_idx = node_idx + 1;
					stack[stack_size++] = node.right_child_or_prim_offset;
				}
				continue;
			}
		}

		if (!stack_size)
		{
			break;
		}

		node_idx = stack[--stack_size];
	} while (true);

	return result;
}

uint32_t trace_test(TracerCtx const& _ctx, Ray const& _ray)
{
	Intersection const isect = g_bvh4 ? trace_bvh4(_ctx, _ray) : trace_bvh2(_ctx, _ray);

	if (isect.prim_idx != UINT32_MAX)
	{
		Vec3 const& n0 = *(((Vec3*)_ctx.mesh->normals) + _ctx.mesh->indices[isect.prim_idx * 3].n);
		Vec3 const& n1 = *(((Vec3*)_ctx.mesh->normals) + _ctx.mesh->indices[isect.prim_idx * 3 + 1].n);
		Vec3 const& n2 = *(((Vec3*)_ctx.mesh->normals) + _ctx.mesh->indices[isect.prim_idx * 3 + 2].n);

		Vec3 const interp = vec3_norm(n0 * isect.u + n1 * isect.v + n2 * (1.0f - isect.u - isect.v)) * 0.5f + vec3_splat(0.5f);

		union
		{
			uint8_t u8[4];
			uint32_t u32;
		} colour; 

		for (uint32_t i = 0; i < 3; ++i)
		{
			colour.u8[i] = uint8_t(min(255u, uint32_t(interp.data[i] * 255.0f)));
		}
		colour.u8[3] = 0xff;

		return colour.u32;
	}
	else
	{
		return 0xFF000000;
	}
}

static void print_perf_timer(TimeAccumulator const& accum)
{
	double const micros = double(accum.nanos) / (1000.0);
	double const millis = micros / (1000.0);

	if (accum.invocations > 1)
	{
		printf("%s took total %.2fms over %u invocations (avg per invocation: %.2fus)\n", accum.name, millis, accum.invocations, micros / double(accum.invocations));
	}
	else
	{
		printf("%s took total %.2fms over %u invocations\n", accum.name, millis, accum.invocations);
	}
}

int main(int argc, char** _argv)
{
	TracerCtx ctx;

	ctx.mesh = fast_obj_read("models/living_room/living_room.obj");

	// Build linear index array
	ctx.pos_indices = (uint32_t*)malloc(sizeof(uint32_t) * ctx.mesh->face_count * 3);

	{
		uint32_t* pnext = ctx.pos_indices;

		for (uint32_t i = 0; i < ctx.mesh->face_count; ++i)
		{
			assert(ctx.mesh->face_vertices[i] == 3);
			*pnext++ = ctx.mesh->indices[i * 3].p;
			*pnext++ = ctx.mesh->indices[i * 3 + 1].p;
			*pnext++ = ctx.mesh->indices[i * 3 + 2].p;
		}
	}

	kt_bvh::TriMesh tri_mesh;
	tri_mesh.set_indices(ctx.pos_indices, ctx.mesh->face_count);
	tri_mesh.set_vertices(ctx.mesh->positions, sizeof(float[3]), ctx.mesh->position_count);
	kt_bvh::IntermediateBVH* bvh2;
	{
		ScopedPerfTimer isectTime(&s_bvhBuildTime);
		kt_bvh::BVHBuildDesc desc;
		//desc.set_median_split(4);
		desc.set_binned_sah(0.85f, 16);
		desc.width = g_bvh4 ? kt_bvh::BVHWidth::BVH4 : kt_bvh::BVHWidth::BVH2;

		bvh2 = kt_bvh::bvh_build_intermediate(&tri_mesh, 1, desc);

		kt_bvh::BVHBuildResult const result = kt_bvh::bvh_build_result(bvh2);

		if (g_bvh4)
		{
			ctx.bvh4 = (kt_bvh::BVH4Node*)malloc(result.total_interior_nodes * sizeof(kt_bvh::BVH4Node));
			kt_bvh::bvh4_intermediate_to_flat(bvh2, ctx.bvh4, result.total_interior_nodes);
		}
		else
		{
			ctx.bvh2 = (kt_bvh::BVH2Node*)malloc(result.total_nodes * sizeof(kt_bvh::BVH2Node));
			kt_bvh::bvh2_intermediate_to_flat(bvh2, ctx.bvh2, result.total_nodes);
		}


		// Write out primitive id without mesh id.
		ctx.prim_id_buf = (uint32_t*)malloc(sizeof(uint32_t) * result.prim_id_array_size);
		for (uint32_t i = 0; i < result.prim_id_array_size; ++i)
		{
			ctx.prim_id_buf[i] = result.prim_id_array[i].mesh_prim_idx;
		}
	}


	ctx.image = (uint8_t*)malloc(sizeof(uint32_t) * c_width * c_height);

	Camera cam;
	//cam.init(Vec3{ .5f, 0.5f, .5f }, vec3_splat(0.0f), 70.0f, float(c_width) / float(c_height), 1.5f);
	cam.init(Vec3{ 0.0f, 1.5f, 0.5f}, Vec3{0.0f, 1.5f, 2.0f}, 70.0f, float(c_width) / float(c_height), 0.5f);

	uint8_t* image_ptr = ctx.image;
	for (uint32_t y = 0; y < c_height; ++y)
	{
		for (uint32_t x = 0; x < c_width; ++x)
		{
			Ray const& ray = cam.get_ray(x / float(c_width), y / float(c_height));
			uint32_t const col = trace_test(ctx, ray);
			memcpy(image_ptr, &col, sizeof(uint32_t));
			image_ptr += 4;
		}
	}

	stbi_flip_vertically_on_write(1);
	stbi_write_bmp("image.bmp", c_width, c_height, 4, ctx.image);

	kt_bvh::bvh_free_intermediate(bvh2);

	print_perf_timer(s_bvhBuildTime);
	print_perf_timer(s_bvhTraverseTime);
	print_perf_timer(s_intersectTime);
}