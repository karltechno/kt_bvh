#include <malloc.h>

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <float.h>

#include "kt_bvh.h"

#define FAST_OBJ_IMPLEMENTATION
#include "fast_obj.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


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

static uint32_t const c_width = 256;
static uint32_t const c_height = 256;


struct TracerCtx
{
	~TracerCtx()
	{
		fast_obj_destroy(mesh);
		free(nodes);
		free(pos_indices);
		free(prim_id_buf);
		free(image);
	}

	fastObjMesh* mesh = nullptr;
	kt_bvh::BVH2Node* nodes = nullptr;
	uint32_t* pos_indices = nullptr;
	uint32_t* prim_id_buf = nullptr;
	uint8_t* image = nullptr;
};

struct Ray
{
	void set(Vec3 const& _o, Vec3 const& _d)
	{
		o = _o;
		d = _d;
		rcp_d = Vec3{1.0f / _d.x, 1.0f / _d.y, 1.0f / _d.z};
	}

	Vec3 o;
	Vec3 d;
	Vec3 rcp_d;
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

bool intersect_ray_aabb(Ray const& _ray, float const* _aabb_min, float const* _aabb_max)
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

	return tmin <= tmax;
}

bool intersect_ray_tri(Ray const& _ray, Vec3 const& _v0, Vec3 const& _v1, Vec3 const& _v2, float* o_t, float* o_u, float* o_v)
{
	Vec3 const v01 = _v1 - _v0;
	Vec3 const v02 = _v2 - _v0;

	Vec3 const pvec = vec3_cross(_ray.d, v02);
	float const det = vec3_dot(pvec, v01);

	if (fabsf(det) < 0.0001f)
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

		screen_bottom_left_corner = origin - basis[0] * widthOverTwo - basis[1] * heightOverTwo - basis[2] * _screen_dist;

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

uint32_t hits = 0;

uint32_t trace_test(TracerCtx const& _ctx, Ray const& _ray)
{
	uint32_t stack[64];
	uint32_t stack_size = 0;

	float t = FLT_MAX;
	float u, v;
	uint32_t best_prim_idx = UINT32_MAX;

	uint32_t node_idx = 0;
	do
	{
		kt_bvh::BVH2Node const& node = _ctx.nodes[node_idx];

		if (intersect_ray_aabb(_ray, node.aabb_min, node.aabb_max))
		{
			if (node.is_leaf())
			{
				uint32_t* prims = _ctx.prim_id_buf + node.right_child_or_prim_offset;
				for (uint32_t i = 0; i < node.num_prims_in_leaf; ++i)
				{
					uint32_t const face_idx = prims[i];
					Vec3 const& v0 = *(((Vec3*)_ctx.mesh->positions) + _ctx.mesh->indices[face_idx * 3].p);
					Vec3 const& v1 = *(((Vec3*)_ctx.mesh->positions) + _ctx.mesh->indices[face_idx * 3 + 1].p);
					Vec3 const& v2 = *(((Vec3*)_ctx.mesh->positions) + _ctx.mesh->indices[face_idx * 3 + 2].p);

					float local_t, local_u, local_v;
					if (intersect_ray_tri(_ray, v0, v1, v2, &local_t, &local_u, &local_v) && local_t < t)
					{
						t = local_t;
						u = local_u;
						v = local_v;
						best_prim_idx = face_idx;
					}
				}
			}
			else
			{
				assert(stack_size < (sizeof(stack) / sizeof(*stack)));
				node_idx = node_idx + 1;
				stack[stack_size++] = node.right_child_or_prim_offset;
				continue;
			}
		}

		if (!stack_size)
		{
			break;
		}

		node_idx = stack[--stack_size];

	} while (true);

	if (best_prim_idx != UINT32_MAX)
	{
		++hits;
		return 0xFF0000FF;
	}
	else
	{
		return 0xFF000000;
	}
}

int main(int argc, char** _argv)
{
	TracerCtx ctx;

	ctx.mesh = fast_obj_read("models/cube.obj");

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

	kt_bvh::IntermediateBVH2* bvh2 = kt_bvh::bvh2_build_intermediate(&tri_mesh, 1, kt_bvh::BVH2BuildDesc::defult_desc());

	uint32_t const num_nodes = kt_bvh::bvh2_intermediate_num_nodes(bvh2);
	ctx.nodes = (kt_bvh::BVH2Node*)malloc(num_nodes * sizeof(kt_bvh::BVH2Node));

	kt_bvh::bvh2_intermediate_to_flat(bvh2, ctx.nodes, num_nodes);

	// Write out primitive id without mesh id.
	{
		kt_bvh::PrimitiveID const* ids;
		uint32_t num_ids;
		kt_bvh::bvh2_get_primitive_id_array(bvh2, &ids, &num_ids);
		ctx.prim_id_buf = (uint32_t*)malloc(sizeof(uint32_t) * num_ids);
		for (uint32_t i = 0; i < num_ids; ++i)
		{
			ctx.prim_id_buf[i] = ids[i].mesh_prim_idx;
		}
	}

	ctx.image = (uint8_t*)malloc(sizeof(uint32_t) * c_width * c_height);


	Camera cam;
	cam.init(Vec3{ 0.9f, 0.2f, 2.0f }, vec3_splat(0.0f), 50.0f, float(c_width) / float(c_height), 0.5f);
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

	stbi_write_bmp("image.png", c_width, c_height, 4, ctx.image);

	kt_bvh::bvh2_free_intermediate(bvh2);
}