#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

using namespace std::chrono;

int div_ceil(int a, int b) { return (a + b - 1) / b; }

cudaError_t cudaStat;
cudaResourceDesc resDesc;
cudaTextureDesc texDesc;
cudaTextureObject_t tex_a = 0;
cudaTextureObject_t tex_b = 0;


#define CVTA_TO_SHARED_PTX(addr, smem_ptr)                                     \
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(addr) : "l"(smem_ptr));

#define LDG32_GUARD_PTX(reg, ptr, guard)                                       \
  {                                                                            \
    asm volatile("{.reg .pred p;\n\t"                                          \
                 "setp.ne.u32 p, %2, 0;\n\t"                                   \
                 "@p ld.global.f32 %0, [%1];}\n\t"                             \
                 : "=f"(reg)                                                   \
                 : "l"(ptr), "r"(guard));                                      \
  }

#define LDG32_GUARD_MOV0_PTX(reg, ptr, guard)                                  \
  {                                                                            \
    asm volatile("{.reg .pred p;\n\t"                                          \
                 "setp.ne.u32 p, %2, 0;\n\t"                                   \
                 "@!p mov.b32 %0, 0;\n\t"                                      \
                 "@p ld.global.f32 %0, [%1];}\n\t"                             \
                 : "=f"(reg)                                                   \
                 : "l"(ptr), "r"(guard));                                      \
  }

#define STS128_PTX(reg0, reg1, reg2, reg3, addr)                               \
  {                                                                            \
    asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n\t"                \
                 :                                                             \
                 : "l"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3));     \
  }

#define LDS128_PTX(reg0, reg1, reg2, reg3, addr)                               \
  {                                                                            \
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n\t"                \
                 : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)              \
                 : "l"(addr));                                                 \
  }

#define STS32_PTX(reg, addr)                                                   \
  { asm volatile("st.shared.f32 [%0], %1;\n" : : "l"(addr), "f"(reg)); }

#define STG32_GUARD_PTX(reg, ptr, guard)                                       \
  {                                                                            \
    asm volatile("{.reg .pred p;\n\t"                                          \
                 "setp.ne.u32 p, %2, 0;\n\t"                                   \
                 "@p st.global.f32 [%0], %1;}\n\t"                             \
                 :                                                             \
                 : "l"(ptr), "f"(reg), "r"(guard));                            \
  }

#define COMMIT_GROUP_PTX asm volatile("cp.async.commit_group;");

#define WAIT_GROUP_PTX(N) asm volatile("cp.async.wait_group %0;" : : "n"(N))

#define WAIT_ALL_PTX asm volatile("cp.async.wait_all ;")

#define CP_ASYNC_GUARD_PTX(addr, ptr, guard)                                   \
  {                                                                            \
    asm volatile("{.reg .pred p;\n\t"                                          \
                 "setp.ne.u32 p, %2, 0;\n\t"                                   \
                 "@p cp.async.ca.shared.global [%0], [%1], 4;}\n"              \
                 :                                                             \
                 : "l"(addr), "l"(ptr), "r"(guard));                           \
  }

#define CP_ASYNC_IGNORE_SRC_PTX(addr, ptr, guard)                              \
  {                                                                            \
    asm volatile("{.reg .pred p;\n\t"                                          \
                 "setp.eq.u32 p, %2, 0;\n\t"                                   \
                 "cp.async.ca.shared.global [%0], [%1], 4, p;}\n"              \
                 :                                                             \
                 : "l"(addr), "l"(ptr), "r"(guard));                           \
  }

__global__ __launch_bounds__(256, 2) void sgemm_texld_128x128x8(
    int m, int n, int k, const float alpha, cudaTextureObject_t tex_a, int lda,
    cudaTextureObject_t tex_b, int ldb, const float beta, float *C, int ldc) {
  // Operands A, B, C: row-major format

  const int smem_a_padding = 128;
  const int smem_a_size = smem_a_padding * 8;
  const int smem_a_ld = 128;
  const int smem_b_padding = 128;
  const int smem_b_size = smem_b_padding * 8;
  const int smem_b_ld = 128;

  __shared__ float __align__(2 * smem_a_size * sizeof(float))
      smem_ptr[2 * (smem_a_size + smem_b_size)];

  float accumulator[8][8]{};

  float4 texld_a_buffer;
  float4 texld_b_buffer;

  float *smem_a_ptr = smem_ptr;
  float *smem_b_ptr = smem_ptr + 2 * smem_a_size;

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  int texld_a_offset_x = threadIdx.x % 2;
  int texld_a_offset_y = blockIdx.y * 128 + threadIdx.x / 2;
  int texld_a_offset = texld_a_offset_x + texld_a_offset_y * lda / 4;

  int texld_b_offset_x = (blockIdx.x * 128) / 4 + threadIdx.x % 32;
  int texld_b_offset_y = threadIdx.x / 32;
  int texld_b_offset = texld_b_offset_x + texld_b_offset_y * ldb / 4;

  int sts_a_offset_x = threadIdx.x / 2;
  int sts_a_offset_y = 4 * (threadIdx.x % 2);
  int sts_a_offset = sts_a_offset_x + sts_a_offset_y * smem_a_ld;
  float *sts_a_ptr = smem_a_ptr + sts_a_offset;

  int sts_b_offset_x = 4 * (threadIdx.x % 32);
  int sts_b_offset_y = threadIdx.x / 32;
  int sts_b_offset = sts_b_offset_x + sts_b_offset_y * smem_b_ld;
  float *sts_b_ptr = smem_b_ptr + sts_b_offset;

  uint64_t sts_a_addr;
  uint64_t sts_b_addr;

  CVTA_TO_SHARED_PTX(sts_a_addr, sts_a_ptr);
  CVTA_TO_SHARED_PTX(sts_b_addr, sts_b_ptr);

  int n_blocks_k = (k + 7) / 8 - 1;

  texld_a_buffer = tex1Dfetch<float4>(tex_a, texld_a_offset);
  STS32_PTX(texld_a_buffer.x, sts_a_addr);
  STS32_PTX(texld_a_buffer.y, sts_a_addr + sizeof(float) * smem_a_ld);
  STS32_PTX(texld_a_buffer.z, sts_a_addr + 2 * sizeof(float) * smem_a_ld);
  STS32_PTX(texld_a_buffer.w, sts_a_addr + 3 * sizeof(float) * smem_a_ld);

  texld_b_buffer = tex1Dfetch<float4>(tex_b, texld_b_offset);
  STS128_PTX(texld_b_buffer.x, texld_b_buffer.y, texld_b_buffer.z,
             texld_b_buffer.w, sts_b_addr);
  __syncthreads();

  float frag_a[2][8];
  float frag_b[2][8];

  uint64_t lds_a_addr;
  uint64_t lds_b_addr;

  int lane_id_mapped_x = 2 * (lane_id / 8) + (lane_id % 2);
  int lane_id_mapped_y = (lane_id / 2) % 4;
  int warp_id_mapped_x = 64 * (warp_id % 2);
  int warp_id_mapped_y = 32 * (warp_id / 2);

  int lds_a_offset = 4 * lane_id_mapped_y + warp_id_mapped_y;
  int lds_b_offset = 4 * lane_id_mapped_x + warp_id_mapped_x;
  float *lds_a_ptr = smem_a_ptr + lds_a_offset;
  float *lds_b_ptr = smem_b_ptr + lds_b_offset;

  CVTA_TO_SHARED_PTX(lds_a_addr, lds_a_ptr);
  CVTA_TO_SHARED_PTX(lds_b_addr, lds_b_ptr);

  LDS128_PTX(frag_a[0][0], frag_a[0][1], frag_a[0][2], frag_a[0][3],
             lds_a_addr);
  LDS128_PTX(frag_a[0][4], frag_a[0][5], frag_a[0][6], frag_a[0][7],
             lds_a_addr + 16 * sizeof(float));
  LDS128_PTX(frag_b[0][0], frag_b[0][1], frag_b[0][2], frag_b[0][3],
             lds_b_addr);
  LDS128_PTX(frag_b[0][4], frag_b[0][5], frag_b[0][6], frag_b[0][7],
             lds_b_addr + 32 * sizeof(float));

  texld_a_offset += 2;
  texld_b_offset += 2 * n;
  sts_a_addr ^= 4096;
  sts_b_addr ^= 4096;

  for (int block_ks = 0; block_ks < n_blocks_k; block_ks++) {
    texld_a_buffer = tex1Dfetch<float4>(tex_a, texld_a_offset);
    texld_b_buffer = tex1Dfetch<float4>(tex_b, texld_b_offset);

#pragma unroll
    for (int warp_k = 0; warp_k < 7; warp_k += 1) {
      int prefetch = warp_k + 1;
      int frag_idx = warp_k & 1;
      int frag_next_idx = (warp_k + 1) & 1;
#pragma unroll
      for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
          accumulator[i][j] += frag_a[frag_idx][i] * frag_b[frag_idx][j];
        }
      }
      LDS128_PTX(frag_b[frag_next_idx][0], frag_b[frag_next_idx][1],
                 frag_b[frag_next_idx][2], frag_b[frag_next_idx][3],
                 lds_b_addr + prefetch * smem_b_ld * sizeof(float));
      LDS128_PTX(frag_b[frag_next_idx][4], frag_b[frag_next_idx][5],
                 frag_b[frag_next_idx][6], frag_b[frag_next_idx][7],
                 lds_b_addr + (prefetch * smem_b_ld + 32) * sizeof(float));
      LDS128_PTX(frag_a[frag_next_idx][0], frag_a[frag_next_idx][1],
                 frag_a[frag_next_idx][2], frag_a[frag_next_idx][3],
                 lds_a_addr + prefetch * smem_a_ld * sizeof(float));
      LDS128_PTX(frag_a[frag_next_idx][4], frag_a[frag_next_idx][5],
                 frag_a[frag_next_idx][6], frag_a[frag_next_idx][7],
                 lds_a_addr + (prefetch * smem_a_ld + 16) * sizeof(float));
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
      for (int j = 0; j < 8; j++) {
        accumulator[i][j] += frag_a[1][i] * frag_b[1][j];
      }
    }
    STS32_PTX(texld_a_buffer.x, sts_a_addr);
    STS32_PTX(texld_a_buffer.y, sts_a_addr + sizeof(float) * smem_a_ld);
    STS32_PTX(texld_a_buffer.z, sts_a_addr + 2 * sizeof(float) * smem_a_ld);
    STS32_PTX(texld_a_buffer.w, sts_a_addr + 3 * sizeof(float) * smem_a_ld);

    STS128_PTX(texld_b_buffer.x, texld_b_buffer.y, texld_b_buffer.z,
               texld_b_buffer.w, sts_b_addr);
    __syncthreads();

    sts_a_addr ^= 4096;
    sts_b_addr ^= 4096;
    lds_a_addr ^= 4096;
    lds_b_addr ^= 4096;

    LDS128_PTX(frag_a[0][0], frag_a[0][1], frag_a[0][2], frag_a[0][3],
               lds_a_addr);
    LDS128_PTX(frag_a[0][4], frag_a[0][5], frag_a[0][6], frag_a[0][7],
               lds_a_addr + 16 * sizeof(float));
    LDS128_PTX(frag_b[0][0], frag_b[0][1], frag_b[0][2], frag_b[0][3],
               lds_b_addr);
    LDS128_PTX(frag_b[0][4], frag_b[0][5], frag_b[0][6], frag_b[0][7],
               lds_b_addr + 32 * sizeof(float));

    texld_a_offset += 2;
    texld_b_offset += 2 * n;
  }

  // Compute last block
#pragma unroll
  for (int warp_k = 0; warp_k < 7; warp_k += 1) {
    int prefetch = warp_k + 1;
    int frag_idx = warp_k & 1;
    int frag_next_idx = (warp_k + 1) & 1;

#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
      for (int j = 0; j < 8; j++) {
        accumulator[i][j] += frag_a[frag_idx][i] * frag_b[frag_idx][j];
      }
    }

    LDS128_PTX(frag_b[frag_next_idx][0], frag_b[frag_next_idx][1],
               frag_b[frag_next_idx][2], frag_b[frag_next_idx][3],
               lds_b_addr + prefetch * smem_b_ld * sizeof(float));
    LDS128_PTX(frag_b[frag_next_idx][4], frag_b[frag_next_idx][5],
               frag_b[frag_next_idx][6], frag_b[frag_next_idx][7],
               lds_b_addr + (prefetch * smem_b_ld + 32) * sizeof(float));
    LDS128_PTX(frag_a[frag_next_idx][0], frag_a[frag_next_idx][1],
               frag_a[frag_next_idx][2], frag_a[frag_next_idx][3],
               lds_a_addr + prefetch * smem_a_ld * sizeof(float));
    LDS128_PTX(frag_a[frag_next_idx][4], frag_a[frag_next_idx][5],
               frag_a[frag_next_idx][6], frag_a[frag_next_idx][7],
               lds_a_addr + (prefetch * smem_a_ld + 16) * sizeof(float));
  }
#pragma unroll
  for (int i = 0; i < 8; i++) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      accumulator[i][j] += frag_a[1][i] * frag_b[1][j];
    }
  }

#pragma unroll
  for (int i = 0; i < 8; i++) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      accumulator[i][j] *= alpha;
    }
  }

  uint64_t sts_c_addr;
  int sts_c_offset =
      512 * warp_id + 4 * 32 * lane_id_mapped_y + 4 * lane_id_mapped_x;
  CVTA_TO_SHARED_PTX(sts_c_addr, smem_ptr + sts_c_offset);

  float *lds_c_ptr = (float *)((smem_ptr + 512 * warp_id + lane_id));

  int m_idx = blockIdx.y * 128 + warp_id_mapped_y;
  int n_idx = blockIdx.x * 128 + warp_id_mapped_x + lane_id;
  float *stg_c_ptr = C + m_idx * ldc + n_idx;

  if (m_idx < m) {
#pragma unroll 1
    for (int i = 0; i < 2; ++i) {
#pragma unroll 1
      for (int j = 0; j < 2; ++j) {
        __syncthreads();
#pragma unroll 2
        for (int p = 0; p < 4; ++p) {
          STS128_PTX(accumulator[i * 4 + p][j * 4],
                     accumulator[i * 4 + p][j * 4 + 1],
                     accumulator[i * 4 + p][j * 4 + 2],
                     accumulator[i * 4 + p][j * 4 + 3],
                     sts_c_addr + p * 8 * sizeof(float4));
        }
        __syncthreads();
#pragma unroll 4
        for (int p = 0; p < 16; ++p) {
          int m_edge = m - (m_idx + i * 16);
          int n_pos = n_idx + j * 32;
          bool guard = p < m_edge && n_pos < n;
          if (beta != 0) {
            float c;
            LDG32_GUARD_MOV0_PTX(c, stg_c_ptr + (i * 16 + p) * n + j * 32,
                                 (unsigned)guard);
            c *= beta;
            STG32_GUARD_PTX(c + lds_c_ptr[p * 32],
                            stg_c_ptr + (i * 16 + p) * n + j * 32,
                            (unsigned)guard);
          } else {
            STG32_GUARD_PTX(lds_c_ptr[p * 32],
                            stg_c_ptr + (i * 16 + p) * n + j * 32,
                            (unsigned)guard);
          }
        }
      }
    }
  }
}

__global__ __launch_bounds__(256, 2) void sgemm_128x128x8(
    int m, int n, int k, const float alpha, const float *A, int lda,
    const float *B, int ldb, const float beta, float *C, int ldc) {
  // Operands A, B, C: row-major format

  // Abbreviations:
  // ldg - load global
  // lds - load shared
  // stg - store global
  // sts - store shared
  // cvta - convert address

  const int smem_a_padding = 256;
  const int smem_a_size = smem_a_padding * 8;
  const int smem_a_ld = 132; // leading dimension
  const int smem_b_padding = 128;
  const int smem_b_size = smem_b_padding * 8;
  const int smem_b_ld = 128; // leading dimension

  __shared__ float __align__(2 * smem_a_size * sizeof(float))
      smem_ptr[2 * (smem_a_size + smem_b_size)];

  // C accumulator
  float accumulator[8][8]{};

  // Registers for (global memory -> shared memory) transfers
  float ldg_a_buffer[4];
  float ldg_b_buffer[4];

  // Bitmasks to track in-bounds and out-of-bounds global memory reads
  unsigned ldg_a_bitmask = 0x0;
  unsigned ldg_b_bitmask = 0x0;

  float *smem_a_ptr = smem_ptr;
  float *smem_b_ptr = smem_ptr + 2 * smem_a_size;

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  int ldg_a_start_x = threadIdx.x % 8;
  int ldg_a_start_y = blockIdx.y * 128 + 4 * (threadIdx.x / 8);
  int ldg_a_start = ldg_a_start_x + ldg_a_start_y * lda;
  const float *ldg_a_ptr = A + ldg_a_start;
  int ldg_a_offsets_y[4];
  int ldg_a_offsets[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    ldg_a_offsets_y[i] = i;
  }
#pragma unroll
  for (int i = 0; i < 4; i++) {
    ldg_a_offsets[i] = ldg_a_offsets_y[i] * lda;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int m_idx = ldg_a_start_y + ldg_a_offsets_y[i];
    // if global memory access is in-bounds, flip corresponding bit
    if (m_idx < m) {
      ldg_a_bitmask ^= (0x1 << i);
    }
  }

  int ldg_b_start_x = blockIdx.x * 128 + threadIdx.x % 32;
  int ldg_b_start_y = threadIdx.x / 32;
  int ldg_b_start = ldg_b_start_x + ldg_b_start_y * ldb;
  const float *ldg_b_ptr = B + ldg_b_start;
  int ldg_b_offsets_x[4];
  int ldg_b_offsets[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    ldg_b_offsets_x[i] = 32 * i;
  }
#pragma unroll
  for (int i = 0; i < 4; i++) {
    ldg_b_offsets[i] = ldg_b_offsets_x[i];
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int n_idx = ldg_b_start_x + ldg_b_offsets_x[i];
    // if global memory access is in-bounds, flip corresponding bit
    if (n_idx < n) {
      ldg_b_bitmask ^= (0x1 << i);
    }
  }

  int sts_a_start_x = 4 * (threadIdx.x / 8);
  int sts_a_start_y = threadIdx.x % 8;
  int sts_a_start = sts_a_start_x + sts_a_start_y * smem_a_ld;
  float *sts_a_ptr = smem_a_ptr + sts_a_start;

  int sts_b_start_x = threadIdx.x % 32;
  int sts_b_start_y = threadIdx.x / 32;
  int sts_b_start = sts_b_start_x + sts_b_start_y * smem_b_ld;
  float *sts_b_ptr = smem_b_ptr + sts_b_start;
  int sts_b_offsets[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    sts_b_offsets[i] = 32 * i;
  }

  uint64_t sts_a_addr;
  uint64_t sts_b_addr;

  // Convert from generic to .shared state space
  CVTA_TO_SHARED_PTX(sts_a_addr, sts_a_ptr);
  CVTA_TO_SHARED_PTX(sts_b_addr, sts_b_ptr);

  // if (k % 8 == 0) {n_blocks_k = k/8 - 1} else {n_blocks_k = k/8;}
  int n_blocks_k = (k + 7) / 8 - 1;
  int first_block_k_size = k - 8 * n_blocks_k;

  // Load first blocks from global memory to shared memory
  // {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    bool guard_k = ldg_a_start_x < first_block_k_size;
    bool guard_m = ldg_a_bitmask & (0x1 << i);
    bool guard = guard_k && guard_m;
    LDG32_GUARD_MOV0_PTX(ldg_a_buffer[i], ldg_a_ptr + ldg_a_offsets[i],
                         (unsigned)guard);
  }
  STS128_PTX(ldg_a_buffer[0], ldg_a_buffer[1], ldg_a_buffer[2], ldg_a_buffer[3],
             sts_a_addr);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    bool guard_k = ldg_b_start_y < first_block_k_size;
    bool guard_n = ldg_b_bitmask & (0x1 << i);
    bool guard = guard_k && guard_n;
    LDG32_GUARD_MOV0_PTX(ldg_b_buffer[i], ldg_b_ptr + ldg_b_offsets[i],
                         (unsigned)guard);
  }
#pragma unroll
  for (int i = 0; i < 4; i += 1) {
    STS32_PTX(ldg_b_buffer[i], sts_b_addr + sts_b_offsets[i] * sizeof(float));
  }
  __syncthreads();
  // }

  float frag_a[2][8];
  float frag_b[2][8];

  uint64_t lds_a_addr;
  uint64_t lds_b_addr;

  int lane_id_mapped_x = 2 * (lane_id / 8) + (lane_id % 2);
  int lane_id_mapped_y = (lane_id / 2) % 4;
  int warp_id_mapped_x = 64 * (warp_id % 2);
  int warp_id_mapped_y = 32 * (warp_id / 2);

  int lds_a_start = 4 * lane_id_mapped_y + warp_id_mapped_y;
  int lds_b_start = 4 * lane_id_mapped_x + warp_id_mapped_x;
  float *lds_a_ptr = smem_a_ptr + lds_a_start;
  float *lds_b_ptr = smem_b_ptr + lds_b_start;

  // Convert from generic to .shared state space
  CVTA_TO_SHARED_PTX(lds_a_addr, lds_a_ptr);
  CVTA_TO_SHARED_PTX(lds_b_addr, lds_b_ptr);

  // Load first fragments from shared memory
  // {
  LDS128_PTX(frag_a[0][0], frag_a[0][1], frag_a[0][2], frag_a[0][3],
             lds_a_addr);
  LDS128_PTX(frag_a[0][4], frag_a[0][5], frag_a[0][6], frag_a[0][7],
             lds_a_addr + 16 * sizeof(float));
  LDS128_PTX(frag_b[0][0], frag_b[0][1], frag_b[0][2], frag_b[0][3],
             lds_b_addr);
  LDS128_PTX(frag_b[0][4], frag_b[0][5], frag_b[0][6], frag_b[0][7],
             lds_b_addr + 32 * sizeof(float));
  // }

  // Move pointers to next blocks
  ldg_a_ptr += first_block_k_size;
  ldg_b_ptr += first_block_k_size * ldb;

  // Switch shared memory buffers
  sts_a_addr ^= 8192;
  sts_b_addr ^= 4096;

  // Iterate over k and divide into ks blocks
  for (int block_k = 0; block_k < n_blocks_k; block_k++) {

    // Prefetch next blocks from global memory
    // {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      bool guard_m = (ldg_a_bitmask & (0x1 << i));
      LDG32_GUARD_PTX(ldg_a_buffer[i], ldg_a_ptr + ldg_a_offsets[i],
                      (unsigned)guard_m);

      bool guard_n = (ldg_b_bitmask & (0x1 << i));
      LDG32_GUARD_PTX(ldg_b_buffer[i], ldg_b_ptr + ldg_b_offsets[i],
                      (unsigned)guard_n);
    }
    // }
#pragma unroll
    for (int warp_k = 0; warp_k < 8; warp_k += 1) {
      int prefetch = (warp_k + 1) % 8;
      int frag_idx = warp_k & 1;
      int frag_next_idx = (warp_k + 1) & 1;

      // Prefetch next fragments from shared memory
      // {
      LDS128_PTX(frag_a[frag_next_idx][0], frag_a[frag_next_idx][1],
                 frag_a[frag_next_idx][2], frag_a[frag_next_idx][3],
                 lds_a_addr + prefetch * smem_a_ld * sizeof(float));
      LDS128_PTX(frag_a[frag_next_idx][4], frag_a[frag_next_idx][5],
                 frag_a[frag_next_idx][6], frag_a[frag_next_idx][7],
                 lds_a_addr + (prefetch * smem_a_ld + 16) * sizeof(float));
      LDS128_PTX(frag_b[frag_next_idx][0], frag_b[frag_next_idx][1],
                 frag_b[frag_next_idx][2], frag_b[frag_next_idx][3],
                 lds_b_addr + prefetch * smem_b_ld * sizeof(float));
      LDS128_PTX(frag_b[frag_next_idx][4], frag_b[frag_next_idx][5],
                 frag_b[frag_next_idx][6], frag_b[frag_next_idx][7],
                 lds_b_addr + (prefetch * smem_b_ld + 32) * sizeof(float));
      // }

      // Update the accumulator
      // {
#pragma unroll
      for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
          accumulator[i][j] += frag_a[frag_idx][i] * frag_b[frag_idx][j];
        }
      }
      // }
    }

    // Store prefetched blocks to shared memory
    // {
    STS128_PTX(ldg_a_buffer[0], ldg_a_buffer[1], ldg_a_buffer[2],
               ldg_a_buffer[3], sts_a_addr);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      STS32_PTX(ldg_b_buffer[i], sts_b_addr + i * 32 * sizeof(float));
    }
    __syncthreads();
    // }

    // Switch shared memory buffers
    sts_a_addr ^= 8192;
    sts_b_addr ^= 4096;
    lds_a_addr ^= 8192;
    lds_b_addr ^= 4096;

    // Move pointers to next blocks
    ldg_a_ptr += 8;
    ldg_b_ptr += 8 * n;

    // Load first fragments from shared memory
    // {
    LDS128_PTX(frag_a[0][0], frag_a[0][1], frag_a[0][2], frag_a[0][3],
               lds_a_addr);
    LDS128_PTX(frag_a[0][4], frag_a[0][5], frag_a[0][6], frag_a[0][7],
               lds_a_addr + 16 * sizeof(float));
    LDS128_PTX(frag_b[0][0], frag_b[0][1], frag_b[0][2], frag_b[0][3],
               lds_b_addr);
    LDS128_PTX(frag_b[0][4], frag_b[0][5], frag_b[0][6], frag_b[0][7],
               lds_b_addr + 32 * sizeof(float));
    // }
  }

  // Compute last block
  // {
#pragma unroll
  for (int warp_k = 0; warp_k < 8; warp_k += 1) {
    int prefetch = (warp_k + 1) % 8;
    int frag_idx = warp_k & 1;
    int frag_next_idx = (warp_k + 1) & 1;

    LDS128_PTX(frag_a[frag_next_idx][0], frag_a[frag_next_idx][1],
               frag_a[frag_next_idx][2], frag_a[frag_next_idx][3],
               lds_a_addr + prefetch * smem_a_ld * sizeof(float));
    LDS128_PTX(frag_a[frag_next_idx][4], frag_a[frag_next_idx][5],
               frag_a[frag_next_idx][6], frag_a[frag_next_idx][7],
               lds_a_addr + (prefetch * smem_a_ld + 16) * sizeof(float));
    LDS128_PTX(frag_b[frag_next_idx][0], frag_b[frag_next_idx][1],
               frag_b[frag_next_idx][2], frag_b[frag_next_idx][3],
               lds_b_addr + prefetch * smem_b_ld * sizeof(float));
    LDS128_PTX(frag_b[frag_next_idx][4], frag_b[frag_next_idx][5],
               frag_b[frag_next_idx][6], frag_b[frag_next_idx][7],
               lds_b_addr + (prefetch * smem_b_ld + 32) * sizeof(float));

#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
      for (int j = 0; j < 8; j++) {
        accumulator[i][j] += frag_a[frag_idx][i] * frag_b[frag_idx][j];
      }
    }
  }
  // }

  // Calculate alpha * A * B
  // {
#pragma unroll
  for (int i = 0; i < 8; i++) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      accumulator[i][j] *= alpha;
    }
  }
  // }

  // Store the accumulator to global memory
  // {
  uint64_t sts_c_addr;
  int sts_c_offset =
      512 * warp_id + 4 * 32 * lane_id_mapped_y + 4 * lane_id_mapped_x;
  CVTA_TO_SHARED_PTX(sts_c_addr, smem_ptr + sts_c_offset);

  float *lds_c_ptr = (float *)((smem_ptr + 512 * warp_id + lane_id));

  int m_idx = blockIdx.y * 128 + warp_id_mapped_y;
  int n_idx = blockIdx.x * 128 + warp_id_mapped_x + lane_id;
  float *stg_c_ptr = C + m_idx * ldc + n_idx;

  if (m_idx < m) {
#pragma unroll 1
    for (int i = 0; i < 2; ++i) {
#pragma unroll 1
      for (int j = 0; j < 2; ++j) {
        __syncthreads();
#pragma unroll 2
        for (int p = 0; p < 4; ++p) {
          STS128_PTX(accumulator[i * 4 + p][j * 4],
                     accumulator[i * 4 + p][j * 4 + 1],
                     accumulator[i * 4 + p][j * 4 + 2],
                     accumulator[i * 4 + p][j * 4 + 3],
                     sts_c_addr + p * 8 * sizeof(float4));
        }
        __syncthreads();
#pragma unroll 4
        for (int p = 0; p < 16; ++p) {
          int m_edge = m - (m_idx + i * 16);
          int n_pos = n_idx + j * 32;
          bool guard = p < m_edge && n_pos < n;
          // if (beta != 0.0) {compute (beta*C + accumulator) and write to
          // global memory}
          if (beta != 0) {
            float c;
            LDG32_GUARD_MOV0_PTX(c, stg_c_ptr + (i * 16 + p) * n + j * 32,
                                 (unsigned)guard);
            c *= beta;
            STG32_GUARD_PTX(c + lds_c_ptr[p * 32],
                            stg_c_ptr + (i * 16 + p) * n + j * 32,
                            (unsigned)guard);
          }
          // if (beta == 0.0) {directly store the accumulator to global memory}
          else {
            STG32_GUARD_PTX(lds_c_ptr[p * 32],
                            stg_c_ptr + (i * 16 + p) * n + j * 32,
                            (unsigned)guard);
          }
        }
      }
    }
  }
  // }
}

void sgemm(int m, int n, int k, const float *alpha, float *A, int lda, float *B,
           int ldb, const float *beta, float *C, int ldc) {
  // C := alpha*A*B + beta*C
  // Operands A, B, C: row-major format
  // For column-major order, swap (A, lda) and (B, ldb) because C^T = B^T * A^T.

#if GPUCC >= 80
  static_assert(
      __CUDACC_VER_MAJOR__ > 11 ||
          ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 5)),
      "For devices with compute compatibility >= 80, install CUDA >= 11.5");
  dim3 grid;
  dim3 threads(256);
  grid.y = div_ceil(m, 128);
  if (m > 2500 || n > 2500) {
    grid.x = div_ceil(n, 256);
    sgemm_128x256x8<<<grid, threads>>>(m, n, k, *alpha, A, lda, B, ldb, *beta,
                                       C, ldc);
  } else {
    grid.x = div_ceil(n, 128);
    sgemm_128x128x8<<<grid, threads>>>(m, n, k, *alpha, A, lda, B, ldb, *beta,
                                       C, ldc);
  }
#else
  dim3 grid;
  dim3 threads(256);
  grid.x = div_ceil(n, 128);
  grid.y = div_ceil(m, 128);
  bool is_aligned =
      (((unsigned)lda & 3u) == 0) && (((unsigned)ldb & 3u) == 0) &&
      (((unsigned long)A & 15u) == 0) && (((unsigned long)B & 15u) == 0);
  if (is_aligned) {
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;
    resDesc.res.linear.desc.y = 32;
    resDesc.res.linear.desc.z = 32;
    resDesc.res.linear.desc.w = 32;
    resDesc.res.linear.devPtr = A;
    resDesc.res.linear.sizeInBytes = m * lda * sizeof(float);
    cudaCreateTextureObject(&tex_a, &resDesc, &texDesc, NULL);
    resDesc.res.linear.devPtr = B;
    resDesc.res.linear.sizeInBytes = k * ldb * sizeof(float);
    cudaCreateTextureObject(&tex_b, &resDesc, &texDesc, NULL);
    sgemm_texld_128x128x8<<<grid, threads>>>(m, n, k, *alpha, tex_a, lda, tex_b,
        ldb, *beta, C, ldc);
        cudaDestroyTextureObject(tex_a);
        cudaDestroyTextureObject(tex_b);
    } else {
        sgemm_128x128x8<<<grid, threads>>>(m, n, k, *alpha, A, lda, B, ldb, *beta,
            C, ldc);
        }
    #endif
}

// Error checking macro for CUDA calls
#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
inline void check_cuda_error(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(result) << " at " << file << ":" << line << " '" << func << "'" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Error checking macro for cuBLAS calls
#define CHECK_CUBLAS_ERROR(val) check_cublas_error((val), #val, __FILE__, __LINE__)
inline void check_cublas_error(cublasStatus_t result, const char* func, const char* file, int line) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error: " << result << " at " << file << ":" << line << " '" << func << "'" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// CUDA kernel to center data by subtracting column means
__global__ void centerDataKernel(float* data, float* means, int ny, int nx) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < nx) {
        float mean_val = means[col];
        for (int row = 0; row < ny; row++) {
            data[row * nx + col] -= mean_val;
        }
    }
}

// CUDA kernel to normalize data by standard deviation
__global__ void normalizeDataKernel(float* data, float* stds, int ny, int nx) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < nx) {
        float std_val = stds[col];
        // Handle zero standard deviation
        if (std_val == 0.0f) {
            std_val = 1.0f;
        }
        
        for (int row = 0; row < ny; row++) {
            data[row * nx + col] /= std_val;
        }
    }
}

// CUDA kernel to set diagonal elements to 1.0
__global__ void setDiagonalKernel(float* matrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        matrix[idx * n + idx] = 1.0f;
    }
}


__global__ void transposeOptimized(float *input, float *output, int nx, int ny) {
  __shared__ float tile[32][33];  // +1 to avoid bank conflicts
  
  int bx = blockIdx.x * blockDim.x;
  int by = blockIdx.y * blockDim.y;
  int ix = bx + threadIdx.x;
  int iy = by + threadIdx.y;
  
  // Load data from input global memory to shared memory
  if (ix < nx && iy < ny) {
      tile[threadIdx.y][threadIdx.x] = input[iy * nx + ix];
  }
  
  __syncthreads();
  
  // Compute transposed indices
  int tx = by + threadIdx.x;
  int ty = bx + threadIdx.y;
  
  // Write back to global memory with transposed indices
  if (tx < ny && ty < nx) {
      output[ty * ny + tx] = tile[threadIdx.x][threadIdx.y];
  }
}

// Helper function to initialize matrix
void initMatrix(float *matrix, int nx, int ny) {
  for (int i = 0; i < ny; i++) {
      for (int j = 0; j < nx; j++) {
          matrix[i * nx + j] = i * nx + j;
      }
  }
}

// Helper function to verify transpose
bool verifyTranspose(float *input, float *output, int nx, int ny) {
  for (int i = 0; i < ny; i++) {
      for (int j = 0; j < nx; j++) {
          if (output[j * ny + i] != input[i * nx + j]) {
              return false;
          }
      }
  }
  return true;
}

/**
 * Fast GPU-based Pearson Correlation Coefficient calculation
 * 
 * @param ny Number of samples (rows)
 * @param nx Number of features (columns)
 * @param data Input matrix (host memory)
 * @param result Output correlation matrix (host memory, must be pre-allocated)
 */
void correlate(int ny, int nx, const float* data, float* result) {
    // Initialize cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    
    // Allocate device memory for input data
    float* d_data;
    size_t size_data = ny * nx * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, size_data));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, data, size_data, cudaMemcpyHostToDevice));
    
    // Allocate device memory for means and standard deviations
    float* d_means;
    float* d_stds;
    CHECK_CUDA_ERROR(cudaMalloc(&d_means, nx * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_stds, nx * sizeof(float)));
    
    // Calculate means
    float* h_means = new float[nx]();
    for (int j = 0; j < nx; j++) {
        float sum = 0.0f;
        for (int i = 0; i < ny; i++) {
            sum += data[i * nx + j];
        }
        h_means[j] = sum / ny;
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_means, h_means, nx * sizeof(float), cudaMemcpyHostToDevice));
    
    // Calculate standard deviations
    float* h_stds = new float[nx]();
    for (int j = 0; j < nx; j++) {
        float sum_sq = 0.0f;
        for (int i = 0; i < ny; i++) {
            float diff = data[i * nx + j] - h_means[j];
            sum_sq += diff * diff;
        }
        h_stds[j] = sqrt(sum_sq / (ny - 1));
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_stds, h_stds, nx * sizeof(float), cudaMemcpyHostToDevice));
    
    // Center data
    int blockSize = 256;
    int numBlocks = (nx + blockSize - 1) / blockSize;
    centerDataKernel<<<numBlocks, blockSize>>>(d_data, d_means, ny, nx);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Normalize data
    normalizeDataKernel<<<numBlocks, blockSize>>>(d_data, d_stds, ny, nx);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Allocate device memory for correlation matrix
    float* d_result;
    size_t size_result = nx * nx * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, size_result));
    
    // Compute correlation matrix using cuBLAS: result = data^T * data / (ny - 1)
    float alpha = 1.0f;
    float beta = 0.0f;

    float* d_data_t;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data_t, size_data));

    dim3 block_t(32, 32);
    dim3 grid_t((nx + block_t.x - 1) / block_t.x, 
                 (ny + block_t.y - 1) / block_t.y);
    transposeOptimized<<<grid_t, block_t >>>(d_data, d_data_t, ny, nx);
    sgemm(ny, ny, nx, &alpha, d_data, ny, d_data_t, ny, &beta, d_result, nx);

    // CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
    //   nx, nx, ny, 
    //   &alpha, d_data, nx, d_data, nx, 
    //   &beta, d_result, nx));
    
    // Set diagonal elements to 1.0 to handle numerical precision issues
    setDiagonalKernel<<<numBlocks, blockSize>>>(d_result, nx);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(result, d_result, size_result, cudaMemcpyDeviceToHost));
    
    // Free allocated memory
    delete[] h_means;
    delete[] h_stds;
    cudaFree(d_data);
    cudaFree(d_data_t);
    cudaFree(d_means);
    cudaFree(d_stds);
    cudaFree(d_result);
    cublasDestroy(handle);
}

// Example usage
int main() {
    // Example parameters
    int ny = 3;  // Number of samples (rows)
    int nx = 2;   // Number of features (columns)
    
    // Create sample data
    std::vector<float> data(ny * nx);
    for (int i = 0; i < ny * nx; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate memory for result
    std::vector<float> result(nx * nx);
    
    // Calculate correlations
    std::cout << "Calculating PCC correlation matrix..." << std::endl;

    auto start = high_resolution_clock::now();
    correlate(ny, nx, data.data(), result.data());
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "Runtime: " << duration.count() << std::endl;
    // Print a small part of the result for verification
    std::cout << "First few values of the correlation matrix:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << result[i * nx + j] << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}