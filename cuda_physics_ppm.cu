#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// =====================
// Simulation Parameters
// =====================
#define N_PARTICLES 30000
#define GRID_SIZE 256
#define CELL_SIZE 0.05f
#define MAX_PARTS_PER_CELL 64

#define DT 0.016f
#define GRAVITY -9.8f
#define ITERATIONS 4
#define FRAMES 300

// =====================
// Image Parameters
// =====================
#define IMG_W 800
#define IMG_H 800
#define DUMP_EVERY 1   // dump every frame for video

// =====================
// Utility
// =====================
__device__ __forceinline__
int cell_index(int x, int y) {
    return y * GRID_SIZE + x;
}

// =====================
// Verlet Integration
// =====================
__global__ void integrate(float2* pos, float2* prev, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float2 p  = pos[i];
    float2 p0 = prev[i];

    float2 vel = { p.x - p0.x, p.y - p0.y };
    float2 acc = { 0.0f, GRAVITY };

    float2 next = {
        p.x + vel.x + acc.x * DT * DT,
        p.y + vel.y + acc.y * DT * DT
    };

    prev[i] = p;
    pos[i]  = next;
}

// =====================
// Clear Spatial Grid
// =====================
__global__ void clear_grid(int* cellCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < GRID_SIZE * GRID_SIZE)
        cellCount[i] = 0;
}

// =====================
// Build Spatial Grid
// =====================
__global__ void build_grid(
    float2* pos,
    int* cellCount,
    int* cellParticles,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float2 p = pos[i];

    int gx = (int)((p.x + 1.0f) / CELL_SIZE);
    int gy = (int)((p.y + 1.0f) / CELL_SIZE);

    if (gx < 0 || gy < 0 || gx >= GRID_SIZE || gy >= GRID_SIZE) return;

    int cell = cell_index(gx, gy);
    int idx  = atomicAdd(&cellCount[cell], 1);

    if (idx < MAX_PARTS_PER_CELL)
        cellParticles[cell * MAX_PARTS_PER_CELL + idx] = i;
}

// =====================
// Collision Solver
// =====================
__global__ void solve_collisions(
    float2* pos,
    int* cellCount,
    int* cellParticles,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float2 p = pos[i];
    int gx = (int)((p.x + 1.0f) / CELL_SIZE);
    int gy = (int)((p.y + 1.0f) / CELL_SIZE);

    const float minDist = 0.015f;

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int nx = gx + dx;
            int ny = gy + dy;
            if (nx < 0 || ny < 0 || nx >= GRID_SIZE || ny >= GRID_SIZE) continue;

            int cell = cell_index(nx, ny);
            int count = cellCount[cell];

            for (int j = 0; j < count; j++) {
                int other = cellParticles[cell * MAX_PARTS_PER_CELL + j];
                if (other == i) continue;

                float2 q = pos[other];
                float2 d = { p.x - q.x, p.y - q.y };
                float dist2 = d.x * d.x + d.y * d.y;

                if (dist2 > 0.0f && dist2 < minDist * minDist) {
                    float dist = sqrtf(dist2);
                    float overlap = 0.5f * (minDist - dist);
                    float inv = 1.0f / dist;

                    p.x += d.x * inv * overlap;
                    p.y += d.y * inv * overlap;
                }
            }
        }
    }
    pos[i] = p;
}

// =====================
// Boundary Constraints
// =====================
__global__ void apply_bounds(float2* pos, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float2 p = pos[i];
    p.x = fminf(1.0f, fmaxf(-1.0f, p.x));
    p.y = fminf(1.0f, fmaxf(-1.0f, p.y));
    pos[i] = p;
}

// =====================
// PPM Writer (Velocity Colored)
// =====================
void write_ppm(const char* filename, float2* pos, float2* prev, int n) {
    unsigned char* img =
        (unsigned char*)calloc(IMG_W * IMG_H * 3, 1);

    for (int i = 0; i < n; i++) {
        int x = (int)((pos[i].x + 1.0f) * 0.5f * IMG_W);
        int y = (int)((1.0f - (pos[i].y + 1.0f) * 0.5f) * IMG_H);

        if (x < 0 || x >= IMG_W || y < 0 || y >= IMG_H) continue;

        float vx = pos[i].x - prev[i].x;
        float vy = pos[i].y - prev[i].y;
        float speed = sqrtf(vx * vx + vy * vy);

        float t = fminf(speed * 20.0f, 1.0f);

        unsigned char r = (unsigned char)(255 * t);
        unsigned char g = (unsigned char)(255 * (1.0f - fabsf(t - 0.5f) * 2));
        unsigned char b = (unsigned char)(255 * (1.0f - t));

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int px = x + dx;
                int py = y + dy;
                if (px < 0 || px >= IMG_W || py < 0 || py >= IMG_H) continue;

                int idx = (py * IMG_W + px) * 3;
                img[idx]     = r;
                img[idx + 1] = g;
                img[idx + 2] = b;
            }
        }
    }

    FILE* f = fopen(filename, "wb");
    fprintf(f, "P6\n%d %d\n255\n", IMG_W, IMG_H);
    fwrite(img, 1, IMG_W * IMG_H * 3, f);
    fclose(f);
    free(img);
}

// =====================
// Main
// =====================
int main() {
    float2* h_pos  = (float2*)malloc(sizeof(float2) * N_PARTICLES);
    float2* h_prev = (float2*)malloc(sizeof(float2) * N_PARTICLES);

    for (int i = 0; i < N_PARTICLES; i++) {
        h_pos[i].x = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        h_pos[i].y = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        h_prev[i] = h_pos[i];
    }

    float2 *d_pos, *d_prev;
    int *d_cellCount, *d_cellParticles;

    cudaMalloc(&d_pos, sizeof(float2) * N_PARTICLES);
    cudaMalloc(&d_prev, sizeof(float2) * N_PARTICLES);
    cudaMalloc(&d_cellCount, sizeof(int) * GRID_SIZE * GRID_SIZE);
    cudaMalloc(&d_cellParticles,
        sizeof(int) * GRID_SIZE * GRID_SIZE * MAX_PARTS_PER_CELL);

    cudaMemcpy(d_pos, h_pos, sizeof(float2) * N_PARTICLES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev, h_prev, sizeof(float2) * N_PARTICLES, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 gridP((N_PARTICLES + block.x - 1) / block.x);
    dim3 gridG((GRID_SIZE * GRID_SIZE + block.x - 1) / block.x);

    for (int frame = 0; frame < FRAMES; frame++) {
        integrate<<<gridP, block>>>(d_pos, d_prev, N_PARTICLES);

        for (int it = 0; it < ITERATIONS; it++) {
            clear_grid<<<gridG, block>>>(d_cellCount);
            build_grid<<<gridP, block>>>(d_pos, d_cellCount, d_cellParticles, N_PARTICLES);
            solve_collisions<<<gridP, block>>>(d_pos, d_cellCount, d_cellParticles, N_PARTICLES);
            apply_bounds<<<gridP, block>>>(d_pos, N_PARTICLES);
        }

        cudaMemcpy(h_pos, d_pos, sizeof(float2) * N_PARTICLES, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_prev, d_prev, sizeof(float2) * N_PARTICLES, cudaMemcpyDeviceToHost);

        char name[64];
        sprintf(name, "frame_%04d.ppm", frame);
        write_ppm(name, h_pos, h_prev, N_PARTICLES);
    }

    cudaDeviceSynchronize();
    printf("Simulation complete.\n");

    cudaFree(d_pos);
    cudaFree(d_prev);
    cudaFree(d_cellCount);
    cudaFree(d_cellParticles);
    free(h_pos);
    free(h_prev);
    return 0;
}
