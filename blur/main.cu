#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <limits>

#define IDX(i, j, m) ((i) * (m) + (j))

using uchar = unsigned char;

const size_t BLOCK_SIZE = 16;

const int BLUR_KERNEL_WIDTH = 9;
const float BLUR_KERNEL_SIGMA = 4;
const float PI = acos(-1);

template <typename T>
void allocate_on_gpu(T** data, size_t n, size_t m) {
  cudaMalloc(data, sizeof(T) * n * m);
}

template <typename T>
void memset_on_gpu(T* data, size_t n, size_t m) {
  cudaMemset(data, 0, sizeof(T) * n * m);
}

template <typename T>
void copy_to_gpu(const T* const from, T* to, size_t n, size_t m) {
  cudaMemcpy(to, from, sizeof(T) * n * m, cudaMemcpyHostToDevice);
}

template <typename T>
void copy_to_host(const T* const from, T* to, size_t n, size_t m) {
  cudaMemcpy(to, from, sizeof(T) * n * m, cudaMemcpyDeviceToHost);
}

template <typename T>
void free_on_gpu(T* data) {
  cudaFree(data);
}

template <typename T>
void allocate(T** data, size_t n, size_t m) {
  *data = (T*)malloc(sizeof(T) * n * m);
}

template <typename T>
T sqr(T x) {
  return x * x;
}

void read_rgb(size_t height, size_t width, size_t max_height, size_t max_width,
               uchar** red, uchar** green, uchar** blue)
{
   for (uchar** channel : {red, green, blue}) {
     allocate(channel, min(height, max_height), min(width, max_width));
     for (size_t i = 0; i < height; ++i) {
       for (size_t j = 0; j < width; ++j) {
         int value;
         std::cin >> value;
         if (i < max_height && j < max_width) {
           (*channel)[IDX(i, j, max_width)] = value;
         }
       }
     }
   }
}

void get_filter_gpu(float** device_filter) {
  float* filter;
  allocate(&filter, BLUR_KERNEL_WIDTH, BLUR_KERNEL_WIDTH);
  const int half_width = BLUR_KERNEL_WIDTH / 2;

  float sum = 0;
  for (int i = -half_width; i <= half_width; ++i) {
    for (int j = -half_width; j <= half_width; ++j) {
      const float value = expf(-(sqr(i) + sqr(j)) / (2 * sqr(BLUR_KERNEL_SIGMA))) / (2 * PI * sqr(BLUR_KERNEL_SIGMA));
      filter[IDX(i + half_width, j + half_width, BLUR_KERNEL_WIDTH)] = value;
      sum += value;
    }
  }

  for (int i = -half_width; i <= half_width; ++i) {
    for (int j = -half_width; j <= half_width; ++j) {
      filter[IDX(i + half_width, j + half_width, BLUR_KERNEL_WIDTH)] /= sum;
    }
  }

  allocate_on_gpu(device_filter, BLUR_KERNEL_WIDTH, BLUR_KERNEL_WIDTH);
  copy_to_gpu(filter, *device_filter, BLUR_KERNEL_WIDTH, BLUR_KERNEL_WIDTH);
}

__device__ size_t get_pos(int initial_pos, int delta, int bound) {
  int result = initial_pos + delta - BLUR_KERNEL_WIDTH / 2;
  if (result < 0) return 0;
  if (result >= bound) return bound - 1;
  return result;
}

__global__ void gaussian_blur(const uchar* const input, uchar* const output, size_t height, size_t width, const float* const filter) {
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= height || col >= width) {
    return;
  }

  float value = 0.0;
  for (size_t i = 0; i < BLUR_KERNEL_WIDTH; ++i) {
    for (size_t j = 0; j < BLUR_KERNEL_WIDTH; ++j) {
      const size_t image_row = get_pos(row, i, height);
      const size_t image_col = get_pos(col, j, width);
      value += filter[IDX(i, j, BLUR_KERNEL_WIDTH)] * input[IDX(image_row, image_col, width)];
    }
  }

  output[IDX(row, col, width)] = value;  
}

void blur_image_gpu_channel(size_t height, size_t width, const uchar* const input_matrix, const float* const filter, uchar** device_output) {
  uchar* device_input;
  allocate_on_gpu(&device_input, height, width);
  copy_to_gpu(input_matrix, device_input, height, width);

  allocate_on_gpu(device_output, height, width);
  memset_on_gpu(*device_output, height, width);

  const dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
  const dim3 dim_grid((width + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);

  gaussian_blur<<<dim_grid, dim_block>>>(device_input, *device_output, height, width, filter);
  free_on_gpu(device_input);
}

void blur_image_channel(size_t height, size_t width, const uchar* const input, const float* const filter, uchar** output) {
  uchar* device_blured;
  blur_image_gpu_channel(height, width, input, filter, &device_blured);

  cudaDeviceSynchronize();

  allocate(output, height, width);
  copy_to_host(device_blured, *output, height, width);

  free_on_gpu(device_blured);
}

void blur_image_gpu(size_t height, size_t width, const uchar* const input_red, const uchar* const input_green, const uchar* const input_blue,
                    uchar** output_red, uchar** output_green, uchar** output_blue)
{
  float* device_filter;
  get_filter_gpu(&device_filter);

  cudaDeviceSynchronize();

  blur_image_channel(height, width, input_red, device_filter, output_red);
  blur_image_channel(height, width, input_green, device_filter, output_green);
  blur_image_channel(height, width, input_blue, device_filter, output_blue);
}

void get_filter_cpu(float** filter) {
  allocate(filter, BLUR_KERNEL_WIDTH, BLUR_KERNEL_WIDTH);
  const int half_width = BLUR_KERNEL_WIDTH / 2;

  float sum = 0;
  for (int i = -half_width; i <= half_width; ++i) {
    for (int j = -half_width; j <= half_width; ++j) {
      const float value = expf(-(sqr(i) + sqr(j)) / (2 * sqr(BLUR_KERNEL_SIGMA))) / (2 * PI * sqr(BLUR_KERNEL_SIGMA));
      (*filter)[IDX(i + half_width, j + half_width, BLUR_KERNEL_WIDTH)] = value;
      sum += value;
    }
  }

  for (int i = -half_width; i <= half_width; ++i) {
    for (int j = -half_width; j <= half_width; ++j) {
      (*filter)[IDX(i + half_width, j + half_width, BLUR_KERNEL_WIDTH)] /= sum;
    }
  }
}

void blur_image_cpu_channel(size_t height, size_t width, const uchar* const input, const float* const filter, uchar** output) {
  allocate(output, height, width);

  auto get_pos = [](int initial_pos, int delta, int bound) -> int {
    int result = initial_pos + delta - BLUR_KERNEL_WIDTH / 2;
    if (result < 0) return 0;
    if (result >= bound) return bound - 1;
    return result;
  };

  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      float value = 0.0;
      for (size_t i = 0; i < BLUR_KERNEL_WIDTH; ++i) {
        for (size_t j = 0; j < BLUR_KERNEL_WIDTH; ++j) {
          const size_t image_row = get_pos(row, i, height);
          const size_t image_col = get_pos(col, j, width);
          value += filter[IDX(i, j, BLUR_KERNEL_WIDTH)] * input[IDX(image_row, image_col, width)];
        }
      }
      (*output)[IDX(row, col, width)] = value;
    }
  }
}

void blur_image_cpu(size_t height, size_t width, const uchar* const input_red, const uchar* const input_green, const uchar* const input_blue,
                    uchar** output_red, uchar** output_green, uchar** output_blue)
{
  float* filter;
  get_filter_cpu(&filter);

  blur_image_cpu_channel(height, width, input_red, filter, output_red);
  blur_image_cpu_channel(height, width, input_green, filter, output_green);
  blur_image_cpu_channel(height, width, input_blue, filter, output_blue);

  free(filter);
}

void print_channel(uchar* channel, size_t height, size_t width) {
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      std::cout << int(channel[IDX(i, j, width)]) << ' ';
    }
    std::cout << '\n';
  }
}

void assert_equal(const uchar* const cpu, const uchar* const gpu, size_t height, size_t width) {
  unsigned int diffs = 0;
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      diffs += abs(int(cpu[IDX(row, col, width)]) - int(gpu[IDX(row, col, width)])) > 1;
    }
  }
  if (diffs != 0) {
    std::cerr << "Number of pixels with diff: " << diffs << std::endl;
    assert(false);
  }
}

int main(int argc, const char* argv[]) {
  if (argc <= 1) {
    std::cerr << "usage: ./<binary> rgb.txt [height] [width]" << std::endl;
    return 1;
  }
  freopen(argv[1], "r", stdin);

  size_t max_height = std::numeric_limits<size_t>::max();
  if (argc >= 3) {
    max_height = std::atoi(argv[2]);
  }
  size_t max_width = std::numeric_limits<size_t>::max();
  if (argc >= 4) {
    max_width = std::atoi(argv[3]);
  }

  size_t height;
  size_t width;
  std::cin >> height >> width;

  uchar* input_red;
  uchar* input_green;
  uchar* input_blue;
  read_rgb(height, width, max_height, max_width, &input_red, &input_green, &input_blue);

  if (height > max_height) height = max_height;
  if (width > max_width) width = max_width;

  std::cerr << "height: " << height << std::endl;
  std::cerr << "width: " << width << std::endl;
  std::cerr << "blur kernel size: " << BLUR_KERNEL_WIDTH << "x" << BLUR_KERNEL_WIDTH << std::endl;

  uchar* output_red_cpu;
  uchar* output_green_cpu;
  uchar* output_blue_cpu;
  {
    auto start_time = std::chrono::high_resolution_clock::now();

    blur_image_cpu(height, width, input_red, input_green, input_blue,
                   &output_red_cpu, &output_green_cpu, &output_blue_cpu);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cerr << "Time (CPU): " << total_time_ms / 1000.0 << " s" << std::endl;
  }

  uchar* output_red_gpu;
  uchar* output_green_gpu;
  uchar* output_blue_gpu;
  {
    auto start_time = std::chrono::high_resolution_clock::now();

    blur_image_gpu(height, width, input_red, input_green, input_blue,
                   &output_red_gpu, &output_green_gpu, &output_blue_gpu);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cerr << "Time (GPU): " << total_time_ms / 1000.0 << " s" << std::endl;
  }

  assert_equal(output_red_cpu, output_red_gpu, height, width);
  assert_equal(output_green_cpu, output_green_gpu, height, width);
  assert_equal(output_blue_cpu, output_blue_gpu, height, width);

  std::cout << height << ' ' << width << '\n';
  print_channel(output_red_gpu, height, width);
  print_channel(output_green_gpu, height, width);
  print_channel(output_blue_gpu, height, width);

  free(input_red);
  free(input_green);
  free(input_blue);

  free(output_red_cpu);
  free(output_green_cpu);
  free(output_blue_cpu);

  free(output_red_gpu);
  free(output_green_gpu);
  free(output_blue_gpu);
}
