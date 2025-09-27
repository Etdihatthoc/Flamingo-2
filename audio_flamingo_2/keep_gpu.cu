#include <cuda_runtime.h>
#include <unistd.h>

__global__ void noop() {}

int main() {
  // Khởi tạo CUDA context mà không cần cấp nhiều bộ nhớ
  cudaFree(0);
  // Gọi một kernel trống để đảm bảo context active
  noop<<<1,1>>>();
  cudaDeviceSynchronize();

  // Giữ tiến trình sống mãi tới khi bị kill/scancel
  while (true) {
    sleep(3600); // ngủ 1 giờ rồi lặp
  }
  return 0;
}
