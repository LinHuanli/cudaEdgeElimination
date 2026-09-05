#pragma once

// 仅作者单核基准使用的同步消息适配器；不是通用 MPI 实现，GPU 主链不链接它。
struct MPI_Status {
  int MPI_SOURCE{}, MPI_TAG{}, count{};
};
constexpr int MPI_COMM_WORLD = 0, MPI_INT = 1, MPI_ANY_SOURCE = -1, MPI_MAX_PROCESSOR_NAME = 128;
int MPI_Send(const void*, int, int, int, int, int);
int MPI_Recv(void*, int, int, int, int, int, MPI_Status*);
int MPI_Iprobe(int, int, int, int*, MPI_Status*);
int MPI_Get_count(const MPI_Status*, int, int*);
