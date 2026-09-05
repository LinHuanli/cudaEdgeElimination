#include "cc_manager.h"
#include "mpi_functions.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <list>
#include <stdexcept>
#include <vector>

void execute_s1_neighbors_job(Instance&);
void execute_s2_el_edges_job(Instance&);
void execute_pt_holder_job(Instance&, std::vector<CC_Manager>&);

namespace {
struct Message {
  int source, destination, tag;
  std::vector<int> values;
};
std::list<Message> queue;
int serial_rank = 0;
Instance* worker = nullptr;
std::vector<CC_Manager> connections;
auto Find(const int source, const int tag) {
  return std::find_if(queue.begin(), queue.end(), [&](const Message& message) {
    return message.destination == serial_rank && message.tag == tag &&
           (source == MPI_ANY_SOURCE || message.source == source);
  });
}
void Status(const Message& message, MPI_Status* status) {
  *status = {message.source, message.tag, static_cast<int>(message.values.size())};
}
} // namespace

int MPI_Send(const void* data, const int count, const int type, const int destination,
             const int tag, const int communicator) {
  if (count < 0 || type != MPI_INT || communicator != MPI_COMM_WORLD || destination < 0 ||
      destination > 1)
    throw std::runtime_error("串行 MPI 消息非法");
  Message message{serial_rank, destination, tag, {}};
  if (count != 0)
    message.values.assign(static_cast<const int*>(data), static_cast<const int*>(data) + count);
  queue.push_back(std::move(message));
  if (serial_rank == 0 && destination == 1 && tag != TAG_TERMINATE) {
    if (worker == nullptr)
      throw std::runtime_error("作者 worker 未初始化");
    // 同一个物理线程执行原版 worker 入口；回复先排队，由原版 master 集成。
    serial_rank = 1;
    if (tag == TAG_S1_EL_NEIGHBORS)
      execute_s1_neighbors_job(*worker);
    else if (tag == TAG_S2_EL_EDGE)
      execute_s2_el_edges_job(*worker);
    else if (tag == TAG_PT_HOLDER)
      execute_pt_holder_job(*worker, connections);
    else
      throw std::runtime_error("串行基准出现未支持的 master 消息");
    serial_rank = 0;
  }
  return 0;
}
int MPI_Iprobe(const int source, const int tag, const int communicator, int* flag,
               MPI_Status* status) {
  if (communicator != MPI_COMM_WORLD)
    throw std::runtime_error("未知 communicator");
  const auto found = Find(source, tag);
  *flag = found != queue.end();
  if (*flag)
    Status(*found, status);
  return 0;
}
int MPI_Get_count(const MPI_Status* status, const int type, int* count) {
  if (type != MPI_INT)
    throw std::runtime_error("未知 MPI 数据类型");
  *count = status->count;
  return 0;
}
int MPI_Recv(void* data, const int count, const int type, const int source, const int tag,
             const int communicator, MPI_Status* status) {
  const auto found = Find(source, tag);
  if (communicator != MPI_COMM_WORLD || type != MPI_INT || found == queue.end() ||
      static_cast<int>(found->values.size()) != count)
    throw std::runtime_error("作者消息队列不匹配");
  Status(*found, status);
  std::copy(found->values.begin(), found->values.end(), static_cast<int*>(data));
  queue.erase(found);
  return 0;
}

int main(int argc, char** argv) try {
  if (argc != 4)
    throw std::invalid_argument("hs2014-serial INSTANCE.tsp OPTIONS OUTPUT_DIRECTORY");
  const auto root = std::filesystem::canonical(CUDAEE_SOURCE_DIR);
  const auto output = std::filesystem::weakly_canonical(std::filesystem::absolute(argv[3]));
  const auto relative = output.lexically_relative(root);
  if (relative.empty() || relative == "." || *relative.begin() == "..")
    throw std::invalid_argument("输出必须位于项目内");
  std::filesystem::create_directories(output);
  char name[] = "single-thread-serial-dispatch";
  Instance master(argv[3], 2, 0, name);
  Instance local_worker(argv[3], 2, 1, name);
  for (Instance* instance : {&master, &local_worker}) {
    if (!instance->read_tsp_file(argv[1]) || !instance->read_options_file(argv[2], argv[3]))
      throw std::runtime_error("作者实例/选项读取失败");
  }
  if (!master.options().s1())
    throw std::invalid_argument("完整基准必须从 S1 完整图开始");
  worker = &local_worker;
  // 关闭 worker 的逐候选调试日志；数学判定与 master 阶段统计不变。
  local_worker.log().setstate(std::ios::failbit);
  master.output_information();
  master.eliminate_edges();
  if (!queue.empty())
    throw std::runtime_error("作者基准结束时存在未消费消息");
  std::cout << "status=OK author=HS2014 source=2015-02-27 execution=single-thread-complete\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
