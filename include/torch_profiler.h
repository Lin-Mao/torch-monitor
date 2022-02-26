#ifndef TORCH_MONITOR_TORCH_PROFILER_H
#define TORCH_MONITOR_TORCH_PROFILER_H

#include <torch/all.h>

#include <unordered_set>

#include "torch_monitor.h"
#include "utils.h"

namespace torch_monitor {

class TorchProfiler {
 public:
  void enable_memory_profiling() { _is_memory_profiling_enabled = true; }

  void disable_memory_profiling() { _is_memory_profiling_enabled = false; }

  bool is_memory_profiling_enabled() { return _is_memory_profiling_enabled; }

  // true: register success
  // false: register fail
  bool register_domain(torch_monitor_domain_t domain);

  // true: domain registered
  // false: domain not registered
  bool has_domain(torch_monitor_domain_t domain);

  // true: register success
  // false: register fail
  bool register_callback(torch_monitor_callback_func_t callback);

  // true: start profiling
  // false: cannot start profiling
  bool start_profiling();

  // true: stop profiling
  // false: cannot stop profiling
  bool stop_profiling();

  // true: start profiling
  // false: cannot start profiling
  bool start_memory_profiling();

  // true: stop profiling
  // false: cannot stop profiling
  bool stop_memory_profiling();

  // Get the singleton instance
  static TorchProfiler& instance();

 private:
  TorchProfiler() {}

  class MemoryState : public c10::MemoryReportingInfoBase {
   public:
    MemoryState() {}

    virtual bool memoryProfilingEnabled() const override { return true; }

    // Memory allocatation callback
    virtual void reportMemoryUsage(void* ptr, int64_t alloc_size, int64_t total_allocated,
                                   int64_t total_reserved, c10::Device device) override;
  };

  // Generate a memory state for every thread
  std::shared_ptr<MemoryState> new_memory_state() {
    if (_is_memory_profiling_enabled) {
      return std::make_shared<MemoryState>();
    } else {
      return nullptr;
    }
  }

  // true: init success
  // false: init fail
  static bool init_callback_data(const at::RecordFunction& fn,
                                 torch_monitor_callback_data_t& callback_data);

 private:
  bool _is_memory_profiling_enabled = false;
};

}  // namespace torch_monitor

#endif  // TORCH_MONITOR_TORCH_PROFILER_H