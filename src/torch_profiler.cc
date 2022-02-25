#include "torch_profiler.h"

namespace torch_monitor {

void TorchProfiler::MemoryState::reportMemoryUsage(void* ptr, int64_t alloc_size,
                                                   int64_t total_allocated, int64_t total_reserved,
                                                   c10::Device device) {
  LOG_INFO("ptr: %p", ptr);
  LOG_INFO("alloc_size: %llu", alloc_size);
  LOG_INFO("total_allocated: %llu", total_allocated);
  LOG_INFO("total_reserved: %llu", total_reserved);

  torch_monitor_callback_data_t callback_data;
  callback_data.domain = TORCH_MONITOR_DOMAIN_MEMORY;
  callback_data.data.mem_data.ptr = ptr;
  callback_data.data.mem_data.alloc_size = alloc_size;
  callback_data.data.mem_data.total_allocated = total_allocated;
  callback_data.data.mem_data.total_reserved = total_reserved;

  auto& profiler = TorchProfiler::instance();
  profiler._callback(TORCH_MONITOR_CALLBACK_ENTER, &callback_data);
}

// A global state variable
// Since Aten record function does not capture anything,
// TorchProfilerState must be a static variable
struct TorchProfilerState {
  std::unordered_set<at::RecordScope> scopes;

  at::CallbackHandle handle = TORCH_MONITOR_HANDLE_NULL;

  torch_monitor_callback_func_t callback = nullptr;

  static void clear() {
    auto& profiler = instance();
    profiler.callback = nullptr;
    profiler.handle = TORCH_MONITOR_HANDLE_NULL;
    profiler.scopes.clear();
  }

  static TorchProfilerState& instance() {
    static TorchProfilerState state;
    return state;
  }

 private:
  TorchProfilerState() {}
};

bool TorchProfiler::init_callback_data(const at::RecordFunction& fn,
                                       torch_monitor_callback_data_t& callback_data) {
  LOG_INFO("thread_id: %llu", fn.threadId());
  LOG_INFO("forward_thread_id: %llu", fn.forwardThreadId());
  LOG_INFO("scope: %llu", fn.scope());
  LOG_INFO("async: %u", fn.isAsync());
  LOG_INFO("active: %u", fn.isActive());
  LOG_INFO("sequence_number: %llu", fn.seqNr());
  LOG_INFO("logical_thread_id: %llu", fn.currentThreadId());
  LOG_INFO("name: %s", fn.name().str());

  if (fn.seqNr() == TORCH_MONITOR_SEQUENCE_NUMBER_NULL) {
    return false;
  }

  auto domain = aten_scope_match(fn.scope());
  if (domain == TORCH_MONITOR_DOMAIN_COUNT) {
    return false;
  }

  callback_data.domain = domain;
  callback_data.data.op_data.start_thread_id = fn.threadId();
  callback_data.data.op_data.forward_thread_id = fn.threadId();
  callback_data.data.op_data.sequence_number = fn.seqNr();
  callback_data.data.op_data.name = fn.name().str();

  return true;
}

TorchProfiler& TorchProfiler::instance() {
  static TorchProfiler profiler;
  return profiler;
}

// True: if a domain is registered
// False: if a domain is not registered
bool TorchProfiler::has_domain(torch_monitor_domain_t domain) {
  at::RecordScope scope = torch_monitor_domain_match(domain);
  if (scope == at::RecordScope::NUM_SCOPES) {
    return false;
  }
  auto& instance = TorchProfilerState::instance();
  return instance.scopes.find(scope) != instance.scopes.end();
}

// True: register success
// False: register fail
bool TorchProfiler::register_domain(torch_monitor_domain_t domain) {
  at::RecordScope scope = torch_monitor_domain_match(domain);
  if (scope == at::RecordScope::NUM_SCOPES) {
    return false;
  }
  TorchProfilerState::instance().scopes.insert(scope);
  return true;
}

// True: register success
// False: register fail
bool TorchProfiler::register_callback(torch_monitor_callback_func_t callback) {
  auto& instance = TorchProfilerState::instance();
  if (instance.callback == nullptr) {
    instance.callback = callback;
    return true;
  }
  return false;
}

bool TorchProfiler::start_profiling() {
  auto handle = at::addGlobalCallback(
      at::RecordFunctionCallback(
          [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
            torch_monitor_callback_data_t callback_data = {};
            if (init_callback_data(fn, callback_data)) {
              TorchProfilerState::instance().callback(TORCH_MONITOR_CALLBACK_ENTER, &callback_data);
            }

            LOG_INFO("Enter function");
            return nullptr;
          },
          [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
            torch_monitor_callback_data_t callback_data = {};
            if (init_callback_data(fn, callback_data)) {
              TorchProfilerState::instance().callback(TORCH_MONITOR_CALLBACK_EXIT, &callback_data);
            }

            LOG_INFO("Exit function");
            return;
          })
          .needsInputs(false)   // TODO(Keren): monitor inputs if needed?
          .needsOutputs(false)  // TODO(Keren): monitor outputs if needed?
          .scopes(TorchProfilerState::instance().scopes));

  if (handle != TORCH_MONITOR_HANDLE_NULL) {
    TorchProfilerState::instance().handle = handle;
    return true;
  }

  return false;
}

bool TorchProfiler::stop_profiling() {
  TorchProfilerState::clear();
  return true;
}

bool TorchProfiler::start_memory_profiling() {
  if (has_domain(TORCH_MONITOR_DOMAIN_MEMORY)) {
    // Register the profiler to thread local state
    // c10 only has a ThreadLocalDebugInfo structure without a global structure
    auto mem_state_ptr = new_memory_state();
    c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, mem_state_ptr);
    return true;
  } else {
    return false;
  }
}

bool TorchProfiler::stop_memory_profiling() {
  if (has_domain(TORCH_MONITOR_DOMAIN_MEMORY)) {
    // XXX(Keren): torch monitor cannot be used together with kineto
    // Both register the profiler_state to ThreadLocalDebugInfo
    if (c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE) != nullptr) {
      return true;
    } else {
      return false;
    }
  }
  return true;
}

}  // namespace torch_monitor