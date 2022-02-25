#include "torch_monitor.h"

#include <torch/all.h>

#include "torch_profiler.h"
#include "utils.h"

namespace torch_monitor {

EXTERNC torch_monitor_status_t
torch_monitor_callback_subscribe(torch_monitor_callback_func_t func) {
  LOG_INFO("Enter torch_monitor_callback_subscribe");

  torch_monitor_status_t status;

  auto &profiler = TorchProfiler::instance();

  if (func) {
    if (profiler.register_callback(func)) {
      status = TORCH_MONITOR_STATUS_SUCCESS;
    } else {
      status = TORCH_MONITOR_STATUS_SUBSCRIBE_EXIST;
    }
  } else {
    status = TORCH_MONITOR_STATUS_SUBSCRIBE_SUBSCRIBER_NULL;
  }

  LOG_INFO("Exit torch_monitor_callback_subscribe");
  return status;
}

EXTERNC torch_monitor_status_t torch_monitor_enable_domain(torch_monitor_domain_t domain) {
  LOG_INFO("Enter torch_monitor_enable_domain");

  torch_monitor_status_t status;

  auto &profiler = TorchProfiler::instance();

  if (domain < TORCH_MONITOR_DOMAIN_COUNT && profiler.register_domain(domain)) {
    status = TORCH_MONITOR_STATUS_SUCCESS;
  } else {
    status = TORCH_MONITOR_STATUS_ENABLE_DOMAIN_OUT_RANGE;
  }

  LOG_INFO("Exit torch_monitor_enable_domain");
  return status;
}

EXTERNC torch_monitor_status_t torch_monitor_init() {
  LOG_INFO("Enter torch_monitor_init");

  torch_monitor_status_t status;

  auto &profiler = TorchProfiler::instance();

  if (profiler.start_profiling()) {
    status = TORCH_MONITOR_STATUS_SUCCESS;
  } else {
    status = TORCH_MONITOR_STATUS_INIT_HANDLE_FAIL;
  }

  torch_monitor_thread_init();

  LOG_INFO("Exit torch_monitor_init");
  return status;
}

EXTERNC torch_monitor_status_t torch_monitor_thread_init() {
  LOG_INFO("Enter torch_monitor_thread_init");

  torch_monitor_status_t status;

  auto &profiler = TorchProfiler::instance();

  if (profiler.start_memory_profiling()) {
    status = TORCH_MONITOR_STATUS_SUCCESS;
  } else {
    status = TORCH_MONITOR_STATUS_INIT_MEMORY_NOT_REGISTER;
  }

  LOG_INFO("Exit torch_monitor_thread_init");
  return status;
}

EXTERNC torch_monitor_status_t torch_monitor_finalize() {
  LOG_INFO("Enter torch_monitor_finalize");

  torch_monitor_status_t status;

  auto &profiler = TorchProfiler::instance();

  if (profiler.stop_profiling()) {
    status = TORCH_MONITOR_STATUS_SUCCESS;
  } else {
    status = TORCH_MONITOR_STATUS_FINALIZE_NOT_INIT;
  }

  LOG_INFO("Exit torch_monitor_finalize");
  return status;
}

EXTERNC torch_monitor_status_t torch_monitor_thread_finalize() {
  LOG_INFO("Enter torch_monitor_thread_finalize");

  torch_monitor_status_t status;

  auto &profiler = TorchProfiler::instance();

  if (profiler.stop_memory_profiling()) {
    status = TORCH_MONITOR_STATUS_SUCCESS;
  } else {
    status = TORCH_MONITOR_STATUS_FINALIZE_MEMORY_FAIL;
  }

  LOG_INFO("Exit torch_monitor_thread_finalize");

  return status;
}

}  // namespace torch_monitor