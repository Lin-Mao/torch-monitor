#include "torch_monitor.h"

#include <torch/all.h>

#include <cstdio>

#define DEBUG

#include "utils.h"

namespace torch_monitor {

EXTERNC torch_monitor_status_t torch_monitor_init() {
  LOG_INFO("Enter torch_monitor_init\n");

  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(
          [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
            LOG_INFO("Enter function\n");
            LOG_INFO("Capture thread_id: %u\n", fn.threadId());
            return nullptr;
          },
          [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
            LOG_INFO("Exit function\n");
            return;
          })
          .needsInputs(false));

  LOG_INFO("Exit torch_monitor_init\n");

  // Handle is a uint64_t number starting from 1
  if (handle != 0) {
    return TORCH_MONITOR_STATUS_SUCCESS;
  } else {
    return TORCH_MONITOR_STATUS_INIT_FAIL;
  }
}

}  // namespace torch_monitor