#ifndef TORCH_MONITOR_UTILS_H
#define TORCH_MONITOR_UTILS_H

#include <torch/all.h>

#include <cstdio>

#include "torch_monitor.h"

#ifdef DEBUG
#define LOG_INFO(...)                    \
  do {                                   \
    fprintf(stdout, "TORCH_MONITOR-> "); \
    fprintf(stdout, __VA_ARGS__);        \
    fprintf(stdout, "\n");               \
  } while (0)
#else
#define LOG_INFO(...)
#endif

namespace torch_monitor {

const uint64_t TORCH_MONITOR_SEQUENCE_NUMBER_NULL = -1;
const uint64_t TORCH_MONITOR_HANDLE_NULL = 0;

torch_monitor_domain_t aten_scope_match(at::RecordScope scope);

at::RecordScope torch_monitor_domain_match(torch_monitor_domain_t domain);

}  // namespace torch_monitor

#endif  // TORCH_MONITOR_UTILS_H