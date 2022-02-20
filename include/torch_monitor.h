#ifndef TORCH_MONITOR_H
#define TORCH_MONITOR_H

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

typedef enum torch_monitor_status {
  TORCH_MONITOR_STATUS_SUCCESS = 0,
  TORCH_MONITOR_STATUS_INIT_FAIL = 1,
  TORCH_MONITOR_STATUS_COUNT = 2
} torch_monitor_status_t;

EXTERNC torch_monitor_status_t torch_monitor_init();

#endif  // TORCH_MONITOR_H