#ifndef TORCH_MONITOR_H
#define TORCH_MONITOR_H

#include <stdint.h>

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

typedef enum torch_monitor_status {
  TORCH_MONITOR_STATUS_SUCCESS = 0,
  TORCH_MONITOR_STATUS_INIT_HANDLE_FAIL = 1,
  TORCH_MONITOR_STATUS_INIT_DOMAIN_NULL = 2,
  TORCH_MONITOR_STATUS_ENABLE_DOMAIN_OUT_RANGE = 3,
  TORCH_MONITOR_STATUS_SUBSCRIBE_EXIST = 4,
  TORCH_MONITOR_STATUS_SUBSCRIBE_SUBSCRIBER_NULL = 5,
  TORCH_MONITOR_STATUS_FINALIZE_NOT_INIT = 6,
  TORCH_MONITOR_STATUS_COUNT = 7
} torch_monitor_status_t;

typedef enum torch_monitor_domain {
  // c10/ATen ops, autograd nodes
  TORCH_MONITOR_DOMAIN_FUNCTION = 0,
  // Functions/nodes called from the autograd
  TORCH_MONITOR_DOMAIN_BACKWARD_FUNCTION = 1,
  // TorchScript functions, methods
  TORCH_MONITOR_DOMAIN_TORCHSCRIPT_FUNCTION = 2,
  // Kernel Function dtype Tag
  TORCH_MONITOR_DOMAIN_KERNEL_FUNCTION_DTYPE = 3,
  // Torchbind custom class,
  TORCH_MONITOR_DOMAIN_CUSTOM_CLASS = 4,
  // Generic Build Feature
  TORCH_MONITOR_DOMAIN_BUILD_FEATURE = 5,
  // Kernel Function dtype Tag
  TORCH_MONITOR_DOMAIN_LITE_INTERPRETER = 6,
  // User defined scope (e.g. with record_function())
  TORCH_MONITOR_DOMAIN_USER_SCOPE = 7,
  // Scopes for static runtime, a specialized TorchScript interpreter
  TORCH_MONITOR_DOMAIN_STATIC_RUNTIME_OP = 8,
  TORCH_MONITOR_DOMAIN_STATIC_RUNTIME_MODEL = 9,
  TORCH_MONITOR_DOMAIN_COUNT = 10,
} torch_monitor_domain_t;

typedef enum torch_monitor_callback_site {
  TORCH_MONITOR_CALLBACK_ENTER = 0,
  TORCH_MONITOR_CALLBACK_EXIT = 1,
  TORCH_MONITOR_CALLBACK_COUNT = 2
} torch_monitor_callback_site_t;

typedef struct torch_monitor_callback_data {
  torch_monitor_domain_t domain;
  uint64_t start_thread_id;
  uint64_t forward_thread_id;
  uint64_t sequence_number;
  const char *name;
} torch_monitor_callback_data_t;

const uint64_t TORCH_MONITOR_SEQUENCE_NUMBER_NULL = -1;
const uint64_t TORCH_MONITOR_HANDLE_NULL = 0;

//
// A callback that handles callback_data at each pytorch function enter/exit
//
typedef void (*torch_monitor_callback_func_t)(torch_monitor_callback_site_t callback_site,
                                              torch_monitor_callback_data_t *callback_data);

//
// Add a callback function. It is supposed to be called only once
//
// MT: not thread safe.
//
EXTERNC torch_monitor_status_t torch_monitor_callback_subscribe(torch_monitor_callback_func_t func);

//
// Enable pytorch domains to be monitored
//
// MT: not thread safe
//
EXTERNC torch_monitor_status_t torch_monitor_enable_domain(torch_monitor_domain_t domain);

//
// Start monitoring pytorch functions in registered domains
//
// MT: not thread safe
//
EXTERNC torch_monitor_status_t torch_monitor_init();

//
// Clears thread states and unregistered all callbacks
//
// MT: not thread safe
//
EXTERNC torch_monitor_status_t torch_monitor_finalize();

#endif  // TORCH_MONITOR_H