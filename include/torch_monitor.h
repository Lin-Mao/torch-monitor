#ifndef TORCH_MONITOR_H
#define TORCH_MONITOR_H

#include <stdint.h>

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

/**
 * @brief Error code
 *
 */
typedef enum torch_monitor_status {
  TORCH_MONITOR_STATUS_SUCCESS = 0,
  TORCH_MONITOR_STATUS_INIT_HANDLE_FAIL = 1,
  TORCH_MONITOR_STATUS_INIT_DOMAIN_NULL = 2,
  TORCH_MONITOR_STATUS_INIT_MEMORY_NOT_REGISTER = 3,
  TORCH_MONITOR_STATUS_ENABLE_DOMAIN_OUT_RANGE = 4,
  TORCH_MONITOR_STATUS_SUBSCRIBE_EXIST = 5,
  TORCH_MONITOR_STATUS_SUBSCRIBE_SUBSCRIBER_NULL = 6,
  TORCH_MONITOR_STATUS_FINALIZE_NOT_INIT = 7,
  TORCH_MONITOR_STATUS_FINALIZE_MEMORY_FAIL = 8,
  TORCH_MONITOR_STATUS_COUNT = 9
} torch_monitor_status_t;

/**
 * @brief Monitor domains. One or more domains can be enabled in a single run.
 *
 */
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
  // Memory allocation
  TORCH_MONITOR_DOMAIN_MEMORY = 10,
  // Python state, support passive and active queries
  // In the active mode, states (e.g., call path) can be obtained using torch_monitor_python_state
  // In the passive mode, states are returned with torch_monitor_callback_data
  TORCH_MONITOR_DOMAIN_PYTOHN_STATE_PASSIVE = 11,
  TORCH_MONITOR_DOMAIN_PYTOHN_STATE_ACTIVE = 12,
  TORCH_MONITOR_DOMAIN_COUNT = 13
} torch_monitor_domain_t;

/**
 * @brief Enter or exit a torch function
 *
 */
typedef enum torch_monitor_callback_site {
  TORCH_MONITOR_CALLBACK_ENTER = 0,
  TORCH_MONITOR_CALLBACK_EXIT = 1,
  TORCH_MONITOR_CALLBACK_COUNT = 2
} torch_monitor_callback_site_t;

/**
 * @brief Information of each aten operation
 *
 */
typedef struct torch_monitor_op_data {
  uint64_t start_thread_id;
  uint64_t forward_thread_id;
  uint64_t sequence_number;
  const char *name;
} torch_monitor_op_data_t;

/**
 * @brief Information of each torch memory alloc operation
 *
 */
typedef struct torch_monitor_mem_data {
  void *ptr;
  int64_t alloc_size;
  int64_t total_allocated;
  int64_t total_reserved;
} torch_monitor_mem_data_t;

/**
 * @brief General callback information container
 *
 */
typedef struct torch_monitor_callback_data {
  torch_monitor_domain_t domain;
  // data can be casted using domain to
  // torch_monitor_callback_op_data_t
  // torch_monitor_callback_mem_data_t
  union {
    torch_monitor_op_data_t op_data;
    torch_monitor_mem_data_t mem_data;
  } data;
} torch_monitor_callback_data_t;

/**
 * @brief A callback that handles callback_data at each pytorch function enter/exit
 *
 * @param callback_site
 * @param callback_data
 *
 */
typedef void (*torch_monitor_callback_func_t)(torch_monitor_callback_site_t callback_site,
                                              torch_monitor_callback_data_t *callback_data);

/**
 * @brief A callback that handles callback_data at each pytorch function enter/exit
 *
 * @param func
 * @return torch_monitor_status_t
 *
 * @note not thread safe
 *
 */
EXTERNC torch_monitor_status_t torch_monitor_callback_subscribe(torch_monitor_callback_func_t func);

/**
 * @brief Enable a domain to be monitored
 *
 * @param domain
 * @return torch_monitor_status_t
 *
 * @note not thread safe
 *
 */
EXTERNC torch_monitor_status_t torch_monitor_enable_domain(torch_monitor_domain_t domain);

/**
 * @brief Return the python state of the query thread
 *
 * @return torch_monitor_status_t
 *
 * @note: not thread safe
 *
 */
EXTERNC torch_monitor_status_t torch_monitor_python_state();

/**
 * @brief Start monitoring pytorch functions in registered domains.
 * This function should be called only once at process initialization.
 *
 * @return torch_monitor_status_t
 *
 * @note: not thread safe
 *
 */
EXTERNC torch_monitor_status_t torch_monitor_init();

/**
 * @brief Init thread local states. This function should be called when each thread initializes.
 *
 * @return torch_monitor_status_t
 *
 * @note not thread safe
 *
 */
EXTERNC torch_monitor_status_t torch_monitor_thread_init();

/**
 * @brief Unregister all callbacks. This function should be called only once at process termination.
 *
 * @return torch_monitor_status_t
 *
 * @note not thread safe
 *
 */
EXTERNC torch_monitor_status_t torch_monitor_finalize();

/**
 * @brief Clear thread local states. This function should be called when each thread terminates.
 *
 * @return torch_monitor_status_t
 *
 * @note not thread safe
 *
 */
EXTERNC torch_monitor_status_t torch_monitor_thread_finalize();

#endif  // TORCH_MONITOR_H