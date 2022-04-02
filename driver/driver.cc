#include <torch_monitor.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

#define TORCH_MONITOR_CALL(func, args)                              \
  do {                                                              \
    torch_monitor_status status = func args;                        \
    if (status != TORCH_MONITOR_STATUS_SUCCESS) {                   \
      std::cerr << "Torch monitor status: " << status << std::endl; \
      exit(1);                                                      \
    }                                                               \
  } while (0)

// If python call path is enabled
volatile static bool python_state_enable = false;
volatile static bool timestamp_enable = false;
volatile static bool driver_debug = false;
// Maximum number of call path frames
const static size_t MAX_NUM_STATES = 30;
// Call path buffer
thread_local static torch_monitor_python_state_t python_states[MAX_NUM_STATES];

static void python_state_report() {
  size_t num_states = 0;
  // Allow empty states
  torch_monitor_python_state_get(MAX_NUM_STATES, python_states, &num_states);
  for (size_t i = 0; i < num_states; ++i) {
    std::cout << "(" << i << ") "
              << "File: " << std::string(python_states[i].file_name) << std::endl;
    std::cout << "\tFunction: " << std::string(python_states[i].function_name) << std::endl;
    std::cout << "\tLine: " << python_states[i].lineno << std::endl;
  }
}

static void driver_callback(torch_monitor_callback_site_t callback_site,
                            torch_monitor_callback_data_t* callback_data) {
  while (driver_debug) {
  }

  if (callback_site == TORCH_MONITOR_CALLBACK_ENTER) {
    std::cout << "Domain: " << callback_data->domain << std::endl;
    if (callback_data->domain != TORCH_MONITOR_DOMAIN_MEMORY) {
      std::cout << "Current thread id: " << callback_data->current_thread_id << std::endl;
      std::cout << "Forward thread id: " << callback_data->data.op_data.forward_thread_id
                << std::endl;
      std::cout << "Sequence number: " << callback_data->data.op_data.sequence_number << std::endl;
      std::cout << "Name: " << std::string(callback_data->data.op_data.name) << std::endl;
      if (timestamp_enable) {
        std::cout << "Enter level: " << callback_data->data.op_data.nested_level << " at "
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count()
                  << std::endl;
      }
      if (python_state_enable) {
        python_state_report();
      }
    } else {
      std::cout << "Current thread id: " << callback_data->current_thread_id << std::endl;
      if (callback_data->data.mem_data.type == TORCH_MONITOR_MEM_DATA_ALLOC) {
        std::cout << "Allocate ptr: " << std::hex << callback_data->data.mem_data.ptr << std::dec
                  << std::endl;
      } else {
        std::cout << "Free ptr: 0x" << std::hex << callback_data->data.mem_data.ptr << std::dec
                  << std::endl;
      }
      std::cout << "Size: " << callback_data->data.mem_data.size << std::endl;
      std::cout << "Total size: " << callback_data->data.mem_data.total_allocated << std::endl;
      std::cout << "Total reserved: " << callback_data->data.mem_data.total_reserved << std::endl;
    }
  } else if (callback_site == TORCH_MONITOR_CALLBACK_EXIT) {
    if (callback_data->domain != TORCH_MONITOR_DOMAIN_MEMORY) {
      if (timestamp_enable) {
        std::cout << "Exit level: " << callback_data->data.op_data.nested_level << " at "
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count()
                  << std::endl;
      }
    }
  }
}

void driver_env_init() {
  if (const char* env = std::getenv("TORCH_MONITOR_PYTHON_STATE_ENABLE")) {
    if (std::atoi(env) == 1) {
      python_state_enable = true;
    }
  }

  if (const char* env = std::getenv("TORCH_MONITOR_DRIVER_DEBUG")) {
    if (std::atoi(env) == 1) {
      driver_debug = true;
    }
  }

  if (const char* env = std::getenv("TORCH_MONITOR_TIMESTAMP_ENABLE")) {
    if (std::atoi(env) == 1) {
      timestamp_enable = true;
    }
  }
}

int driver_register() {
  driver_env_init();

  TORCH_MONITOR_CALL(torch_monitor_enable_domain, (TORCH_MONITOR_DOMAIN_FUNCTION));
  TORCH_MONITOR_CALL(torch_monitor_enable_domain, (TORCH_MONITOR_DOMAIN_BACKWARD_FUNCTION));
  TORCH_MONITOR_CALL(torch_monitor_enable_domain, (TORCH_MONITOR_DOMAIN_MEMORY));
  TORCH_MONITOR_CALL(torch_monitor_callback_subscribe, (driver_callback));
  TORCH_MONITOR_CALL(torch_monitor_init, ());
  return 0;
}

int _ret = driver_register();
