#include <torch_monitor.h>

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

static void driver_callback(torch_monitor_callback_site_t callback_site,
                            torch_monitor_callback_data_t *callback_data) {
  if (callback_site == TORCH_MONITOR_CALLBACK_ENTER) {
    std::cout << "Domain: " << callback_data->domain << std::endl;
    if (callback_data->domain != TORCH_MONITOR_DOMAIN_MEMORY) {
      std::cout << "Current thread id: " << callback_data->current_thread_id << std::endl;
      std::cout << "Forward thread id: " << callback_data->data.op_data.forward_thread_id
                << std::endl;
      std::cout << "Sequence number: " << callback_data->data.op_data.sequence_number << std::endl;
      std::cout << "Name: " << std::string(callback_data->data.op_data.name) << std::endl;
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
  }
}

int driver_register() {
  TORCH_MONITOR_CALL(torch_monitor_enable_domain, (TORCH_MONITOR_DOMAIN_FUNCTION));
  TORCH_MONITOR_CALL(torch_monitor_enable_domain, (TORCH_MONITOR_DOMAIN_BACKWARD_FUNCTION));
  TORCH_MONITOR_CALL(torch_monitor_enable_domain, (TORCH_MONITOR_DOMAIN_MEMORY));
  TORCH_MONITOR_CALL(torch_monitor_callback_subscribe, (driver_callback));
  TORCH_MONITOR_CALL(torch_monitor_init, ());
  return 0;
}

int _ret = driver_register();
