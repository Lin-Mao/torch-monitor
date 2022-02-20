#include <torch_monitor.h>

int driver_register() {
  torch_monitor_init();
  return 0;
}

int _ret = driver_register();
