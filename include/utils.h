#ifndef TORCH_MONITOR_UTILS_H
#define TORCH_MONITOR_UTILS_H

#include <cstdio>

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

#endif