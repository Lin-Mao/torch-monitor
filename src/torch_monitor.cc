#include "torch_monitor.h"

#include <torch/all.h>

#include <cstdio>
#include <unordered_set>

#include "utils.h"

namespace torch_monitor {

// True: init success
// False: init fail
static bool callback_data_init(const at::RecordFunction& fn,
                               torch_monitor_callback_data_t& callback_data) {
  LOG_INFO("thread_id: %llu", fn.threadId());
  LOG_INFO("forward_thread_id: %llu", fn.forwardThreadId());
  LOG_INFO("scope: %llu", fn.scope());
  LOG_INFO("async: %u", fn.isAsync());
  LOG_INFO("active: %u", fn.isActive());
  LOG_INFO("sequence_number: %llu", fn.seqNr());
  LOG_INFO("logical_thread_id: %llu", fn.currentThreadId());
  LOG_INFO("name: %s", fn.name().str());

  if (fn.seqNr() == TORCH_MONITOR_SEQUENCE_NUMBER_NULL) {
    return false;
  }

  auto domain = aten_scope_match(fn.scope());
  if (domain == TORCH_MONITOR_DOMAIN_COUNT) {
    return false;
  }

  callback_data.domain = domain;
  callback_data.start_thread_id = fn.threadId();
  callback_data.forward_thread_id = fn.threadId();
  callback_data.sequence_number = fn.seqNr();
  callback_data.name = fn.name().str();

  return true;
}

struct TorchMonitorState {
  TorchMonitorState() {}

  std::unordered_set<at::RecordScope> domains;

  at::CallbackHandle handle = TORCH_MONITOR_HANDLE_NULL;

  torch_monitor_callback_func_t callback = nullptr;

  // True: add success
  // False: add fail
  bool add_domain(torch_monitor_domain_t domain) {
    at::RecordScope scope = torch_monitor_domain_match(domain);
    if (scope == at::RecordScope::NUM_SCOPES) {
      return false;
    }
    domains.insert(scope);
    return true;
  }
};

static TorchMonitorState torch_monitor_state;

EXTERNC torch_monitor_status_t
torch_monitor_callback_subscribe(torch_monitor_callback_func_t func) {
  LOG_INFO("Enter torch_monitor_callback_subscribe");

  torch_monitor_status_t status;

  if (func) {
    if (torch_monitor_state.callback) {
      status = TORCH_MONITOR_STATUS_SUBSCRIBE_EXIST;
    } else {
      torch_monitor_state.callback = func;
      status = TORCH_MONITOR_STATUS_SUCCESS;
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

  if (domain < TORCH_MONITOR_DOMAIN_COUNT && torch_monitor_state.add_domain(domain)) {
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

  if (torch_monitor_state.domains.size() == 0) {
    status = TORCH_MONITOR_STATUS_INIT_DOMAIN_NULL;
  } else {
    torch_monitor_state.handle = at::addGlobalCallback(
        at::RecordFunctionCallback(
            [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
              torch_monitor_callback_data_t callback_data = {};
              if (callback_data_init(fn, callback_data)) {
                torch_monitor_state.callback(TORCH_MONITOR_CALLBACK_ENTER, &callback_data);
              }

              LOG_INFO("Enter function");
              return nullptr;
            },
            [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
              torch_monitor_callback_data_t callback_data = {};
              if (callback_data_init(fn, callback_data)) {
                torch_monitor_state.callback(TORCH_MONITOR_CALLBACK_EXIT, &callback_data);
              }

              LOG_INFO("Exit function");
              return;
            })
            .needsInputs(false)
            .needsOutputs(false)
            .scopes(torch_monitor_state.domains));

    // Handle is a uint64_t number starting from 1
    if (torch_monitor_state.handle != TORCH_MONITOR_SEQUENCE_NUMBER_NULL) {
      status = TORCH_MONITOR_STATUS_SUCCESS;
    } else {
      status = TORCH_MONITOR_STATUS_INIT_HANDLE_FAIL;
    }
  }

  LOG_INFO("Exit torch_monitor_init");
  return status;
}

EXTERNC torch_monitor_status_t torch_monitor_finalize() {
  LOG_INFO("Enter torch_monitor_finalize");

  torch_monitor_status_t status;
  if (torch_monitor_state.handle == TORCH_MONITOR_HANDLE_NULL) {
    status = TORCH_MONITOR_STATUS_FINALIZE_NOT_INIT;
  } else {
    status = TORCH_MONITOR_STATUS_SUCCESS;
  }

  LOG_INFO("Exit torch_monitor_finalize");
  return status;
}

}  // namespace torch_monitor