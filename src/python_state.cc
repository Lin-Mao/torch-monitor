#include "python_state.h"

#include <Python.h>
#include <pybind11/pybind11.h>

#include "utils.h"

namespace torch_monitor {

PythonStateMonitor& PythonStateMonitor::instance() {
  static PythonStateMonitor monitor;
  return monitor;
}

// Take from PyTorch::THPUtils_unpackStringView
std::string PythonStateMonitor::unpack_pyobject(PyObject* obj) {
  if (PyBytes_Check(obj)) {
    size_t size = PyBytes_GET_SIZE(obj);
    return std::string(PyBytes_AS_STRING(obj), size);
  }
  if (PyUnicode_Check(obj)) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Py_ssize_t size;
    const char* data = PyUnicode_AsUTF8AndSize(obj, &size);
    if (!data) {
      // If we get any runtime error, just return an empty string to continue running
      LOG_INFO("obj %p utf8 parsing error", obj);
      return "";
    }
    return std::string(data, (size_t)size);
  }
  LOG_INFO("obj %p not bytes or unicode", obj);
  return "";
}

std::vector<PythonState>& PythonStateMonitor::get_states(bool cached) {
  if (cached) {
    return _states;
  }

  // GIL lock is required
  pybind11::gil_scoped_acquire gil;

  PyFrameObject* frame = PyEval_GetFrame();
  _states.clear();

  while (nullptr != frame) {
    size_t lineno = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
    std::string file_name = unpack_pyobject(frame->f_code->co_filename);
    std::string function_name = unpack_pyobject(frame->f_code->co_name);
    _states.emplace_back(PythonState{file_name, function_name, lineno});
    frame = frame->f_back;
  }
  return _states;
}

}  // namespace torch_monitor