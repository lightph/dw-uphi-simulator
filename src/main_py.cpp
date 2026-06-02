#include "ODPendulum.h"
#include <cmath>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <stdexcept>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

py::list fourierBatch(
    py::array_t<double, py::array::c_style | py::array::forcecast> dcBias,
    py::array_t<double, py::array::c_style | py::array::forcecast>
        restoringAmplitude,
    py::array_t<double, py::array::c_style | py::array::forcecast>
        forcingAmplitude,
    double h, int waitNPeriods, int averageNPeriods, double t0, double tf = 0.,
    double x0 = 1.0) {
  if (dcBias.size() == 0)
    throw std::runtime_error("Input arrays are empty");

  py::ssize_t numExperiments = dcBias.size();
  auto dcBiasProxy = dcBias.unchecked<1>();
  auto restoringProxy = restoringAmplitude.unchecked<1>();
  auto forcingProxy = forcingAmplitude.unchecked<1>();

  DrivenODPendulum dodp(dcBiasProxy(0), restoringProxy(0), forcingProxy(0), h,
                        t0, tf, x0, waitNPeriods, averageNPeriods);

  py::list results;

  for (py::ssize_t i = 0; i < numExperiments; ++i) {
    dodp.setA(dcBiasProxy(i));
    dodp.setB(restoringProxy(i));
    dodp.setC(forcingProxy(i));

    dodp.solve();

    std::vector<double> currentPower = dodp.computePowerSpectrum(true);
    std::vector<double> currentFreqs = dodp.computeFftfreq();

    results.append(
        py::array_t<double>(currentFreqs.size(), currentFreqs.data()));

    results.append(
        py::array_t<double>(currentPower.size(), currentPower.data()));
  }

  return results;
}

PYBIND11_MODULE(pyODPend, m) {
  m.def("fourier_batch", &fourierBatch, "Compute a batch of Fourier transforms",
        py::arg("dc_bias"), py::arg("restoring_amplitude"),
        py::arg("forcing_amplitude"), py::arg("timestep"),
        py::arg("wait_n_periods"), py::arg("average_n_periods"), py::arg("t0"),
        py::arg("tf") = 0.0, py::arg("x0") = 1.0);
}
