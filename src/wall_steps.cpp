#include "filesaver.h"
#include "slonczewski_experiments.h"
#include "slonczewski_wall.h"
#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>

struct StepWidthResult {
  double Omega;
  double h0;
  double system_size;
  int n;
  double h_min;
  double h_max;
  double width;
};

// Helper to evaluate a single h value using the stateful wall object
inline double
evaluate_adiabatic_speed(Slonczewski_Wall<RandomInitialCondition> &wall,
                         double h_test, double dt, int trans_steps,
                         int meas_steps, double sim_tol) {

  wall.set_h(h_test);
  SweepMetricsExperiment<RandomInitialCondition> exp;

  // Since the wall retains its spatial buffer, the transient
  // phase will be very short if h_test is close to the previous h
  run_simulation(wall, exp, dt, trans_steps, meas_steps, sim_tol);

  return exp.final_speed_mean_phi;
}

// Finds the width of the n-th Shapiro step for a given wall configuration
inline StepWidthResult
measure_shapiro_step_width(Slonczewski_Wall<RandomInitialCondition> &wall,
                           double Omega, double h0, double system_size, int n,
                           double alpha, double dh, double speed_tol, double dt,
                           int trans_steps, int meas_steps, double sim_tol) {

  double target_speed = n * Omega / 2.0;

  // The step usually occurs around h ≈ alpha * velocity
  double h_guess = std::sqrt(target_speed * target_speed + 1.0);

  // Start slightly below the expected plateau to catch the rising edge
  double current_h = h_guess - 1.;

  double h_min = -1.0;
  double h_max = -1.0;
  bool on_step = false;

  // Coarse alignment run to let the system settle at the starting h
  evaluate_adiabatic_speed(wall, current_h, dt, trans_steps * 5, meas_steps,
                           sim_tol);

  // targeted forward sweep
  for (int step = 0; step < 100; ++step) {
    double v = evaluate_adiabatic_speed(wall, current_h, dt, trans_steps,
                                        meas_steps, sim_tol);

    bool matches_plateau = std::abs(v - target_speed) < speed_tol;

    if (matches_plateau && !on_step) {
      on_step = true;
      h_min = current_h;
    } else if (!matches_plateau && on_step) {
      h_max = current_h - dh; // We just fell off the plateau
      break;
    }

    current_h += dh;
  }

  double width = (on_step && h_max > h_min) ? (h_max - h_min) : 0.0;
  return {Omega, h0, system_size, n, h_min, h_max, width};
}

void run_parameter_sweep() {
  std::vector<double> Omegas = {0.5, 1.0, 1.5, 2.0};

  std::vector<double> h0s;
  for (int i = 0; i <= 50; ++i) {
    h0s.push_back(5.0 - i * 0.1);
  }

  std::vector<double> system_sizes;
  for (int i = 1; i <= 150; ++i) {
    double k = i * 0.01;
    system_sizes.push_back(2.0 * M_PI / k);
  }

  std::vector<int> n_steps = {1, 2};

  // Fixed simulation parameters
  double alpha = 0.27;
  double dt = 0.1;
  double min_res = 0.1;
  double dh = 0.01;
  double speed_tol = 1e-2;
  double sim_tol = 1e-2;
  int trans_steps = 50000;
  int meas_steps = 50000;
  double eps_noise = 1e-2;

  // 1. Package parameters for the MasterLogger
  std::map<std::string, std::string> params = {
      {"alpha", std::to_string(alpha)},
      {"dt", std::to_string(dt)},
      {"dh", std::to_string(dh)},
      {"speed_tol", std::to_string(speed_tol)},
      {"sim_tol", std::to_string(sim_tol)},
      {"trans_steps", std::to_string(trans_steps)},
      {"meas_steps", std::to_string(meas_steps)},
      {"eps_noise", std::to_string(eps_noise)}};

  // 2. Define headers and initialize the saver
  std::vector<std::string> headers = {"Omega", "h0",    "L",    "n",
                                      "h_min", "h_max", "width"};
  std::string outputDir = "./sweep_data";
  CsvSaver saver(outputDir, "shapiro_widths", params, headers);

// Use collapse to parallelize across the nested loops
#pragma omp parallel for collapse(3) schedule(dynamic)
  for (size_t i = 0; i < Omegas.size(); ++i) {
    for (size_t j = 0; j < h0s.size(); ++j) {
      for (size_t k_idx = 0; k_idx < system_sizes.size(); ++k_idx) {

        double Omega = Omegas[i];
        double h0 = h0s[j];
        double sys_size = system_sizes[k_idx];

        int seed_offset = omp_get_thread_num();
        RandomInitialCondition ic(eps_noise, seed_offset);

        std::unique_ptr<Slonczewski_Wall<RandomInitialCondition>> wall;

#pragma omp critical(fftw_plan_creation)
        {
          wall = std::make_unique<Slonczewski_Wall<RandomInitialCondition>>(
              alpha, 0.0, h0, Omega, dt, sys_size, min_res, ic);
        }

        for (int n : n_steps) {
          StepWidthResult res = measure_shapiro_step_width(
              *wall, Omega, h0, sys_size, n, alpha, dh, speed_tol, dt,
              trans_steps, meas_steps, sim_tol);

          // 3. Save the row (CsvSaver has its own internal mutex, so this is
          // thread-safe)
          saver.saveRow(DataRow({res.Omega, res.h0, res.system_size,
                                 static_cast<double>(res.n), res.h_min,
                                 res.h_max, res.width}));

// Keeping cout in a critical section just to prevent jumbled terminal output
#pragma omp critical(terminal_out)
          {
            std::cout << "Omega: " << res.Omega << ", h0: " << res.h0
                      << ", L: " << res.system_size << ", n: " << res.n
                      << " -> Width: " << res.width << "\n";
          }
        }
      }
    }
  }
}

int main() {
  run_parameter_sweep();
  return 0;
}
