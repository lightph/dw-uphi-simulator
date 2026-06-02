#include "filesaver.h"
#include "slonczewski_wall.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

// Mutexes for thread safety
std::mutex io_mutex;
std::mutex fftw_mutex; // Protects FFTW plan creation and destruction

// Worker function to be executed by each thread
void run_h0_sweep(double h0, double Omega, double alpha, double dt,
                  double min_resolution, int transient_steps,
                  int measurement_steps, double tolerance, int base_seed) {

  int local_seed = base_seed;

  // Updated DataSaver initialization
  std::string outDir = "output/heatmap3";
  std::string expName = "parameter_sweep_threaded";

  std::ostringstream h0_stream;
  h0_stream << std::fixed << std::setprecision(1) << h0;

  std::map<std::string, std::string> params = {
      {"h0", h0_stream.str()},
      {"Omega", std::to_string(Omega)},
      {"alpha", std::to_string(alpha)},
      {"dt", std::to_string(dt)},
      {"base_seed", std::to_string(base_seed)}};

  BinarySaver local_saver(outDir, expName, params);

  for (double k = 0.05432; k <= 1.47; k += 0.05432) {
    double system_size = 2 * M_PI / k;
    for (double h = 0.0; h <= 7.0; h += 0.1) {

      RandomInitialCondition ic(0.01, local_seed++);

      // Instantiate the experiment class to track metrics during the run
      SweepMetricsExperiment<RandomInitialCondition> experiment;

      {
        std::unique_ptr<Slonczewski_Wall<RandomInitialCondition>> wall_ptr;

        {
          std::lock_guard<std::mutex> lock(fftw_mutex);
          wall_ptr = std::make_unique<Slonczewski_Wall<RandomInitialCondition>>(
              alpha, h, h0, Omega, dt, system_size, min_resolution, ic);
        }

        // Pass the experiment object by reference to run_simulation
        run_simulation(*wall_ptr, experiment, dt, transient_steps,
                       measurement_steps, tolerance);

        {
          std::lock_guard<std::mutex> lock(fftw_mutex);
          wall_ptr.reset();
        }
      }

      // Extract the calculated results directly from the experiment object
      std::vector<double> row_data = {system_size,
                                      alpha,
                                      h,
                                      h0,
                                      Omega,
                                      experiment.final_time_avg_rugosity,
                                      experiment.final_speed_mean_u,
                                      experiment.final_speed_mean_phi};

      local_saver.saveRow(DataRow(row_data));

      {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cout << "Thread h0=" << h0 << " completed: "
                  << "sys_size=" << system_size << ", h=" << h << "\n";
      }
    }
  }
}

int main() {
  const double alpha = 0.27;
  const double dt = 0.1;
  const double min_resolution = 0.1;
  const int transient_window_steps = 50000;
  const int measurement_window_steps = 50000;
  const double tolerance = 1e-5;
  const double Omega = 0.5;

  std::vector<double> h0_vals = {0.0, 0.5, 1.0, 1.5, 2.0, 20.};
  std::vector<std::thread> threads;

  auto sweep_start_time = std::chrono::steady_clock::now();
  std::cout << "Starting parallel parameter sweep...\n";

  int seed_multiplier = 0;
  for (double h0 : h0_vals) {
    int thread_base_seed = seed_multiplier * 10000;

    threads.emplace_back(run_h0_sweep, h0, Omega, alpha, dt, min_resolution,
                         transient_window_steps, measurement_window_steps,
                         tolerance, thread_base_seed);
    seed_multiplier++;
  }

  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  auto current_time = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds =
      current_time - sweep_start_time;

  int hours = static_cast<int>(elapsed_seconds.count()) / 3600;
  int minutes = (static_cast<int>(elapsed_seconds.count()) % 3600) / 60;
  int seconds = static_cast<int>(elapsed_seconds.count()) % 60;

  std::cout << "Parameter sweep complete. Total time: " << std::setfill('0')
            << std::setw(2) << hours << ":" << std::setfill('0') << std::setw(2)
            << minutes << ":" << std::setfill('0') << std::setw(2) << seconds
            << "\n";

  return 0;
}
