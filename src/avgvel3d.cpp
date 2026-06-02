#include "ODPendulum.h"
#include "PhiSweeper.h"
#include "filesaver.h"
#include <cmath>
#include <map>
#include <string>

void runAndSavePhiSweep3D(const std::string &outDir, double hmin, double hmax,
                          size_t nh, double B_min, double B_max, size_t Bn,
                          double h0min, double h0max, size_t nh0) {

  int waitNPeriods = 100;
  int averageNPeriods = 100;

  double initial_A = 1.0;
  double initial_B = -0.5;
  double initial_C = 25.0;

  double dt = 0.01;
  double t0 = 0.0;
  double x0 = 0.01;
  size_t N_steps =
      static_cast<size_t>((waitNPeriods + averageNPeriods) * 2 * M_PI / dt + 5);

  std::map<std::string, std::string> sweep_params = {
      {"Type", "ODPendulum_PhiSweep3D"},

      // Numerical & Solver Initial State
      {"Wait_Periods", std::to_string(waitNPeriods)},
      {"Average_Periods", std::to_string(averageNPeriods)},
      {"TimeStep_dt", std::to_string(dt)},
      {"Initial_A", std::to_string(initial_A)},
      {"Initial_B", std::to_string(initial_B)},
      {"Initial_C", std::to_string(initial_C)},
      {"Initial_t0", std::to_string(t0)},
      {"Initial_x0", std::to_string(x0)},
      {"Total_Steps_N", std::to_string(N_steps)},

      // Sweep Boundaries: h
      {"h_Min", std::to_string(hmin)},
      {"h_Max", std::to_string(hmax)},
      {"h_Points", std::to_string(nh)},

      // Sweep Boundaries: Omega (passed as B_min/B_max in the sweeper
      // arguments)
      {"Omega_Min", std::to_string(B_min)},
      {"Omega_Max", std::to_string(B_max)},
      {"Omega_Points", std::to_string(Bn)},

      // Sweep Boundaries: h0
      {"h0_Min", std::to_string(h0min)},
      {"h0_Max", std::to_string(h0max)},
      {"h0_Points", std::to_string(nh0)},

      {"Total_Runs", std::to_string(nh * Bn * nh0)}};

  DrivenODPendulum solver(initial_A, initial_B, initial_C, dt, t0, x0, N_steps);
  PendulumAnalyser analyser(waitNPeriods, averageNPeriods);
  PhiSweeper sweeper;

  BinarySaver saver(outDir, "phi_sweep_3D", sweep_params);

  sweeper.getAvgVelRange3D(solver, analyser, hmin, hmax, nh, B_min, B_max, Bn,
                           h0min, h0max, nh0, &saver, nullptr);
}

int main(int argc, char **argv) {
  std::string outDir = "output/phi3d";

  const double h_min = 0.0, h_max = 10.0;
  const size_t h_points = 1000;

  const double Omega_min = 0.1, Omega_max = 10.0;
  const size_t Omega_points = 100;

  const double h0_min = 0.0, h0_max = 10.00;
  const size_t h0_points = 100;

  runAndSavePhiSweep3D(outDir, h_min, h_max, h_points, Omega_min, Omega_max,
                       Omega_points, h0_min, h0_max, h0_points);

  return 0;
}
