#pragma once

#include "cufftHandler.cuh"
#include "types.cuh"

#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <vector>

namespace pseudospectral_detail
{
  __global__ void fused_fourier_step_kernel(
      cuda_math::COMPLEX *zhat,
      const cuda_math::COMPLEX *nhat,
      const cuda_math::COMPLEX *denom,
      const cuda_math::COMPLEX *factor,
      cuda_math::REAL dt,
      size_t n)
  {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
      zhat[i] = denom[i] * (factor[i] * zhat[i] + dt * nhat[i]);
    }
  }

  __global__ void scale_physical_kernel(
      cuda_math::COMPLEX *z,
      cuda_math::REAL invN,
      size_t n)
  {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
      z[i] = cuda_math::COMPLEX(z[i].real() * invN, z[i].imag() * invN);
    }
  }

  __global__ void advance_time_kernel(cuda_math::REAL *d_t, cuda_math::REAL dt)
  {
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
      d_t[0] += dt;
    }
  }

  template <typename N>
  __global__ void evaluate_nonlinear_kernel(
      cuda_math::COMPLEX *n_ptr,
      const cuda_math::COMPLEX *z_ptr,
      const cuda_math::REAL *d_t_ptr,
      N nonlin,
      size_t n_modes)
  {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_modes)
    {
      n_ptr[i] = nonlin(z_ptr[i], d_t_ptr[0]);
    }
  }
}

template <typename NonlinearPart, typename InitialCondition, typename LinearPart>
class gpu_pde_pseudospectral_stepper
{
private:
  using REAL = cuda_math::REAL;
  using COMPLEX = cuda_math::COMPLEX;
  using REAL_VECTOR = cuda_math::REAL_DEVICE_VECTOR;
  using COMPLEX_VECTOR = cuda_math::COMPLEX_DEVICE_VECTOR;

  static constexpr REAL PI = static_cast<REAL>(3.14159265358979323846);

public:
  gpu_pde_pseudospectral_stepper(size_t n_modes, REAL total_length, REAL dtt,
                                 NonlinearPart nonlinear_part,
                                 InitialCondition initial_condition,
                                 LinearPart linear_part);

  ~gpu_pde_pseudospectral_stepper()
  {
    if (graphCreated)
    {
      cudaGraphExecDestroy(stepExec);
      cudaGraphDestroy(stepGraph);
    }
    cudaStreamDestroy(exec_stream);
  }

  gpu_pde_pseudospectral_stepper(const gpu_pde_pseudospectral_stepper &) = delete;
  gpu_pde_pseudospectral_stepper &operator=(const gpu_pde_pseudospectral_stepper &) = delete;

  void step();
  void step_on_stream(cudaStream_t stream);

  const COMPLEX_VECTOR &get_z() const { return z_buf; }
  const COMPLEX_VECTOR &get_zhat() const { return zhat_buf; }

  void update_precomputations();

  void set_dt(REAL dtt)
  {
    dt = dtt;
    update_precomputations();
  }

  void set_nonlinear_part(const NonlinearPart &new_nonlin)
  {
    nonlinear_part = new_nonlin;
  }

  void set_linear_part(const LinearPart &new_lin)
  {
    linear_part = new_lin;
    update_precomputations();
  }

  REAL get_dt() const { return dt; }
  REAL get_time() const { return t; }

  void set_time(REAL new_t)
  {
    t = new_t;
    if (d_t_vec.size() == 1)
    {
      REAL *d_t_ptr = thrust::raw_pointer_cast(d_t_vec.data());
      cudaMemcpyAsync(d_t_ptr, &t, sizeof(REAL), cudaMemcpyHostToDevice, exec_stream);
    }
  }

  void initialize();
  void create_step_graph(size_t steps);
  void run_block(size_t steps);

  void reset(InitialCondition new_ic)
  {
    initial_condition = new_ic;
    t = static_cast<REAL>(0.0);
    initialize();
  }

private:
  NonlinearPart nonlinear_part;
  InitialCondition initial_condition;
  LinearPart linear_part;

  size_t n_modes;
  REAL total_length;
  REAL dt;
  REAL t = static_cast<REAL>(0.0);
  REAL_VECTOR d_t_vec;

  REAL_VECTOR k_vals;
  COMPLEX_VECTOR denom_precomp;
  COMPLEX_VECTOR factor_precomp;

  COMPLEX_VECTOR z_buf;
  COMPLEX_VECTOR N_buf;
  COMPLEX_VECTOR Nhat_buf;
  COMPLEX_VECTOR zhat_buf;

  CufftHandlerC2COut cufft_handler;

  cudaStream_t exec_stream;
  cudaGraph_t stepGraph;
  cudaGraphExec_t stepExec;
  bool graphCreated = false;
  size_t captured_steps = 0;
  REAL invN;
};

template <typename N, typename I, typename L>
gpu_pde_pseudospectral_stepper<N, I, L>::gpu_pde_pseudospectral_stepper(
    size_t modes, REAL length, REAL dtt, N nonlin, I init_cond, L lin)
    : nonlinear_part(nonlin), initial_condition(init_cond), linear_part(lin),
      n_modes(modes), total_length(length), dt(dtt),
      z_buf(modes), N_buf(modes), Nhat_buf(modes), zhat_buf(modes)
{
  cudaStreamCreate(&exec_stream);

  k_vals.resize(n_modes);
  denom_precomp.resize(n_modes);
  factor_precomp.resize(n_modes);
  cufft_handler.prepare(n_modes);
  initialize();
}

template <typename N, typename I, typename L>
void gpu_pde_pseudospectral_stepper<N, I, L>::initialize()
{
  REAL local_L = total_length;
  REAL local_dt = dt;

  d_t_vec.resize(1);
  d_t_vec[0] = t;

  size_t local_n = n_modes;
  L local_lin = linear_part;
  I local_init = initial_condition;

  invN = 1.0 / static_cast<cuda_math::REAL>(local_n);

  REAL *k_ptr = thrust::raw_pointer_cast(k_vals.data());
  COMPLEX *d_ptr = thrust::raw_pointer_cast(denom_precomp.data());
  COMPLEX *f_ptr = thrust::raw_pointer_cast(factor_precomp.data());
  COMPLEX *z_ptr = thrust::raw_pointer_cast(z_buf.data());

  thrust::for_each(thrust::device, thrust::counting_iterator<size_t>(0), thrust::counting_iterator<size_t>(local_n),
                   [=] __device__(size_t i)
                   {
                     REAL delta_k = (2.0 * PI) / local_L;
                     REAL k = (i <= local_n / 2) ? i * delta_k : (static_cast<REAL>(i) - static_cast<REAL>(local_n)) * delta_k;

                     k_ptr[i] = k;
                     COMPLEX lp = local_lin(k);
                     d_ptr[i] = COMPLEX(1.0, 0.0) / (COMPLEX(1.0, 0.0) - 0.5 * local_dt * lp);
                     f_ptr[i] = COMPLEX(1.0, 0.0) + 0.5 * local_dt * lp;
                   });

  thrust::for_each(thrust::device, thrust::counting_iterator<size_t>(0), thrust::counting_iterator<size_t>(local_n),
                   [=] __device__(size_t i)
                   {
                     REAL dx = local_L / local_n;
                     z_ptr[i] = local_init(i * dx);
                   });

  cufft_handler.do_fft(z_buf, zhat_buf);
  cudaDeviceSynchronize();
}

template <typename N, typename I, typename L>
void gpu_pde_pseudospectral_stepper<N, I, L>::step()
{
  cufft_handler.setStream(exec_stream);
  step_on_stream(exec_stream);
  cudaStreamSynchronize(exec_stream);
  t += dt;
}

template <typename N, typename I, typename L>
void gpu_pde_pseudospectral_stepper<N, I, L>::update_precomputations()
{
  REAL local_dt = dt;
  L local_lin = linear_part;
  REAL *k_ptr = thrust::raw_pointer_cast(k_vals.data());
  COMPLEX *d_ptr = thrust::raw_pointer_cast(denom_precomp.data());
  COMPLEX *f_ptr = thrust::raw_pointer_cast(factor_precomp.data());

  thrust::for_each(thrust::device, thrust::counting_iterator<size_t>(0), thrust::counting_iterator<size_t>(n_modes),
                   [=] __device__(size_t i)
                   {
                     COMPLEX lp = local_lin(k_ptr[i]);
                     COMPLEX one(static_cast<REAL>(1.0), static_cast<REAL>(0.0));
                     REAL half = static_cast<REAL>(0.5);

                     d_ptr[i] = one / (one - half * local_dt * lp);
                     f_ptr[i] = one + half * local_dt * lp;
                   });
}

template <typename N, typename I, typename L>
void gpu_pde_pseudospectral_stepper<N, I, L>::step_on_stream(cudaStream_t stream)
{
  COMPLEX *z_ptr = thrust::raw_pointer_cast(z_buf.data());
  COMPLEX *n_ptr = thrust::raw_pointer_cast(N_buf.data());
  N local_nonlin = nonlinear_part;
  REAL *d_t_ptr = thrust::raw_pointer_cast(d_t_vec.data());

  int threads = 256;
  int blocks = (n_modes + threads - 1) / threads;

  pseudospectral_detail::evaluate_nonlinear_kernel<<<blocks, threads, 0, stream>>>(
      n_ptr, z_ptr, d_t_ptr, local_nonlin, n_modes);

  cufft_handler.do_fft(N_buf, Nhat_buf);

  COMPLEX *zhat_ptr = thrust::raw_pointer_cast(zhat_buf.data());
  COMPLEX *nhat_ptr = thrust::raw_pointer_cast(Nhat_buf.data());
  COMPLEX *d_ptr = thrust::raw_pointer_cast(denom_precomp.data());
  COMPLEX *f_ptr = thrust::raw_pointer_cast(factor_precomp.data());

  pseudospectral_detail::fused_fourier_step_kernel<<<blocks, threads, 0, stream>>>(
      zhat_ptr, nhat_ptr, d_ptr, f_ptr, dt, n_modes);

  cufft_handler.do_ifft(zhat_buf, z_buf);
  pseudospectral_detail::scale_physical_kernel<<<blocks, threads, 0, stream>>>(
      z_ptr, invN, n_modes);

  pseudospectral_detail::advance_time_kernel<<<1, 1, 0, stream>>>(d_t_ptr, dt);
}

template <typename N, typename I, typename L>
void gpu_pde_pseudospectral_stepper<N, I, L>::create_step_graph(size_t steps)
{
  if (steps == 0)
    return;

  if (graphCreated)
  {
    cudaGraphExecDestroy(stepExec);
    cudaGraphDestroy(stepGraph);
    graphCreated = false;
  }

  const size_t MAX_GRAPH_STEPS = 1000;
  captured_steps = std::min(steps, MAX_GRAPH_STEPS);

  cufft_handler.setStream(exec_stream);

  COMPLEX_VECTOR z_backup = z_buf;

  this->step_on_stream(exec_stream);
  cudaStreamSynchronize(exec_stream);

  z_buf = z_backup;

  REAL *d_t_ptr = thrust::raw_pointer_cast(d_t_vec.data());
  cudaMemcpyAsync(d_t_ptr, &t, sizeof(REAL), cudaMemcpyHostToDevice, exec_stream);
  cudaStreamSynchronize(exec_stream);

  cudaStreamBeginCapture(exec_stream, cudaStreamCaptureModeGlobal);

  for (size_t i = 0; i < captured_steps; ++i)
  {
    this->step_on_stream(exec_stream);
  }

  cudaError_t cap_err = cudaStreamEndCapture(exec_stream, &stepGraph);
  if (cap_err != cudaSuccess || stepGraph == nullptr)
  {
    std::cerr << "FATAL: CUDA Graph Capture failed: " << cudaGetErrorString(cap_err) << std::endl;
    exit(EXIT_FAILURE);
  }

  cudaError_t inst_err = cudaGraphInstantiate(&stepExec, stepGraph, NULL, NULL, 0);
  if (inst_err != cudaSuccess)
  {
    std::cerr << "FATAL: CUDA Graph Instantiate failed: " << cudaGetErrorString(inst_err) << std::endl;
    exit(EXIT_FAILURE);
  }

  graphCreated = true;
}

template <typename N, typename I, typename L>
void gpu_pde_pseudospectral_stepper<N, I, L>::run_block(size_t steps)
{
  if (steps == 0)
    return;

  if (!graphCreated || captured_steps == 0)
  {
    for (size_t i = 0; i < steps; ++i)
    {
      this->step_on_stream(exec_stream);
    }
  }
  else
  {
    size_t num_chunks = steps / captured_steps;
    size_t remainder = steps % captured_steps;

    for (size_t i = 0; i < num_chunks; ++i)
    {
      cudaGraphLaunch(stepExec, exec_stream);
    }

    for (size_t i = 0; i < remainder; ++i)
    {
      this->step_on_stream(exec_stream);
    }
  }

  cudaStreamSynchronize(exec_stream);

  t += dt * steps;
}
