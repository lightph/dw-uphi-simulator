#include <iostream>
#include "types.cuh"
#include <fstream>
#include <cmath>
#include <iomanip>
#include "cufftHandler.cuh"

void save_to_csv(const std::string &filename,
                 const std::vector<double> &original,
                 const thrust::host_vector<cuda_math::REAL> &r2c_res,
                 const thrust::host_vector<cuda_math::COMPLEX> &c2c_res)
{
    std::ofstream file(filename);
    file << "index,original,r2c_reconstructed,c2c_reconstructed_real,c2c_reconstructed_imag\n";
    for (size_t i = 0; i < original.size(); ++i)
    {
        file << i << ","
             << original[i] << ","
             << r2c_res[i] << ","
             << c2c_res[i].real() << ","
             << c2c_res[i].imag() << "\n";
    }
    file.close();
}

int main()
{
    const size_t N = 1024;
    const double fs = 1000.0;
    const double f1 = 50.0;
    const double f2 = 120.0;

    // 1. Prepare Original Signal on Host
    std::vector<double> h_input(N);
    thrust::host_vector<cuda_math::COMPLEX> h_complex_input(N);
    for (size_t i = 0; i < N; ++i)
    {
        double t = static_cast<double>(i) / fs;
        double val = std::sin(2.0 * M_PI * f1 * t) + 0.5 * std::sin(2.0 * M_PI * f2 * t);
        h_input[i] = val;
        h_complex_input[i] = cuda_math::COMPLEX(static_cast<cuda_math::REAL>(val), 0.0f);
    }

    // 2. Move to Device
    thrust::device_vector<cuda_math::REAL> d_real_input(N);
    for (size_t i = 0; i < N; ++i)
        d_real_input[i] = static_cast<cuda_math::REAL>(h_input[i]);
    thrust::device_vector<cuda_math::COMPLEX> d_complex_input = h_complex_input;

    // 3. Test R2C and C2R (Real to Complex and back)
    CufftHandlerR2C r2c_handler;
    CufftHandlerC2R c2r_handler;
    thrust::device_vector<cuda_math::COMPLEX> d_freq_domain;
    thrust::device_vector<cuda_math::REAL> d_real_reconstructed(N);

    r2c_handler.do_fft(d_real_input, d_freq_domain);
    c2r_handler.do_ifft(d_freq_domain, d_real_reconstructed);

    // 4. Test C2C (Complex to Complex and back)
    CufftHandlerC2COut c2c_handler;
    thrust::device_vector<cuda_math::COMPLEX> d_c2c_freq(N);
    thrust::device_vector<cuda_math::COMPLEX> d_c2c_reconstructed(N);

    c2c_handler.do_fft(d_complex_input, d_c2c_freq);
    c2c_handler.do_ifft(d_c2c_freq, d_c2c_reconstructed);

    // 5. Gather results and save
    thrust::host_vector<cuda_math::REAL> h_r2c_reconstructed = d_real_reconstructed;
    thrust::host_vector<cuda_math::COMPLEX> h_c2c_reconstructed = d_c2c_reconstructed;

    save_to_csv("fft_results.csv", h_input, h_r2c_reconstructed, h_c2c_reconstructed);

    std::cout << "Test completed. Results saved to fft_results.csv" << std::endl;

    return 0;
}