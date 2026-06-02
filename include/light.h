#pragma once
#include <complex>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace light {

template <class T> inline T SQR(const T a) { return a * a; }

template <class T> inline T MAX(const T x1, const T x2) {
  return x1 > x2 ? x1 : x2;
}

template <class T> inline T MIN(const T x1, const T x2) {
  return x1 > x2 ? x2 : x1;
}

template <class T> inline T ABS(const T x) { return x >= 0 ? x : -x; }
template <class T> inline T SIGN(const T x1, const T x2) {
  return ABS(x1) * ((x2 >= 0) ? 1 : -1);
}

template <class T>
inline void writeToCsv(const std::vector<std::vector<T>> &data,
                       const std::string &filename,
                       const std::vector<std::string> &headers = {}) {
  if (data.empty()) {
    throw std::runtime_error("Attempted to write empty vector to csv");
  }
  int nData = data[0].size();
  for (auto &v : data) {
    if (v.size() != nData)
      throw std::runtime_error("Data size mismatch while writing to csv");
  }
  if (!headers.empty() && data.size() != headers.size()) {
    throw std::runtime_error("Data and header mismatch while writing to csv");
  }
  std::ofstream fileStream(filename);
  if (!fileStream) {
    throw std::runtime_error(std::string("Unable to open file ") + filename);
  }
  for (size_t h_ind = 0; h_ind < headers.size(); ++h_ind) {
    fileStream << headers[h_ind];
    if (h_ind < headers.size() - 1)
      fileStream << ',';
  }
  if (headers.size()) {
    fileStream << "\n";
  }
  for (size_t row = 0; row < nData; row++) {
    for (size_t column = 0; column < data.size(); ++column) {
      fileStream << data[column][row];
      if (column < data.size() - 1)
        fileStream << ',';
      else
        fileStream << '\n';
    }
  }
  return;
}

inline std::vector<double> getPSD(std::vector<std::complex<double>> fourier,
                                  int N, double fs) {
  std::vector<double> PSD(fourier.size());
  for (size_t i = 0; i < fourier.size(); ++i) {
    double re = fourier[i].real();
    double im = fourier[i].imag();
    double magsq = (re * re + im * im) / (static_cast<double>(N * N));
    if (i == 0 || (i == fourier.size() - 1 && !(N & 1))) {
      PSD[i] = magsq / fs;
      continue;
    }
    PSD[i] = 2. * magsq / fs;
  }
  return PSD;
}

inline std::vector<double>
getPowerSpectrum(std::vector<std::complex<double>> fourier, long long N) {
  std::vector<double> powerSpectrum(fourier.size());
  for (size_t i = 0; i < fourier.size(); ++i) {
    double re = fourier[i].real();
    double im = fourier[i].imag();
    double magsq = (re * re + im * im) / (static_cast<double>(N * N));
    if (magsq < 0)
      magsq = -1. * magsq;
    if (i == 0 || (i == fourier.size() - 1 && !(N & 1))) {
      powerSpectrum[i] = magsq;
      continue;
    }
    powerSpectrum[i] = 2. * magsq;
  }
  return powerSpectrum;
}
} // namespace light
