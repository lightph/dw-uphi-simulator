#pragma once
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

struct DataRow {
  std::vector<double> data;
  DataRow(std::vector<double> dataa) : data(std::move(dataa)) {}
  DataRow() = default;
  void saveTo(std::ostream &os, char separator = ',') const {
    for (size_t i = 0; i < data.size(); ++i) {
      os << data[i];
      if (i < data.size() - 1)
        os << separator;
    }
    os << '\n';
  }
  friend std::ostream &operator<<(std::ostream &os, const DataRow &row) {
    row.saveTo(os, ',');
    return os;
  }
};

class DataSaver {
public:
  virtual ~DataSaver() = default;
  virtual void saveRow(const DataRow &row) = 0;
  virtual void saveMultipleRows(const std::vector<DataRow> &rows) = 0;
};

class CsvSaver : public DataSaver {
private:
  std::ofstream filestream;
  std::vector<std::string> headers;
  std::mutex mtx;

public:
  CsvSaver(const std::string &outputDir, const std::string &name,
           const std::vector<std::string> &headers)
      : headers(headers) {

    std::filesystem::create_directories(outputDir);

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream time_ss;
    time_ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");

    std::string base_filename = time_ss.str() + "__" + name;
    std::string filepath = outputDir + "/" + base_filename + ".csv";

    int counter = 1;
    while (std::filesystem::exists(filepath)) {
      filepath = outputDir + "/" + base_filename + "_" +
                 std::to_string(counter++) + ".csv";
    }

    filestream.open(filepath);

    if (filestream.is_open() && !this->headers.empty()) {
      writeHeaders();
    }
  }

  ~CsvSaver() override {
    if (filestream.is_open())
      filestream.close();
  }

  void writeHeaders() {
    for (size_t i = 0; i < headers.size(); ++i) {
      filestream << headers[i] << (i == headers.size() - 1 ? "" : ",");
    }
    filestream << "\n";
  }

  void saveRow(const DataRow &row) override {
    std::lock_guard<std::mutex> lock(mtx);
    if (filestream.is_open()) {
      filestream << row;
    }
  }

  void saveMultipleRows(const std::vector<DataRow> &rows) override {
    std::lock_guard<std::mutex> lock(mtx);
    if (!filestream.is_open())
      return;
    for (const auto &row : rows) {
      filestream << row;
    }
  }
};

class BinarySaver : public DataSaver {
private:
  std::ofstream filestream;
  std::mutex mtx;

public:
  BinarySaver(const std::string &outputDir, const std::string &name) {
    std::filesystem::create_directories(outputDir);

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream time_ss;
    time_ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");

    std::string base_filename = time_ss.str() + "__" + name;
    std::string filepath = outputDir + "/" + base_filename + ".bin";

    int counter = 1;
    while (std::filesystem::exists(filepath)) {
      filepath = outputDir + "/" + base_filename + "_" +
                 std::to_string(counter++) + ".bin";
    }

    filestream.open(filepath, std::ios::binary | std::ios::app);
  }

  ~BinarySaver() override {
    if (filestream.is_open())
      filestream.close();
  }

  void saveRow(const DataRow &row) override {
    std::lock_guard<std::mutex> lock(mtx);
    if (filestream.is_open() && !row.data.empty()) {
      filestream.write(reinterpret_cast<const char *>(row.data.data()),
                       row.data.size() * sizeof(double));
      filestream.flush();
    }
  }

  void saveMultipleRows(const std::vector<DataRow> &rows) override {
    std::lock_guard<std::mutex> lock(mtx);
    if (!filestream.is_open())
      return;

    for (const auto &row : rows) {
      if (!row.data.empty()) {
        filestream.write(reinterpret_cast<const char *>(row.data.data()),
                         row.data.size() * sizeof(double));
      }
    }
    filestream.flush();
  }
};
