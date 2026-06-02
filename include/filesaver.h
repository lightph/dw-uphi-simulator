#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

struct DataRow
{
  std::vector<double> data;
  DataRow(std::vector<double> dataa) : data(std::move(dataa)) {}
  DataRow() = default;
};

class MasterLogger
{
public:
  static void logRun(const std::string &outputDir, const std::string &filepath,
                     const std::string &expName,
                     const std::map<std::string, std::string> &params)
  {

    std::filesystem::create_directories(outputDir);
    std::string logPath = outputDir + "/master_log.jsonl";

    auto now =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream time_ss;
    time_ss << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S");

    // Manually construct the JSON object payload first
    std::stringstream ss;
    ss << "{";
    ss << "\"Timestamp\":\"" << time_ss.str() << "\",";
    ss << "\"FilePath\":\"" << filepath << "\",";
    ss << "\"Experiment\":\"" << expName << "\",";

    ss << "\"Parameters\":{";
    for (auto it = params.begin(); it != params.end(); ++it)
    {
      ss << "\"" << it->first << "\":\"" << it->second << "\"";
      if (std::next(it) != params.end())
      {
        ss << ",";
      }
    }
    ss << "},";
    ss << "\"Comments\":\"\"";
    ss << "}\n";

    std::string payload = ss.str();

    // 1. Open the file with OS-level file descriptor
    int fd = open(logPath.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd == -1)
    {
      // Silently fail or handle error if the log file cannot be opened
      return;
    }

    // 2. Configure a POSIX write lock
    struct flock fl;
    std::memset(&fl, 0, sizeof(fl));
    fl.l_type = F_WRLCK; // Write lock
    fl.l_whence = SEEK_SET;
    fl.l_start = 0; // Lock the whole file
    fl.l_len = 0;

    // 3. Acquire the lock (F_SETLKW will block/wait until the lock is
    // available)
    if (fcntl(fd, F_SETLKW, &fl) != -1)
    {
      // 4. Write the entire payload safely
      write(fd, payload.c_str(), payload.size());

      // 5. Release the lock
      fl.l_type = F_UNLCK;
      fcntl(fd, F_SETLK, &fl);
    }

    // 6. Close the file descriptor
    close(fd);
  }
};

class DataSaver
{
protected:
  std::ofstream filestream;
  std::string filepath;

  std::vector<DataRow> currentBuffer;
  size_t bufferLimit = 5000;

  std::queue<std::vector<DataRow>> writeQueue;
  std::mutex mtx;
  std::condition_variable cv;
  std::thread writerThread;
  std::atomic<bool> stopFlag{false};

  virtual void writeBatch(const std::vector<DataRow> &batch) = 0;

  void writerLoop()
  {
    while (true)
    {
      std::vector<DataRow> batch;
      {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]
                { return stopFlag || !writeQueue.empty(); });

        if (stopFlag && writeQueue.empty())
          return;

        batch = std::move(writeQueue.front());
        writeQueue.pop();
      }
      writeBatch(batch);
    }
  }

  void flushBufferUnsafe()
  {
    if (!currentBuffer.empty())
    {
      writeQueue.push(std::move(currentBuffer));
      currentBuffer.clear();
      currentBuffer.reserve(bufferLimit);
      cv.notify_one();
    }
  }

  // Moved cleanup logic here so derived classes can safely halt the thread
  void stopWriter()
  {
    {
      std::lock_guard<std::mutex> lock(mtx);
      if (stopFlag)
        return; // Prevent double execution
      flushBufferUnsafe();
      stopFlag = true;
    }
    cv.notify_one();

    if (writerThread.joinable())
    {
      writerThread.join();
    }
    if (filestream.is_open())
    {
      filestream.close();
    }
  }

public:
  DataSaver(const std::string &outputDir, const std::string &expName,
            const std::map<std::string, std::string> &params,
            const std::string &ext)
  {

    std::filesystem::create_directories(outputDir);

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream time_ss;
    time_ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");

    std::string base_filename = time_ss.str() + "_" + expName;

    const char *job_id = std::getenv("JOB_ID");
    const char *task_id = std::getenv("SGE_TASK_ID");

    bool has_job = (job_id && std::strcmp(job_id, "undefined") != 0);
    bool has_task = (task_id && std::strcmp(task_id, "undefined") != 0);

    if (has_job || has_task)
    {
      if (has_job)
      {
        base_filename += "_job" + std::string(job_id);
      }
      if (has_task)
      {
        base_filename += "_task" + std::string(task_id);
      }
    }
    else
    {
      // Fallback for local runs: use the OS Process ID
      base_filename += "_pid" + std::to_string(getpid());
    }

    filepath = outputDir + "/" + base_filename + ext;
    int counter = 1;

    // This loop acts as a final safety net, but collisions are now practically
    // impossible
    while (std::filesystem::exists(filepath))
    {
      filepath = outputDir + "/" + base_filename + "_" +
                 std::to_string(counter++) + ext;
    }

    MasterLogger::logRun(outputDir, filepath, expName, params);

    currentBuffer.reserve(bufferLimit);
    writerThread = std::thread(&DataSaver::writerLoop, this);
  }

  virtual ~DataSaver() { stopWriter(); }

  void saveRow(DataRow row)
  {
    std::lock_guard<std::mutex> lock(mtx);
    currentBuffer.push_back(std::move(row));
    if (currentBuffer.size() >= bufferLimit)
    {
      flushBufferUnsafe();
    }
  }

  void saveMultipleRows(const std::vector<DataRow> &rows)
  {
    std::lock_guard<std::mutex> lock(mtx);
    for (const auto &row : rows)
    {
      currentBuffer.push_back(row);
      if (currentBuffer.size() >= bufferLimit)
      {
        flushBufferUnsafe();
      }
    }
  }

  void flush()
  {
    std::lock_guard<std::mutex> lock(mtx);
    flushBufferUnsafe();
  }
};

class CsvSaver : public DataSaver
{
private:
  std::vector<std::string> headers;

protected:
  void writeBatch(const std::vector<DataRow> &batch) override
  {
    if (!filestream.is_open())
      return;
    for (const auto &row : batch)
    {
      for (size_t i = 0; i < row.data.size(); ++i)
      {
        filestream << row.data[i];
        if (i < row.data.size() - 1)
          filestream << ",";
      }
      filestream << "\n";
    }
    filestream.flush();
  }

public:
  CsvSaver(const std::string &outputDir, const std::string &expName,
           const std::map<std::string, std::string> &params,
           const std::vector<std::string> &headers)
      : DataSaver(outputDir, expName, params, ".csv"), headers(headers)
  {

    filestream.open(filepath);
    if (filestream.is_open() && !this->headers.empty())
    {
      for (size_t i = 0; i < this->headers.size(); ++i)
      {
        filestream << this->headers[i]
                   << (i == this->headers.size() - 1 ? "" : ",");
      }
      filestream << "\n";
    }
  }

  // Safely stop the thread while CsvSaver is still fully intact
  ~CsvSaver() override { stopWriter(); }
};

class BinarySaver : public DataSaver
{
protected:
  void writeBatch(const std::vector<DataRow> &batch) override
  {
    if (!filestream.is_open())
      return;
    for (const auto &row : batch)
    {
      if (!row.data.empty())
      {
        filestream.write(reinterpret_cast<const char *>(row.data.data()),
                         row.data.size() * sizeof(double));
      }
    }
    filestream.flush();
  }

public:
  BinarySaver(const std::string &outputDir, const std::string &expName,
              const std::map<std::string, std::string> &params)
      : DataSaver(outputDir, expName, params, ".bin")
  {

    filestream.open(filepath, std::ios::binary | std::ios::app);
  }

  // Safely stop the thread while BinarySaver is still fully intact
  ~BinarySaver() override { stopWriter(); }
};
