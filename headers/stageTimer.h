#ifndef STAGE_TIMER_H
#define STAGE_TIMER_H

#include <chrono>
#include <iosfwd>
#include <string>

/**
 * @brief Optional stage instrumentation (wall-clock stage timers and
 *        counters).
 *
 * @details
 * Disabled by default. When disabled, ScopedStage and recordCounter() do
 * nothing, so the instrumented code paths run exactly as without them.
 * When enabled (e.g. `bin/mosp --timing stages.csv`), each ScopedStage
 * records its wall time. (Same interface as the CUDA version, which also
 * emits NVTX ranges and can synchronize the device at stage boundaries.)
 */
void setInstrumentation(bool enabled);
bool instrumentationEnabled();

/**
 * @brief Prefix prepended to every stage and counter name recorded from now
 *        on (e.g. "obj1/"); lets a driver label the stages of a call.
 */
void setStagePrefix(const std::string &prefix);
const std::string &stagePrefix();

/** @brief Append a stage record (milliseconds). */
void recordStage(const std::string &name, double milliseconds);

/** @brief Add @p value to a named counter. */
void recordCounter(const std::string &name, double value);

/** @brief Sum of all stage records whose name ends with @p suffix. */
double totalStageTime(const std::string &suffix);

/** @brief Drop all records and counters. */
void clearInstrumentation();

/** @brief Print "STAGE name ms" and "COUNTER name value" lines. */
void printInstrumentation(std::ostream &out);

/** @brief Write the records as CSV: kind,name,value. */
bool writeInstrumentationCsv(const std::string &path);

/** @brief RAII stage timer; see file comment. */
class ScopedStage {
public:
  explicit ScopedStage(const std::string &name);
  ~ScopedStage() { stop(); }
  ScopedStage(const ScopedStage &) = delete;
  ScopedStage &operator=(const ScopedStage &) = delete;
  void stop();

private:
  std::string name_;
  bool active_;
  std::chrono::steady_clock::time_point start_;
};

#endif // STAGE_TIMER_H
