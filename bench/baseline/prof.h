// Stage timers for timing the original code (baseline-2026-09); not part
// of the original sources. build.sh copies this header next to them and
// stage-timers.patch adds the timers to parallelSOSPUpdate() and
// parallelCombinedGraph(). Usage:
//   StageTimer st("obj0/");  st.begin("read_csr"); ... st.begin("next"); ...
//   profCounter("obj0/iterations") += 1;
// profDump() prints one "STAGE <name> <ms>" line per timed stage (wall
// clock, steady_clock) and one "COUNTER <name> <value>" line per counter.
#pragma once
#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

struct ProfRec {
  std::string name;
  double ms;
};
inline std::vector<ProfRec> &profRecs() {
  static std::vector<ProfRec> r;
  return r;
}
inline std::map<std::string, double> &profCounters() {
  static std::map<std::string, double> c;
  return c;
}
inline double &profCounter(const std::string &k) { return profCounters()[k]; }
/// Prefix of the stage names of the current call (e.g. "obj1/").
inline std::string &profPrefix() {
  static std::string p;
  return p;
}

class StageTimer {
public:
  explicit StageTimer(const std::string &prefix) : prefix_(prefix) {}
  ~StageTimer() { end(); }
  void begin(const std::string &stage) {
    end();
    cur_ = prefix_ + stage;
    open_ = true;
    t0_ = std::chrono::steady_clock::now();
  }
  void end() {
    if (!open_) return;
    auto t1 = std::chrono::steady_clock::now();
    profRecs().push_back(
        {cur_, std::chrono::duration<double, std::milli>(t1 - t0_).count()});
    open_ = false;
  }

private:
  std::string prefix_, cur_;
  bool open_ = false;
  std::chrono::steady_clock::time_point t0_;
};

inline void profDump(FILE *f = stdout) {
  std::fprintf(f, "\n==== STAGE TIMES (wall ms) ====\n");
  for (auto &r : profRecs()) {
    std::fprintf(f, "STAGE %-55s %12.3f\n", r.name.c_str(), r.ms);
  }
  std::fprintf(f, "==== COUNTERS ====\n");
  for (auto &kv : profCounters())
    std::fprintf(f, "COUNTER %-53s %14.0f\n", kv.first.c_str(), kv.second);
}
