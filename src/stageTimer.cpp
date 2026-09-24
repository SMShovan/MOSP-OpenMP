/**
 * @file stageTimer.cpp
 * @brief Optional stage timers and counters.
 */

#include "stageTimer.h"

#include <cstdio>
#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace {

struct Registry {
  bool enabled = false;
  string prefix;
  vector<pair<string, double>> stages;
  map<string, double> counters;
};

Registry &registry() {
  static Registry instance;
  return instance;
}

} // namespace

void setInstrumentation(bool enabled) { registry().enabled = enabled; }

bool instrumentationEnabled() { return registry().enabled; }

void setStagePrefix(const string &prefix) { registry().prefix = prefix; }

const string &stagePrefix() { return registry().prefix; }

void recordStage(const string &name, double milliseconds) {
  if (registry().enabled) {
    registry().stages.emplace_back(registry().prefix + name, milliseconds);
  }
}

void recordCounter(const string &name, double value) {
  if (registry().enabled) {
    registry().counters[registry().prefix + name] += value;
  }
}

double totalStageTime(const string &suffix) {
  double total = 0;
  for (const auto &stage : registry().stages) {
    const string &name = stage.first;
    if (name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      total += stage.second;
    }
  }
  return total;
}

void clearInstrumentation() {
  registry().stages.clear();
  registry().counters.clear();
}

void printInstrumentation(ostream &out) {
  char line[256];
  for (const auto &stage : registry().stages) {
    snprintf(line, sizeof(line), "STAGE   %-52s %12.3f ms\n",
             stage.first.c_str(), stage.second);
    out << line;
  }
  for (const auto &counter : registry().counters) {
    snprintf(line, sizeof(line), "COUNTER %-52s %15.0f\n",
             counter.first.c_str(), counter.second);
    out << line;
  }
}

bool writeInstrumentationCsv(const string &path) {
  FILE *file = fopen(path.c_str(), "w");
  if (file == nullptr) {
    return false;
  }
  fprintf(file, "kind,name,value\n");
  for (const auto &stage : registry().stages) {
    fprintf(file, "stage,%s,%.6f\n", stage.first.c_str(), stage.second);
  }
  for (const auto &counter : registry().counters) {
    fprintf(file, "counter,%s,%.0f\n", counter.first.c_str(), counter.second);
  }
  return fclose(file) == 0;
}

ScopedStage::ScopedStage(const string &name)
    : name_(registry().prefix + name), active_(registry().enabled) {
  if (active_) {
    start_ = chrono::steady_clock::now();
  }
}

void ScopedStage::stop() {
  if (!active_) {
    return;
  }
  double ms =
      chrono::duration<double, milli>(chrono::steady_clock::now() - start_)
          .count();
  registry().stages.emplace_back(name_, ms);
  active_ = false;
}
