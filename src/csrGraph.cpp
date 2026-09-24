/**
 * @file csrGraph.cpp
 * @brief In-memory multi-objective CSR graph: fast text I/O, a binary
 *        cache, change batches, batch application and transposition.
 *
 * The original pipeline re-read the text CSR through readCSR() (an
 * istream-based parser that builds a vector<vector<Edge>> with one heap
 * allocation per edge) K+1 times per run. These helpers parse each file
 * once into flat arrays so the graph can be shared by all objectives.
 */

#include "csrGraph.h"

#include <sys/stat.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

namespace {

// ============================================================================
// Raw file helpers
// ============================================================================

bool readWholeFile(const string &path, string &buffer) {
  FILE *file = fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return false;
  }
  struct stat info;
  if (fstat(fileno(file), &info) != 0) {
    fclose(file);
    return false;
  }
  buffer.resize(static_cast<size_t>(info.st_size));
  size_t got = buffer.empty() ? 0 : fread(&buffer[0], 1, buffer.size(), file);
  fclose(file);
  return got == buffer.size();
}

void createParentDirectory(const string &path) {
  filesystem::path parent = filesystem::path(path).parent_path();
  if (!parent.empty()) {
    filesystem::create_directories(parent);
  }
}

/// Buffered text writer with fast integer formatting.
class TextWriter {
public:
  explicit TextWriter(const string &path) {
    createParentDirectory(path);
    file_ = fopen(path.c_str(), "wb");
    buffer_.reserve(kFlushSize + 64);
  }
  ~TextWriter() { close(); }
  bool ok() const { return file_ != nullptr && !failed_; }
  void put(long long value) {
    char digits[24];
    auto result = to_chars(digits, digits + sizeof(digits), value);
    buffer_.append(digits, result.ptr);
    maybeFlush();
  }
  void put(const char *text) {
    buffer_.append(text);
    maybeFlush();
  }
  void putChar(char c) {
    buffer_.push_back(c);
    maybeFlush();
  }
  bool close() {
    if (file_ == nullptr) {
      return !failed_;
    }
    flush();
    if (fclose(file_) != 0) {
      failed_ = true;
    }
    file_ = nullptr;
    return !failed_;
  }

private:
  static constexpr size_t kFlushSize = 1 << 22;
  void maybeFlush() {
    if (buffer_.size() >= kFlushSize) {
      flush();
    }
  }
  void flush() {
    if (file_ != nullptr && !buffer_.empty() &&
        fwrite(buffer_.data(), 1, buffer_.size(), file_) != buffer_.size()) {
      failed_ = true;
    }
    buffer_.clear();
  }
  FILE *file_ = nullptr;
  string buffer_;
  bool failed_ = false;
};

/// Minimal scanner over an in-memory text file.
class TextScanner {
public:
  explicit TextScanner(const string &text)
      : pos_(text.data()), end_(text.data() + text.size()) {}

  /// Skip spaces, tabs and carriage returns (not newlines).
  void skipBlanks() {
    while (pos_ < end_ && (*pos_ == ' ' || *pos_ == '\t' || *pos_ == '\r')) {
      ++pos_;
    }
  }
  /// Skip all whitespace including newlines.
  void skipWhitespace() {
    while (pos_ < end_ && (*pos_ == ' ' || *pos_ == '\t' || *pos_ == '\r' ||
                           *pos_ == '\n')) {
      ++pos_;
    }
  }
  bool atEnd() const { return pos_ >= end_; }
  bool atNewline() const { return pos_ < end_ && *pos_ == '\n'; }
  void advance() { ++pos_; }
  bool startsWith(const char *word) const {
    size_t len = strlen(word);
    return static_cast<size_t>(end_ - pos_) >= len &&
           memcmp(pos_, word, len) == 0;
  }
  void skip(size_t count) { pos_ += count; }

  /// Parse a (possibly negative) integer at the current position.
  bool readInt(long long &value) {
    auto result = from_chars(pos_, end_, value);
    if (result.ec != errc()) {
      return false;
    }
    pos_ = result.ptr;
    return true;
  }

private:
  const char *pos_;
  const char *end_;
};

/// Parse a file of whitespace-separated integers.
bool readIntFile(const string &path, vector<int> &values) {
  string text;
  if (!readWholeFile(path, text)) {
    return false;
  }
  TextScanner scanner(text);
  values.clear();
  long long value = 0;
  scanner.skipWhitespace();
  while (!scanner.atEnd()) {
    if (!scanner.readInt(value)) {
      return false;
    }
    values.push_back(static_cast<int>(value));
    scanner.skipWhitespace();
  }
  return true;
}

bool fileIsNewer(const string &a, const string &b) {
  error_code ec1, ec2;
  auto ta = filesystem::last_write_time(a, ec1);
  auto tb = filesystem::last_write_time(b, ec2);
  return !ec1 && !ec2 && ta >= tb;
}

constexpr char kBinaryMagic[8] = {'M', 'O', 'S', 'P', 'C', 'S', 'R', '1'};

} // namespace

// ============================================================================
// Graph I/O
// ============================================================================

namespace {

/// Parse a Values file: one line per edge with K weights (K inferred).
bool readValuesFile(const string &path, vector<int> &weights, int &objectives,
                    long long &lines, string &error) {
  string text;
  if (!readWholeFile(path, text)) {
    error = "Could not read CSR values file.";
    return false;
  }
  TextScanner scanner(text);
  weights.clear();
  weights.reserve(text.size() / 3);
  objectives = 0;
  lines = 0;
  while (true) {
    scanner.skipBlanks();
    int onLine = 0;
    long long value = 0;
    while (!scanner.atEnd() && !scanner.atNewline()) {
      if (!scanner.readInt(value)) {
        error = "Invalid token in CSR values file.";
        return false;
      }
      weights.push_back(static_cast<int>(value));
      ++onLine;
      scanner.skipBlanks();
    }
    if (onLine > 0) {
      if (objectives == 0) {
        objectives = onLine;
      } else if (onLine != objectives) {
        error = "Inconsistent number of objectives.";
        return false;
      }
      ++lines;
    }
    if (scanner.atEnd()) {
      break;
    }
    scanner.advance(); // newline
  }
  return true;
}

} // namespace

bool runConcurrently(const vector<function<bool()>> &jobs) {
  const int count = static_cast<int>(jobs.size());
  vector<char> ok(count, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(count)
  for (int i = 0; i < count; ++i) {
    ok[i] = jobs[i]() ? 1 : 0;
  }
#else
  vector<future<bool>> running;
  for (const auto &job : jobs) {
    running.push_back(async(launch::async, job));
  }
  for (int i = 0; i < count; ++i) {
    ok[i] = running[i].get() ? 1 : 0;
  }
#endif
  bool all = true;
  for (char flag : ok) {
    all = all && flag;
  }
  return all;
}

bool readCsrGraph(const string &prefix, CsrGraph &graph) {
  graph = CsrGraph();
  // The three files are independent: parse them concurrently.
  bool rowsOk = false, colsOk = false, valuesOk = false;
  int objectives = 0;
  long long lines = 0;
  string valuesError;
  runConcurrently({
      [&] { return rowsOk = readIntFile(prefix + "RowPtr.txt", graph.rowPtr); },
      [&] { return colsOk = readIntFile(prefix + "ColInd.txt", graph.colInd); },
      [&] {
        return valuesOk = readValuesFile(prefix + "Values.txt", graph.weights,
                                         objectives, lines, valuesError);
      },
  });

  if (!rowsOk || graph.rowPtr.size() < 2) {
    cout << "Error: Could not read CSR row pointers: " << prefix
         << "RowPtr.txt\n";
    return false;
  }
  graph.numberOfNodes = static_cast<int>(graph.rowPtr.size()) - 1;
  const int numberOfEdges = graph.rowPtr.back();
  if (graph.rowPtr[0] != 0) {
    cout << "Error: CSR row pointers must start at 0.\n";
    return false;
  }
  for (int i = 0; i < graph.numberOfNodes; ++i) {
    if (graph.rowPtr[i + 1] < graph.rowPtr[i]) {
      cout << "Error: CSR row pointers are not monotone.\n";
      return false;
    }
  }
  if (!colsOk || static_cast<int>(graph.colInd.size()) != numberOfEdges) {
    cout << "Error: CSR column index file missing or size mismatch.\n";
    return false;
  }
  for (int v : graph.colInd) {
    if (v < 0 || v >= graph.numberOfNodes) {
      cout << "Error: CSR column index out of range.\n";
      return false;
    }
  }
  if (!valuesOk) {
    cout << "Error: " << valuesError << "\n";
    return false;
  }
  if (lines != numberOfEdges) {
    cout << "Error: values size mismatch.\n";
    return false;
  }
  graph.numberOfObjectives = objectives;
  return true;
}

bool writeCsrGraph(const string &prefix, const CsrGraph &graph) {
  TextWriter rows(prefix + "RowPtr.txt");
  TextWriter cols(prefix + "ColInd.txt");
  TextWriter values(prefix + "Values.txt");
  if (!rows.ok() || !cols.ok() || !values.ok()) {
    cout << "Error: Could not write CSR files: " << prefix << "\n";
    return false;
  }
  for (int x : graph.rowPtr) {
    rows.put(x);
    rows.putChar('\n');
  }
  for (int x : graph.colInd) {
    cols.put(x);
    cols.putChar('\n');
  }
  const int K = graph.numberOfObjectives;
  for (int e = 0; e < graph.numberOfEdges(); ++e) {
    for (int k = 0; k < K; ++k) {
      values.put(graph.weight(e, k));
      if (k + 1 < K) {
        values.putChar(' ');
      }
    }
    values.putChar('\n');
  }
  return rows.close() && cols.close() && values.close();
}

bool saveCsrGraphBinary(const string &path, const CsrGraph &graph) {
  createParentDirectory(path);
  FILE *file = fopen(path.c_str(), "wb");
  if (file == nullptr) {
    return false;
  }
  int32_t header[2] = {graph.numberOfNodes, graph.numberOfObjectives};
  int64_t numberOfEdges = graph.numberOfEdges();
  bool ok = fwrite(kBinaryMagic, 1, 8, file) == 8 &&
            fwrite(header, sizeof(header), 1, file) == 1 &&
            fwrite(&numberOfEdges, sizeof(numberOfEdges), 1, file) == 1 &&
            fwrite(graph.rowPtr.data(), sizeof(int), graph.rowPtr.size(),
                   file) == graph.rowPtr.size() &&
            fwrite(graph.colInd.data(), sizeof(int), graph.colInd.size(),
                   file) == graph.colInd.size() &&
            fwrite(graph.weights.data(), sizeof(int), graph.weights.size(),
                   file) == graph.weights.size();
  ok = (fclose(file) == 0) && ok;
  if (!ok) {
    filesystem::remove(path);
  }
  return ok;
}

bool loadCsrGraphBinary(const string &path, CsrGraph &graph) {
  FILE *file = fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return false;
  }
  char magic[8];
  int32_t header[2];
  int64_t numberOfEdges = 0;
  bool ok = fread(magic, 1, 8, file) == 8 &&
            memcmp(magic, kBinaryMagic, 8) == 0 &&
            fread(header, sizeof(header), 1, file) == 1 &&
            fread(&numberOfEdges, sizeof(numberOfEdges), 1, file) == 1 &&
            header[0] > 0 && header[1] > 0 && numberOfEdges >= 0;
  if (ok) {
    graph.numberOfNodes = header[0];
    graph.numberOfObjectives = header[1];
    graph.rowPtr.resize(static_cast<size_t>(header[0]) + 1);
    graph.colInd.resize(static_cast<size_t>(numberOfEdges));
    graph.weights.resize(static_cast<size_t>(numberOfEdges) * header[1]);
    ok = fread(graph.rowPtr.data(), sizeof(int), graph.rowPtr.size(), file) ==
             graph.rowPtr.size() &&
         fread(graph.colInd.data(), sizeof(int), graph.colInd.size(), file) ==
             graph.colInd.size() &&
         fread(graph.weights.data(), sizeof(int), graph.weights.size(),
               file) == graph.weights.size() &&
         graph.rowPtr.back() == numberOfEdges;
  }
  fclose(file);
  return ok;
}

bool loadCsrGraph(const string &prefix, CsrGraph &graph,
                  const string &cachePath) {
  if (!cachePath.empty()) {
    bool fresh = fileIsNewer(cachePath, prefix + "RowPtr.txt") &&
                 fileIsNewer(cachePath, prefix + "ColInd.txt") &&
                 fileIsNewer(cachePath, prefix + "Values.txt");
    if (fresh && loadCsrGraphBinary(cachePath, graph)) {
      return true;
    }
  }
  if (!readCsrGraph(prefix, graph)) {
    return false;
  }
  if (!cachePath.empty() && !saveCsrGraphBinary(cachePath, graph)) {
    cout << "Warning: could not write binary cache " << cachePath << "\n";
  }
  return true;
}

// ============================================================================
// Change batches
// ============================================================================

bool readChangeBatch(const string &insertPath, const string &deletePath,
                     int numberOfObjectives, int numberOfNodes,
                     ChangeBatch &batch) {
  batch = ChangeBatch();
  batch.numberOfObjectives = numberOfObjectives;
  auto inRange = [&](long long x) { return x >= 0 && x < numberOfNodes; };

  string text;
  if (!readWholeFile(insertPath, text)) {
    cout << "Error: Could not open insert file: " << insertPath << "\n";
    return false;
  }
  {
    TextScanner scanner(text);
    vector<long long> tokens;
    while (true) {
      scanner.skipBlanks();
      tokens.clear();
      long long value = 0;
      while (!scanner.atEnd() && !scanner.atNewline()) {
        if (!scanner.readInt(value)) {
          cout << "Error: Invalid insert line.\n";
          return false;
        }
        tokens.push_back(value);
        scanner.skipBlanks();
      }
      if (!tokens.empty()) {
        if (static_cast<int>(tokens.size()) < 2 + numberOfObjectives) {
          cout << "Error: Invalid insert line.\n";
          return false;
        }
        if (!inRange(tokens[0]) || !inRange(tokens[1])) {
          cout << "Error: Inserted edge endpoint out of range.\n";
          return false;
        }
        batch.insertFrom.push_back(static_cast<int>(tokens[0]));
        batch.insertTo.push_back(static_cast<int>(tokens[1]));
        for (int k = 0; k < numberOfObjectives; ++k) {
          batch.insertWeights.push_back(static_cast<int>(tokens[2 + k]));
        }
      }
      if (scanner.atEnd()) {
        break;
      }
      scanner.advance();
    }
  }

  if (!readWholeFile(deletePath, text)) {
    cout << "Error: Could not open delete file: " << deletePath << "\n";
    return false;
  }
  {
    TextScanner scanner(text);
    while (true) {
      scanner.skipBlanks();
      long long pair[2];
      int count = 0;
      long long value = 0;
      while (!scanner.atEnd() && !scanner.atNewline()) {
        if (!scanner.readInt(value)) {
          cout << "Error: Invalid delete line.\n";
          return false;
        }
        if (count < 2) {
          pair[count] = value;
        }
        ++count;
        scanner.skipBlanks();
      }
      if (count >= 2) { // shorter lines are ignored, as before
        if (!inRange(pair[0]) || !inRange(pair[1])) {
          cout << "Error: Deleted edge endpoint out of range.\n";
          return false;
        }
        batch.deleteFrom.push_back(static_cast<int>(pair[0]));
        batch.deleteTo.push_back(static_cast<int>(pair[1]));
      }
      if (scanner.atEnd()) {
        break;
      }
      scanner.advance();
    }
  }
  return true;
}

bool applyChangeBatch(const CsrGraph &original, ChangeBatch &batch,
                      CsrGraph &updated) {
  const int n = original.numberOfNodes;
  const int K = original.numberOfObjectives;
  if (batch.numberOfObjectives != K) {
    cout << "Error: change batch has " << batch.numberOfObjectives
         << " objectives, graph has " << K << ".\n";
    return false;
  }
  if (K > 32) {
    cout << "Error: at most 32 objectives are supported.\n";
    return false;
  }
  const int numInserts = batch.numberOfInserts();
  const int numDeletes = batch.numberOfDeletes();

  // Group changes by source row, keeping file order inside each row.
  vector<int> delStart(n + 1, 0), insStart(n + 1, 0);
  for (int i = 0; i < numDeletes; ++i) {
    ++delStart[batch.deleteFrom[i] + 1];
  }
  for (int i = 0; i < numInserts; ++i) {
    ++insStart[batch.insertFrom[i] + 1];
  }
  for (int u = 0; u < n; ++u) {
    delStart[u + 1] += delStart[u];
    insStart[u + 1] += insStart[u];
  }
  vector<int> delOrder(numDeletes), insOrder(numInserts);
  {
    vector<int> cursor(delStart.begin(), delStart.end() - 1);
    for (int i = 0; i < numDeletes; ++i) {
      delOrder[cursor[batch.deleteFrom[i]]++] = i;
    }
    cursor.assign(insStart.begin(), insStart.end() - 1);
    for (int i = 0; i < numInserts; ++i) {
      insOrder[cursor[batch.insertFrom[i]]++] = i;
    }
  }

  // Rebuild every changed row. Each row entry is (neighbour, weight slot);
  // slot >= 0 refers to an original edge, slot < 0 to insertion -slot-1.
  struct RowEntry {
    int vertex;
    long long slot;
    bool alive;
  };
  vector<int> changedRows;
  vector<vector<RowEntry>> changedContents;
  vector<int> rowIndex(n, -1);
  batch.weightIncreaseMask.assign(numInserts, 0u);
  vector<int> finalSlotOfInsert(numInserts, 0);

  for (int u = 0; u < n; ++u) {
    if (delStart[u] == delStart[u + 1] && insStart[u] == insStart[u + 1]) {
      continue;
    }
    vector<RowEntry> row;
    for (int e = original.rowPtr[u]; e < original.rowPtr[u + 1]; ++e) {
      row.push_back({original.colInd[e], e, true});
    }
    for (int j = delStart[u]; j < delStart[u + 1]; ++j) {
      int v = batch.deleteTo[delOrder[j]];
      for (auto &entry : row) {
        if (entry.alive && entry.vertex == v) {
          entry.alive = false;
          break;
        }
      }
    }
    for (int j = insStart[u]; j < insStart[u + 1]; ++j) {
      int i = insOrder[j];
      int v = batch.insertTo[i];
      bool replaced = false;
      for (auto &entry : row) {
        if (entry.alive && entry.vertex == v) {
          entry.slot = -static_cast<long long>(i) - 1;
          replaced = true;
          break;
        }
      }
      if (!replaced) {
        row.push_back({v, -static_cast<long long>(i) - 1, true});
      }
    }
    rowIndex[u] = static_cast<int>(changedRows.size());
    changedRows.push_back(u);
    changedContents.push_back(move(row));
  }

  // Assemble the updated CSR.
  updated = CsrGraph();
  updated.numberOfNodes = n;
  updated.numberOfObjectives = K;
  updated.rowPtr.assign(n + 1, 0);
  for (int u = 0; u < n; ++u) {
    int degree = original.rowPtr[u + 1] - original.rowPtr[u];
    if (rowIndex[u] >= 0) {
      degree = 0;
      for (const auto &entry : changedContents[rowIndex[u]]) {
        degree += entry.alive ? 1 : 0;
      }
    }
    updated.rowPtr[u + 1] = updated.rowPtr[u] + degree;
  }
  const size_t m = static_cast<size_t>(updated.rowPtr[n]);
  updated.colInd.resize(m);
  updated.weights.resize(m * K);
  for (int u = 0; u < n; ++u) {
    int out = updated.rowPtr[u];
    if (rowIndex[u] < 0) {
      int begin = original.rowPtr[u], end = original.rowPtr[u + 1];
      copy(original.colInd.begin() + begin, original.colInd.begin() + end,
           updated.colInd.begin() + out);
      copy(original.weights.begin() + static_cast<size_t>(begin) * K,
           original.weights.begin() + static_cast<size_t>(end) * K,
           updated.weights.begin() + static_cast<size_t>(out) * K);
      continue;
    }
    for (const auto &entry : changedContents[rowIndex[u]]) {
      if (!entry.alive) {
        continue;
      }
      updated.colInd[out] = entry.vertex;
      const int *src =
          entry.slot >= 0
              ? &original.weights[static_cast<size_t>(entry.slot) * K]
              : &batch.insertWeights[static_cast<size_t>(-entry.slot - 1) *
                                     K];
      copy(src, src + K, updated.weights.begin() + static_cast<size_t>(out) * K);
      ++out;
    }
  }

  // Weight increases: compare the final weight of (u,v) with its weight in
  // the original graph (first occurrence of v in row u).
  for (int i = 0; i < numInserts; ++i) {
    int u = batch.insertFrom[i], v = batch.insertTo[i];
    int originalEdge = -1;
    for (int e = original.rowPtr[u]; e < original.rowPtr[u + 1]; ++e) {
      if (original.colInd[e] == v) {
        originalEdge = e;
        break;
      }
    }
    if (originalEdge < 0) {
      continue;
    }
    int finalEdge = -1;
    for (int e = updated.rowPtr[u]; e < updated.rowPtr[u + 1]; ++e) {
      if (updated.colInd[e] == v) {
        finalEdge = e;
        break;
      }
    }
    if (finalEdge < 0) {
      continue;
    }
    unsigned int mask = 0;
    for (int k = 0; k < K; ++k) {
      if (updated.weight(finalEdge, k) > original.weight(originalEdge, k)) {
        mask |= 1u << k;
      }
    }
    batch.weightIncreaseMask[i] = mask;
  }
  return true;
}

void transposeCsrGraph(const CsrGraph &graph, CsrGraph &reverse) {
  const int n = graph.numberOfNodes;
  const int K = graph.numberOfObjectives;
  const size_t m = static_cast<size_t>(graph.numberOfEdges());
  reverse = CsrGraph();
  reverse.numberOfNodes = n;
  reverse.numberOfObjectives = K;
  reverse.rowPtr.assign(n + 1, 0);
  for (size_t e = 0; e < m; ++e) {
    ++reverse.rowPtr[graph.colInd[e] + 1];
  }
  for (int v = 0; v < n; ++v) {
    reverse.rowPtr[v + 1] += reverse.rowPtr[v];
  }
  reverse.colInd.resize(m);
  reverse.weights.resize(m * K);
  vector<int> cursor(reverse.rowPtr.begin(), reverse.rowPtr.end() - 1);
  for (int u = 0; u < n; ++u) {
    for (int e = graph.rowPtr[u]; e < graph.rowPtr[u + 1]; ++e) {
      int pos = cursor[graph.colInd[e]]++;
      reverse.colInd[pos] = u;
      for (int k = 0; k < K; ++k) {
        reverse.weights[static_cast<size_t>(pos) * K + k] = graph.weight(e, k);
      }
    }
  }
}

// ============================================================================
// Distance / parent files
// ============================================================================

bool readDistances(const string &path, int numberOfNodes,
                   vector<long long> &distances) {
  string text;
  if (!readWholeFile(path, text)) {
    cout << "Error: Could not open distances file: " << path << "\n";
    return false;
  }
  distances.assign(numberOfNodes, DISTANCE_INF);
  TextScanner scanner(text);
  while (true) {
    scanner.skipWhitespace();
    if (scanner.atEnd()) {
      break;
    }
    long long vertex = 0, value = 0;
    if (!scanner.readInt(vertex)) {
      cout << "Error: Invalid line in distances file: " << path << "\n";
      return false;
    }
    scanner.skipBlanks();
    if (scanner.startsWith("INF")) {
      scanner.skip(3);
      value = DISTANCE_INF;
    } else if (!scanner.readInt(value)) {
      cout << "Error: Invalid line in distances file: " << path << "\n";
      return false;
    }
    if (vertex < 0 || vertex >= numberOfNodes) {
      cout << "Error: Vertex ID out of range in distances file.\n";
      return false;
    }
    distances[vertex] = value;
  }
  return true;
}

bool readParents(const string &path, int numberOfNodes, vector<int> &parent) {
  string text;
  if (!readWholeFile(path, text)) {
    cout << "Error: Could not open SSSP tree file: " << path << "\n";
    return false;
  }
  parent.assign(numberOfNodes, -1);
  TextScanner scanner(text);
  while (true) {
    scanner.skipWhitespace();
    if (scanner.atEnd()) {
      break;
    }
    long long vertex = 0, value = 0;
    if (!scanner.readInt(vertex)) {
      cout << "Error: Invalid line in SSSP tree file: " << path << "\n";
      return false;
    }
    scanner.skipBlanks();
    if (!scanner.readInt(value)) {
      cout << "Error: Invalid line in SSSP tree file: " << path << "\n";
      return false;
    }
    if (vertex < 0 || vertex >= numberOfNodes || value < -1 ||
        value >= numberOfNodes) {
      cout << "Error: Vertex ID out of range in SSSP tree file.\n";
      return false;
    }
    parent[vertex] = static_cast<int>(value);
  }
  return true;
}

bool writeDistances(const string &path, const vector<long long> &distances) {
  TextWriter out(path);
  if (!out.ok()) {
    cout << "Error: Could not write distances file: " << path << "\n";
    return false;
  }
  for (size_t v = 0; v < distances.size(); ++v) {
    out.put(static_cast<long long>(v));
    out.putChar(' ');
    if (distances[v] >= DISTANCE_INF / 2) {
      out.put("INF");
    } else {
      out.put(distances[v]);
    }
    out.putChar('\n');
  }
  return out.close();
}

bool writeParents(const string &path, const vector<int> &parent) {
  TextWriter out(path);
  if (!out.ok()) {
    cout << "Error: Could not write SSSP tree file: " << path << "\n";
    return false;
  }
  for (size_t v = 0; v < parent.size(); ++v) {
    out.put(static_cast<long long>(v));
    out.putChar(' ');
    out.put(static_cast<long long>(parent[v]));
    out.putChar('\n');
  }
  return out.close();
}
