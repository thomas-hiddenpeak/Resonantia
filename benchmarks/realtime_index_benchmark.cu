// Copyright 2026 Resonantia Authors.
// Licensed under the MIT License.
//
// R0-only diagnostic benchmark for exact Flat Index retrieval on CUDA.
//
// This file is intentionally standalone and is not part of the product build.
// See benchmarks/README.md for the evidence boundary and manual build command.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <sys/utsname.h>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kHeaderBytes = 16;
constexpr std::uint64_t kMaxRows = 1'000'000;
constexpr std::uint32_t kMaxDimension = 4'096;
constexpr std::uint32_t kMaxQueries = 16;
constexpr std::uint32_t kMaxK = 16;
constexpr std::uint32_t kMaxWarmups = 10'000;
constexpr std::uint32_t kMaxRepetitions = 10'000;
constexpr std::uint64_t kMaxDatabaseBytes = 2ULL * 1024 * 1024 * 1024;
constexpr std::uint64_t kMaxDistanceWorkspaceBytes = 512ULL * 1024 * 1024;
constexpr std::uint64_t kMaxPersistentDeviceBytes = 3ULL * 1024 * 1024 * 1024;
constexpr std::size_t kCublasWorkspaceBytes = 4U * 1024 * 1024;
constexpr std::size_t kIoChunkFloats = 1U << 18;
constexpr double kDistanceAbsoluteTolerance = 5e-4;
constexpr double kDistanceRelativeTolerance = 5e-5;
constexpr double kWeightAbsoluteTolerance = 1e-5;
constexpr double kWeightRelativeTolerance = 1e-4;
constexpr double kRetrievedAbsoluteTolerance = 1e-4;
constexpr double kRetrievedCosineMinimum = 0.999999;
constexpr double kBlendedAbsoluteTolerance = 1e-4;
constexpr double kBlendedCosineMinimum = 0.999999;

static_assert(std::endian::native == std::endian::little,
              "the existing Flat Index payload is little-endian float32");

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void check_cuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        fail(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, std::string_view operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fail(std::string(operation) + ": cuBLAS status " +
             std::to_string(static_cast<int>(status)));
    }
}

bool checked_mul(std::uint64_t left, std::uint64_t right,
                 std::uint64_t& result) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool checked_add(std::uint64_t left, std::uint64_t right,
                 std::uint64_t& result) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    result = left + right;
    return true;
}

std::uint64_t checked_product(std::uint64_t left, std::uint64_t right,
                              std::string_view label) {
    std::uint64_t result = 0;
    if (!checked_mul(left, right, result)) {
        fail(std::string(label) + " size overflows uint64");
    }
    return result;
}

std::uint64_t parse_unsigned(std::string_view text, std::string_view option) {
    if (text.empty()) fail(std::string(option) + " requires a value");
    std::uint64_t value = 0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value, 10);
    if (error != std::errc{} || end != text.data() + text.size()) {
        fail("invalid unsigned integer for " + std::string(option) + ": " +
             std::string(text));
    }
    return value;
}

std::uint32_t parse_u32(std::string_view text, std::string_view option) {
    const auto value = parse_unsigned(text, option);
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        fail(std::string(option) + " exceeds uint32");
    }
    return static_cast<std::uint32_t>(value);
}

double parse_real(std::string_view text, std::string_view option) {
    if (text.empty()) fail(std::string(option) + " requires a value");
    double value = 0.0;
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), value, std::chars_format::general);
    if (error != std::errc{} || end != text.data() + text.size() ||
        !std::isfinite(value)) {
        fail("invalid finite number for " + std::string(option) + ": " +
             std::string(text));
    }
    return value;
}

struct Options {
    std::optional<std::uint64_t> synthetic_rows;
    std::optional<std::string> index_path;
    std::optional<std::string> query_index_path;
    std::uint32_t dimension = 768;
    std::uint32_t queries = 1;
    std::uint32_t k = 8;
    std::uint32_t warmups = 10;
    std::uint32_t repetitions = 100;
    std::uint32_t seed = 0x12345678U;
    double index_rate = 0.75;
    int device = 0;
};

void print_usage(std::ostream& stream) {
    stream
        << "Usage:\n"
        << "  realtime_index_benchmark (--rows <N> | --index <flat.index>) [options]\n\n"
        << "Input:\n"
        << "  --rows <N>             Generate a deterministic synthetic database\n"
        << "  --index <path>         Load an existing Flat Index\n"
        << "  --query-index <path>   Use the first --queries rows from another Flat Index\n\n"
        << "Envelope:\n"
        << "  --dim <N>              Synthetic dimension (default: 768)\n"
        << "  --nq <N>               Query batch, 1..16 (default: 1)\n"
        << "  --queries <N>          Alias for --nq\n"
        << "  --k <N>                Exact top-k, 1..16 (default: 8)\n"
        << "  --index-rate <R>       Positive blend rate in (0,1] (default: 0.75)\n"
        << "  --seed <N>             Synthetic uint32 seed (default: 305419896)\n"
        << "  --warmup <N>           Untimed warmup calls (default: 10)\n"
        << "  --reps <N>             Timed calls (default: 100)\n"
        << "  --device <N>           CUDA device ordinal (default: 0)\n"
        << "  --help                 Print this help\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool seen_dim = false;
    bool seen_queries = false;
    bool seen_k = false;
    bool seen_warmup = false;
    bool seen_repetitions = false;
    bool seen_seed = false;
    bool seen_index_rate = false;
    bool seen_device = false;

    const auto next = [&](int& index, std::string_view option) -> std::string_view {
        if (index + 1 >= argc) fail(std::string(option) + " requires a value");
        return argv[++index];
    };
    const auto duplicate = [](bool& seen, std::string_view option) {
        if (seen) fail("duplicate option: " + std::string(option));
        seen = true;
    };

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--help") {
            print_usage(std::cout);
            std::exit(0);
        } else if (argument == "--rows") {
            if (options.synthetic_rows.has_value()) fail("duplicate option: --rows");
            options.synthetic_rows = parse_unsigned(next(index, argument), argument);
        } else if (argument == "--index") {
            if (options.index_path.has_value()) fail("duplicate option: --index");
            const auto value = next(index, argument);
            if (value.empty()) fail("--index cannot be empty");
            options.index_path = std::string(value);
        } else if (argument == "--query-index") {
            if (options.query_index_path.has_value()) {
                fail("duplicate option: --query-index");
            }
            const auto value = next(index, argument);
            if (value.empty()) fail("--query-index cannot be empty");
            options.query_index_path = std::string(value);
        } else if (argument == "--dim") {
            duplicate(seen_dim, argument);
            options.dimension = parse_u32(next(index, argument), argument);
        } else if (argument == "--nq" || argument == "--queries") {
            duplicate(seen_queries, argument);
            options.queries = parse_u32(next(index, argument), argument);
        } else if (argument == "--k") {
            duplicate(seen_k, argument);
            options.k = parse_u32(next(index, argument), argument);
        } else if (argument == "--warmup") {
            duplicate(seen_warmup, argument);
            options.warmups = parse_u32(next(index, argument), argument);
        } else if (argument == "--reps") {
            duplicate(seen_repetitions, argument);
            options.repetitions = parse_u32(next(index, argument), argument);
        } else if (argument == "--seed") {
            duplicate(seen_seed, argument);
            options.seed = parse_u32(next(index, argument), argument);
        } else if (argument == "--index-rate") {
            duplicate(seen_index_rate, argument);
            options.index_rate = parse_real(next(index, argument), argument);
        } else if (argument == "--device") {
            duplicate(seen_device, argument);
            const auto value = parse_unsigned(next(index, argument), argument);
            if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                fail("--device exceeds int");
            }
            options.device = static_cast<int>(value);
        } else {
            fail("unknown option: " + std::string(argument));
        }
    }

    if (options.synthetic_rows.has_value() == options.index_path.has_value()) {
        fail("exactly one of --rows or --index is required");
    }
    if (options.index_path.has_value() && seen_dim) {
        fail("--dim is only valid with --rows");
    }
    if (options.index_path.has_value() && seen_seed) {
        fail("--seed is only valid with --rows");
    }
    if (options.synthetic_rows.has_value() &&
        (*options.synthetic_rows == 0 || *options.synthetic_rows > kMaxRows)) {
        fail("--rows must be in 1.." + std::to_string(kMaxRows));
    }
    if (options.dimension == 0 || options.dimension > kMaxDimension) {
        fail("--dim must be in 1.." + std::to_string(kMaxDimension));
    }
    if (options.queries == 0 || options.queries > kMaxQueries) {
        fail("--queries must be in 1.." + std::to_string(kMaxQueries));
    }
    if (options.k == 0 || options.k > kMaxK) {
        fail("--k must be in 1.." + std::to_string(kMaxK));
    }
    if (options.warmups > kMaxWarmups) {
        fail("--warmup must be in 0.." + std::to_string(kMaxWarmups));
    }
    if (options.repetitions == 0 || options.repetitions > kMaxRepetitions) {
        fail("--reps must be in 1.." + std::to_string(kMaxRepetitions));
    }
    if (!(options.index_rate > 0.0 && options.index_rate <= 1.0)) {
        fail("--index-rate must be finite and in (0,1]");
    }
    return options;
}

class Sha256 {
public:
    Sha256()
        : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                 0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

    void update(const std::uint8_t* data, std::size_t size) {
        for (std::size_t index = 0; index < size; ++index) {
            buffer_[buffer_size_++] = data[index];
            if (buffer_size_ == buffer_.size()) {
                transform();
                bit_count_ += 512;
                buffer_size_ = 0;
            }
        }
    }

    std::string finish_hex() {
        const std::uint64_t total_bits = bit_count_ + buffer_size_ * 8;
        buffer_[buffer_size_++] = 0x80U;
        if (buffer_size_ > 56) {
            while (buffer_size_ < 64) buffer_[buffer_size_++] = 0;
            transform();
            buffer_size_ = 0;
        }
        while (buffer_size_ < 56) buffer_[buffer_size_++] = 0;
        for (int index = 7; index >= 0; --index) {
            buffer_[buffer_size_++] =
                static_cast<std::uint8_t>(total_bits >> (index * 8));
        }
        transform();

        std::ostringstream result;
        result << std::hex << std::setfill('0');
        for (const auto value : state_) result << std::setw(8) << value;
        return result.str();
    }

private:
    static std::uint32_t rotate_right(std::uint32_t value, int amount) {
        return (value >> amount) | (value << (32 - amount));
    }

    void transform() {
        static constexpr std::array<std::uint32_t, 64> constants{
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
        };
        std::array<std::uint32_t, 64> words{};
        for (int index = 0; index < 16; ++index) {
            words[index] =
                (static_cast<std::uint32_t>(buffer_[index * 4]) << 24) |
                (static_cast<std::uint32_t>(buffer_[index * 4 + 1]) << 16) |
                (static_cast<std::uint32_t>(buffer_[index * 4 + 2]) << 8) |
                static_cast<std::uint32_t>(buffer_[index * 4 + 3]);
        }
        for (int index = 16; index < 64; ++index) {
            const auto s0 = rotate_right(words[index - 15], 7) ^
                            rotate_right(words[index - 15], 18) ^
                            (words[index - 15] >> 3);
            const auto s1 = rotate_right(words[index - 2], 17) ^
                            rotate_right(words[index - 2], 19) ^
                            (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }

        auto a = state_[0];
        auto b = state_[1];
        auto c = state_[2];
        auto d = state_[3];
        auto e = state_[4];
        auto f = state_[5];
        auto g = state_[6];
        auto h = state_[7];
        for (int index = 0; index < 64; ++index) {
            const auto sigma1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                                rotate_right(e, 25);
            const auto choose = (e & f) ^ (~e & g);
            const auto temporary1 = h + sigma1 + choose + constants[index] + words[index];
            const auto sigma0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                                rotate_right(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temporary2 = sigma0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint8_t, 64> buffer_{};
    std::array<std::uint32_t, 8> state_{};
    std::uint64_t bit_count_ = 0;
    std::size_t buffer_size_ = 0;
};

std::uint64_t read_u64_le(const std::uint8_t* bytes) {
    std::uint64_t value = 0;
    for (int index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8);
    }
    return value;
}

double maximum_l2_safe_component(std::uint32_t dimension) {
    return std::sqrt(static_cast<double>(std::numeric_limits<float>::max()) /
                     (8.0 * dimension));
}

struct MatrixSource {
    std::uint64_t rows = 0;
    std::uint32_t dimension = 0;
    std::vector<float> values;
    std::string sha256;
};

MatrixSource load_flat_index(const std::string& path,
                             std::optional<std::uint64_t> rows_to_store) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error) {
        fail("Flat Index is not a readable regular file: " + path);
    }
    const auto file_size = std::filesystem::file_size(path, error);
    if (error || file_size < kHeaderBytes) {
        fail("cannot determine a valid Flat Index size: " + path);
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) fail("cannot open Flat Index: " + path);
    std::array<std::uint8_t, kHeaderBytes> header{};
    input.read(reinterpret_cast<char*>(header.data()),
               static_cast<std::streamsize>(header.size()));
    if (!input) fail("cannot read Flat Index header: " + path);

    MatrixSource result;
    result.rows = read_u64_le(header.data());
    const auto dimension_u64 = read_u64_le(header.data() + 8);
    if (result.rows == 0 || result.rows > kMaxRows || dimension_u64 == 0 ||
        dimension_u64 > kMaxDimension) {
        fail("Flat Index shape exceeds the diagnostic envelope");
    }
    result.dimension = static_cast<std::uint32_t>(dimension_u64);
    const auto elements = checked_product(result.rows, result.dimension,
                                          "Flat Index element count");
    const auto payload_bytes = checked_product(elements, sizeof(float),
                                               "Flat Index payload");
    if (payload_bytes > kMaxDatabaseBytes) {
        fail("Flat Index payload exceeds the 2 GiB diagnostic cap");
    }
    std::uint64_t expected_size = 0;
    if (!checked_add(kHeaderBytes, payload_bytes, expected_size) ||
        expected_size != file_size) {
        fail("Flat Index size does not exactly match its header");
    }

    const auto stored_rows = std::min(result.rows, rows_to_store.value_or(result.rows));
    const auto stored_elements = checked_product(stored_rows, result.dimension,
                                                 "stored Flat Index elements");
    if (stored_elements > std::vector<float>().max_size()) {
        fail("stored Flat Index exceeds vector max_size");
    }
    result.values.resize(static_cast<std::size_t>(stored_elements));

    Sha256 hash;
    hash.update(header.data(), header.size());
    std::vector<float> chunk(kIoChunkFloats);
    const double maximum_component = maximum_l2_safe_component(result.dimension);
    std::uint64_t consumed = 0;
    while (consumed < elements) {
        const auto count_u64 = std::min<std::uint64_t>(elements - consumed, chunk.size());
        const auto count = static_cast<std::size_t>(count_u64);
        const auto bytes = count * sizeof(float);
        input.read(reinterpret_cast<char*>(chunk.data()),
                   static_cast<std::streamsize>(bytes));
        if (!input) fail("Flat Index payload is truncated: " + path);
        hash.update(reinterpret_cast<const std::uint8_t*>(chunk.data()), bytes);
        for (std::size_t index = 0; index < count; ++index) {
            if (!std::isfinite(chunk[index])) {
                fail("Flat Index contains a non-finite value");
            }
            if (std::abs(static_cast<double>(chunk[index])) > maximum_component) {
                fail("Flat Index contains a value that can overflow FP32 squared L2");
            }
        }
        if (consumed < stored_elements) {
            const auto copy_count = static_cast<std::size_t>(
                std::min<std::uint64_t>(count_u64, stored_elements - consumed));
            std::copy_n(chunk.data(), copy_count,
                        result.values.data() + static_cast<std::size_t>(consumed));
        }
        consumed += count_u64;
    }
    result.sha256 = hash.finish_hex();
    return result;
}

std::uint32_t lcg(std::uint32_t& state) {
    state = state * 1664525U + 1013904223U;
    return state;
}

float synthetic_value(std::uint32_t& state) {
    return static_cast<float>(static_cast<double>(lcg(state)) / 4294967296.0 - 0.5);
}

MatrixSource make_synthetic(std::uint64_t rows, std::uint32_t dimension,
                            std::uint32_t seed) {
    const auto elements = checked_product(rows, dimension, "synthetic database elements");
    const auto bytes = checked_product(elements, sizeof(float), "synthetic database bytes");
    if (bytes > kMaxDatabaseBytes || elements > std::vector<float>().max_size()) {
        fail("synthetic database exceeds the 2 GiB diagnostic cap");
    }
    MatrixSource result;
    result.rows = rows;
    result.dimension = dimension;
    result.values.resize(static_cast<std::size_t>(elements));
    auto state = seed;
    for (auto& value : result.values) value = synthetic_value(state);
    return result;
}

std::vector<float> make_synthetic_queries(std::uint32_t queries,
                                          std::uint32_t dimension,
                                          std::uint32_t seed) {
    const auto elements = checked_product(queries, dimension,
                                          "synthetic query elements");
    std::vector<float> result(static_cast<std::size_t>(elements));
    auto state = seed;
    for (auto& value : result) value = synthetic_value(state);
    return result;
}

void validate_envelope(const Options& options, std::uint64_t rows,
                       std::uint32_t dimension) {
    if (options.k > rows) fail("--k cannot exceed database rows");
    const auto database_elements = checked_product(rows, dimension,
                                                   "database elements");
    const auto database_bytes = checked_product(database_elements, sizeof(float),
                                                "database bytes");
    const auto distance_elements = checked_product(rows, options.queries,
                                                   "distance workspace elements");
    const auto distance_bytes = checked_product(distance_elements, sizeof(float),
                                                "distance workspace bytes");
    if (database_bytes > kMaxDatabaseBytes) {
        fail("database exceeds the 2 GiB diagnostic cap");
    }
    if (distance_bytes > kMaxDistanceWorkspaceBytes) {
        fail("distance workspace exceeds the 512 MiB diagnostic cap");
    }

    std::uint64_t persistent = database_bytes;
    const std::array<std::uint64_t, 10> additions{
        checked_product(rows, sizeof(float), "database norms"),
        checked_product(options.queries, dimension * sizeof(float), "query bytes"),
        checked_product(options.queries, sizeof(float), "query norms"),
        distance_bytes,
        checked_product(options.queries, options.k * sizeof(float), "top distances"),
        checked_product(options.queries, options.k * sizeof(std::int32_t), "top indices"),
        checked_product(options.queries, options.k * sizeof(double), "top weights"),
        checked_product(options.queries, dimension * sizeof(float), "retrieved bytes"),
        checked_product(options.queries, dimension * sizeof(float), "blended bytes"),
        kCublasWorkspaceBytes,
    };
    for (const auto addition : additions) {
        if (!checked_add(persistent, addition, persistent)) {
            fail("persistent device byte count overflows");
        }
    }
    if (persistent > kMaxPersistentDeviceBytes) {
        fail("persistent device state exceeds the 3 GiB diagnostic cap");
    }
}

struct Candidate {
    float distance;
    std::int32_t index;
};

__device__ bool candidate_better(const Candidate& left, const Candidate& right) {
    return left.distance < right.distance ||
           (left.distance == right.distance && left.index < right.index);
}

__device__ void insert_candidate(Candidate* values, int k, Candidate value) {
    if (!candidate_better(value, values[k - 1])) return;
    int position = k - 1;
    while (position > 0 && candidate_better(value, values[position - 1])) {
        values[position] = values[position - 1];
        --position;
    }
    values[position] = value;
}

__global__ void row_norms_kernel(const float* values, float* norms,
                                 int rows, int dimension) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0F;
    for (int column = threadIdx.x; column < dimension; column += blockDim.x) {
        const float value = values[static_cast<std::size_t>(row) * dimension + column];
        sum += value * value;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    __shared__ float warp_sums[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float total = lane < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0F;
        for (int offset = 16; offset > 0; offset /= 2) {
            total += __shfl_down_sync(0xffffffffU, total, offset);
        }
        if (lane == 0) norms[row] = total;
    }
}

__global__ void add_l2_norms_kernel(float* distances,
                                    const float* database_norms,
                                    const float* query_norms,
                                    int rows, int queries) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = rows * queries;
    if (linear >= count) return;
    const int query = linear / rows;
    const int row = linear - query * rows;
    const float distance = distances[linear] + database_norms[row] + query_norms[query];
    distances[linear] = distance > 0.0F ? distance : 0.0F;
}

__global__ void topk_kernel(const float* distances, int rows, int k,
                            float* top_distances, std::int32_t* top_indices) {
    const int query = blockIdx.x;
    Candidate local[kMaxK];
    for (int index = 0; index < kMaxK; ++index) {
        local[index] = {INFINITY, 2'147'483'647};
    }
    const float* query_distances =
        distances + static_cast<std::size_t>(query) * rows;
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        insert_candidate(local, k,
                         {query_distances[row], static_cast<std::int32_t>(row)});
    }

    __shared__ Candidate candidates[256 * kMaxK];
    for (int index = 0; index < k; ++index) {
        candidates[threadIdx.x * kMaxK + index] = local[index];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        Candidate best[kMaxK];
        for (int index = 0; index < kMaxK; ++index) {
            best[index] = {INFINITY, 2'147'483'647};
        }
        for (int index = 0; index < blockDim.x * kMaxK; ++index) {
            if (index % kMaxK < k) insert_candidate(best, k, candidates[index]);
        }
        for (int index = 0; index < k; ++index) {
            top_distances[query * k + index] = best[index].distance;
            top_indices[query * k + index] = best[index].index;
        }
    }
}

__global__ void weighted_blend_kernel(
        const float* database, const float* queries, int rows, int dimension, int k,
        double index_rate, const float* top_distances,
        const std::int32_t* top_indices, double* top_weights,
        float* retrieved_output, float* blended_output) {
    const int query = blockIdx.x;
    __shared__ double weights[kMaxK];
    __shared__ int valid;
    if (threadIdx.x == 0) {
        valid = 1;
        double sum = 0.0;
        for (int index = 0; index < k; ++index) {
            const float distance = top_distances[query * k + index];
            const auto row = top_indices[query * k + index];
            if (!isfinite(distance) || row < 0 || row >= rows) valid = 0;
            double weight = 1.0 / (static_cast<double>(distance) + 1e-8);
            weight *= weight;
            weights[index] = weight;
            sum += weight;
        }
        if (!isfinite(sum) || sum <= 0.0) valid = 0;
        if (valid != 0) {
            for (int index = 0; index < k; ++index) weights[index] /= sum;
        }
        for (int index = 0; index < k; ++index) {
            top_weights[query * k + index] = valid != 0 ? weights[index] : NAN;
        }
    }
    __syncthreads();
    for (int column = threadIdx.x; column < dimension; column += blockDim.x) {
        const auto offset = static_cast<std::size_t>(query) * dimension + column;
        if (valid == 0) {
            retrieved_output[offset] = NAN;
            blended_output[offset] = NAN;
            continue;
        }
        float retrieved = 0.0F;
        for (int index = 0; index < k; ++index) {
            const auto row = top_indices[query * k + index];
            retrieved += static_cast<float>(
                weights[index] *
                database[static_cast<std::size_t>(row) * dimension + column]);
        }
        retrieved_output[offset] = retrieved;
        blended_output[offset] = static_cast<float>(
            static_cast<double>(queries[offset]) * (1.0 - index_rate) +
            static_cast<double>(retrieved) * index_rate);
    }
}

class PersistentGpuExactIndex {
public:
    PersistentGpuExactIndex(const std::vector<float>& database,
                            const std::vector<float>& queries,
                            std::uint64_t rows, std::uint32_t dimension,
                            std::uint32_t query_count, std::uint32_t k,
                            double index_rate)
        : rows_(static_cast<int>(rows)),
          dimension_(static_cast<int>(dimension)),
          queries_(static_cast<int>(query_count)),
          k_(static_cast<int>(k)),
          index_rate_(index_rate) {
        try {
            check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags");
            check_cublas(cublasCreate(&handle_), "cublasCreate");
            check_cublas(cublasSetStream(handle_, stream_), "cublasSetStream");
            check_cublas(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH),
                         "cublasSetMathMode");

            allocate(cublas_workspace_, kCublasWorkspaceBytes,
                     "cuBLAS workspace");
            check_cublas(cublasSetWorkspace(handle_, cublas_workspace_,
                                             kCublasWorkspaceBytes),
                         "cublasSetWorkspace");
            allocate(database_, database.size(), "database");
            allocate(database_norms_, static_cast<std::size_t>(rows_),
                     "database norms");
            allocate(query_values_, queries.size(), "queries");
            allocate(query_norms_, static_cast<std::size_t>(queries_), "query norms");
            allocate(distances_, static_cast<std::size_t>(rows_) * queries_,
                     "distance workspace");
            allocate(top_distances_, static_cast<std::size_t>(queries_) * k_,
                     "top distances");
            allocate(top_indices_, static_cast<std::size_t>(queries_) * k_,
                     "top indices");
            allocate(top_weights_, static_cast<std::size_t>(queries_) * k_,
                     "top weights");
            allocate(retrieved_output_, queries.size(), "retrieved output");
            allocate(blended_output_, queries.size(), "blended output");

            check_cuda(cudaMemcpyAsync(database_, database.data(),
                                       database.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream_),
                       "copy database");
            check_cuda(cudaMemcpyAsync(query_values_, queries.data(),
                                       queries.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream_),
                       "copy queries");
            row_norms_kernel<<<rows_, 256, 0, stream_>>>(
                database_, database_norms_, rows_, dimension_);
            check_cuda(cudaGetLastError(), "launch database row norms");
            row_norms_kernel<<<queries_, 256, 0, stream_>>>(
                query_values_, query_norms_, queries_, dimension_);
            check_cuda(cudaGetLastError(), "launch query row norms");
            check_cuda(cudaStreamSynchronize(stream_), "initialize persistent index");
        } catch (...) {
            release();
            throw;
        }
    }

    PersistentGpuExactIndex(const PersistentGpuExactIndex&) = delete;
    PersistentGpuExactIndex& operator=(const PersistentGpuExactIndex&) = delete;

    ~PersistentGpuExactIndex() { release(); }

    void run() {
        const float alpha = -2.0F;
        const float beta = 0.0F;
        check_cublas(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 rows_, queries_, dimension_, &alpha,
                                 database_, dimension_, query_values_, dimension_,
                                 &beta, distances_, rows_),
                     "cublasSgemm exact distances");
        const int count = rows_ * queries_;
        add_l2_norms_kernel<<<(count + 255) / 256, 256, 0, stream_>>>(
            distances_, database_norms_, query_norms_, rows_, queries_);
        check_cuda(cudaGetLastError(), "launch add L2 norms");
        topk_kernel<<<queries_, 256, 0, stream_>>>(
            distances_, rows_, k_, top_distances_, top_indices_);
        check_cuda(cudaGetLastError(), "launch exact top-k");
        weighted_blend_kernel<<<queries_, 256, 0, stream_>>>(
            database_, query_values_, rows_, dimension_, k_, index_rate_,
            top_distances_, top_indices_, top_weights_, retrieved_output_,
            blended_output_);
        check_cuda(cudaGetLastError(), "launch weighted blend");
    }

    void synchronize() {
        check_cuda(cudaStreamSynchronize(stream_), "synchronize exact retrieval");
    }

    cudaStream_t stream() const noexcept { return stream_; }
    std::uint64_t persistent_bytes() const noexcept { return persistent_bytes_; }

    void copy_results(std::vector<float>& top_distances,
                      std::vector<std::int32_t>& top_indices,
                      std::vector<double>& top_weights,
                      std::vector<float>& retrieved_output,
                      std::vector<float>& blended_output) {
        check_cuda(cudaMemcpyAsync(top_distances.data(), top_distances_,
                                   top_distances.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_),
                   "copy top distances");
        check_cuda(cudaMemcpyAsync(top_indices.data(), top_indices_,
                                   top_indices.size() * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, stream_),
                   "copy top indices");
        check_cuda(cudaMemcpyAsync(top_weights.data(), top_weights_,
                                   top_weights.size() * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream_),
                   "copy top weights");
        check_cuda(cudaMemcpyAsync(retrieved_output.data(), retrieved_output_,
                                   retrieved_output.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_),
                   "copy retrieved output");
        check_cuda(cudaMemcpyAsync(blended_output.data(), blended_output_,
                                   blended_output.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_),
                   "copy blended output");
        synchronize();
    }

private:
    template <typename T>
    void allocate(T*& pointer, std::size_t count, std::string_view label) {
        if (count == 0 || count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            fail(std::string(label) + " allocation size is invalid");
        }
        const auto bytes = count * sizeof(T);
        check_cuda(cudaMalloc(&pointer, bytes), "allocate " + std::string(label));
        persistent_bytes_ += bytes;
    }

    void release() noexcept {
        if (stream_ != nullptr) cudaStreamSynchronize(stream_);
        if (handle_ != nullptr) cublasDestroy(handle_);
        cudaFree(blended_output_);
        cudaFree(retrieved_output_);
        cudaFree(top_weights_);
        cudaFree(top_indices_);
        cudaFree(top_distances_);
        cudaFree(distances_);
        cudaFree(query_norms_);
        cudaFree(query_values_);
        cudaFree(database_norms_);
        cudaFree(database_);
        cudaFree(cublas_workspace_);
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
        blended_output_ = nullptr;
        retrieved_output_ = nullptr;
        top_weights_ = nullptr;
        top_indices_ = nullptr;
        top_distances_ = nullptr;
        distances_ = nullptr;
        query_norms_ = nullptr;
        query_values_ = nullptr;
        database_norms_ = nullptr;
        database_ = nullptr;
        cublas_workspace_ = nullptr;
        handle_ = nullptr;
        stream_ = nullptr;
        persistent_bytes_ = 0;
    }

    int rows_ = 0;
    int dimension_ = 0;
    int queries_ = 0;
    int k_ = 0;
    double index_rate_ = 0.0;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t handle_ = nullptr;
    std::uint8_t* cublas_workspace_ = nullptr;
    float* database_ = nullptr;
    float* database_norms_ = nullptr;
    float* query_values_ = nullptr;
    float* query_norms_ = nullptr;
    float* distances_ = nullptr;
    float* top_distances_ = nullptr;
    std::int32_t* top_indices_ = nullptr;
    double* top_weights_ = nullptr;
    float* retrieved_output_ = nullptr;
    float* blended_output_ = nullptr;
    std::uint64_t persistent_bytes_ = 0;
};

class CudaEvent {
public:
    CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    ~CudaEvent() { cudaEventDestroy(event_); }
    operator cudaEvent_t() const noexcept { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

struct CpuReference {
    std::vector<float> top_distances;
    std::vector<std::int32_t> top_indices;
    std::vector<double> top_weights;
    std::vector<float> retrieved_output;
    std::vector<float> blended_output;
    double elapsed_ms = 0.0;
};

CpuReference compute_cpu_reference(const std::vector<float>& database,
                                   const std::vector<float>& queries,
                                   std::uint64_t rows, std::uint32_t dimension,
                                   std::uint32_t query_count, std::uint32_t k,
                                   double index_rate) {
    CpuReference result;
    result.top_distances.resize(static_cast<std::size_t>(query_count) * k);
    result.top_indices.resize(static_cast<std::size_t>(query_count) * k);
    result.top_weights.resize(static_cast<std::size_t>(query_count) * k);
    result.retrieved_output.resize(
        static_cast<std::size_t>(query_count) * dimension);
    result.blended_output.resize(static_cast<std::size_t>(query_count) * dimension);
    std::vector<float> distances(static_cast<std::size_t>(rows));
    std::vector<std::int32_t> indices(static_cast<std::size_t>(rows));
    std::array<double, kMaxK> weights{};

    const auto begin = Clock::now();
    for (std::uint32_t query = 0; query < query_count; ++query) {
        const float* query_values =
            queries.data() + static_cast<std::size_t>(query) * dimension;
        for (std::uint64_t row = 0; row < rows; ++row) {
            float distance = 0.0F;
            const float* vector =
                database.data() + static_cast<std::size_t>(row) * dimension;
            for (std::uint32_t column = 0; column < dimension; ++column) {
                const float difference = query_values[column] - vector[column];
                distance += difference * difference;
            }
            distances[static_cast<std::size_t>(row)] = distance;
        }
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(
            indices.begin(), indices.begin() + k, indices.end(),
            [&distances](std::int32_t left, std::int32_t right) {
                const float left_distance = distances[static_cast<std::size_t>(left)];
                const float right_distance = distances[static_cast<std::size_t>(right)];
                return left_distance < right_distance ||
                       (left_distance == right_distance && left < right);
            });

        double weight_sum = 0.0;
        for (std::uint32_t neighbor = 0; neighbor < k; ++neighbor) {
            const auto row = indices[neighbor];
            const auto distance = distances[static_cast<std::size_t>(row)];
            double weight = 1.0 / (static_cast<double>(distance) + 1e-8);
            weight *= weight;
            weights[neighbor] = weight;
            weight_sum += weight;
            result.top_distances[static_cast<std::size_t>(query) * k + neighbor] =
                distance;
            result.top_indices[static_cast<std::size_t>(query) * k + neighbor] = row;
        }
        if (!std::isfinite(weight_sum) || weight_sum <= 0.0) {
            fail("CPU reference produced invalid retrieval weights");
        }
        for (std::uint32_t neighbor = 0; neighbor < k; ++neighbor) {
            weights[neighbor] /= weight_sum;
            result.top_weights[static_cast<std::size_t>(query) * k + neighbor] =
                weights[neighbor];
        }
        for (std::uint32_t column = 0; column < dimension; ++column) {
            float retrieved = 0.0F;
            for (std::uint32_t neighbor = 0; neighbor < k; ++neighbor) {
                const auto row = result.top_indices[
                    static_cast<std::size_t>(query) * k + neighbor];
                retrieved += static_cast<float>(
                    weights[neighbor] *
                    database[static_cast<std::size_t>(row) * dimension + column]);
            }
            const auto offset = static_cast<std::size_t>(query) * dimension + column;
            result.retrieved_output[offset] = retrieved;
            result.blended_output[offset] = static_cast<float>(
                static_cast<double>(queries[offset]) * (1.0 - index_rate) +
                static_cast<double>(retrieved) * index_rate);
        }
    }
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    return result;
}

double percentile(std::vector<double> samples, double quantile) {
    if (samples.empty()) fail("cannot compute a percentile of zero samples");
    std::sort(samples.begin(), samples.end());
    const double position = quantile * static_cast<double>(samples.size() - 1);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1, samples.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return samples[lower] * (1.0 - fraction) + samples[upper] * fraction;
}

struct LatencyStats {
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double maximum = 0.0;
};

LatencyStats summarize(const std::vector<double>& samples) {
    return {
        percentile(samples, 0.50),
        percentile(samples, 0.95),
        percentile(samples, 0.99),
        *std::max_element(samples.begin(), samples.end()),
    };
}

struct Comparison {
    std::uint64_t positional_matches = 0;
    std::uint64_t set_overlap = 0;
    double maximum_distance_absolute_error = 0.0;
    double maximum_distance_relative_error = 0.0;
    double maximum_weight_absolute_error = 0.0;
    double maximum_weight_relative_error = 0.0;
    double retrieved_cosine = 0.0;
    double maximum_retrieved_absolute_error = 0.0;
    double blended_cosine = 0.0;
    double maximum_blended_absolute_error = 0.0;
    bool distance_within_tolerance = true;
    bool weights_within_tolerance = true;
    bool finite = true;
    bool passed = false;
};

struct VectorComparison {
    double cosine = 0.0;
    double maximum_absolute_error = 0.0;
    bool finite = true;
};

VectorComparison compare_vectors(const std::vector<float>& cpu,
                                 const std::vector<float>& gpu) {
    if (cpu.size() != gpu.size()) fail("CPU/GPU vector sizes do not match");
    VectorComparison result;
    double dot = 0.0;
    double cpu_norm = 0.0;
    double gpu_norm = 0.0;
    for (std::size_t index = 0; index < gpu.size(); ++index) {
        const double cpu_value = cpu[index];
        const double gpu_value = gpu[index];
        if (!std::isfinite(cpu_value) || !std::isfinite(gpu_value)) {
            result.finite = false;
            continue;
        }
        dot += cpu_value * gpu_value;
        cpu_norm += cpu_value * cpu_value;
        gpu_norm += gpu_value * gpu_value;
        result.maximum_absolute_error = std::max(
            result.maximum_absolute_error, std::abs(cpu_value - gpu_value));
    }
    if (!result.finite) {
        result.cosine = 0.0;
    } else if (cpu_norm == 0.0 && gpu_norm == 0.0) {
        result.cosine = 1.0;
    } else if (cpu_norm == 0.0 || gpu_norm == 0.0) {
        result.cosine = 0.0;
    } else {
        result.cosine = dot / std::sqrt(cpu_norm * gpu_norm);
    }
    return result;
}

Comparison compare_results(const CpuReference& cpu,
                           const std::vector<float>& gpu_top_distances,
                           const std::vector<std::int32_t>& gpu_top_indices,
                           const std::vector<double>& gpu_top_weights,
                           const std::vector<float>& gpu_retrieved_output,
                           const std::vector<float>& gpu_blended_output,
                           std::uint32_t query_count, std::uint32_t k) {
    Comparison result;
    for (std::uint32_t query = 0; query < query_count; ++query) {
        for (std::uint32_t neighbor = 0; neighbor < k; ++neighbor) {
            const auto offset = static_cast<std::size_t>(query) * k + neighbor;
            if (cpu.top_indices[offset] == gpu_top_indices[offset]) {
                ++result.positional_matches;
            }
            const auto begin = cpu.top_indices.begin() + static_cast<std::ptrdiff_t>(query * k);
            const auto end = begin + k;
            if (std::find(begin, end, gpu_top_indices[offset]) != end) ++result.set_overlap;
            const double cpu_distance = cpu.top_distances[offset];
            const double gpu_distance = gpu_top_distances[offset];
            const double cpu_weight = cpu.top_weights[offset];
            const double gpu_weight = gpu_top_weights[offset];
            if (!std::isfinite(cpu_distance) || !std::isfinite(gpu_distance) ||
                !std::isfinite(cpu_weight) || !std::isfinite(gpu_weight)) {
                result.finite = false;
                result.distance_within_tolerance = false;
                result.weights_within_tolerance = false;
                continue;
            }

            const double distance_absolute_error =
                std::abs(cpu_distance - gpu_distance);
            const double distance_relative_error = distance_absolute_error /
                std::max(std::abs(cpu_distance), 1e-12);
            result.maximum_distance_absolute_error = std::max(
                result.maximum_distance_absolute_error, distance_absolute_error);
            result.maximum_distance_relative_error = std::max(
                result.maximum_distance_relative_error, distance_relative_error);
            result.distance_within_tolerance =
                result.distance_within_tolerance &&
                (distance_absolute_error <= kDistanceAbsoluteTolerance ||
                 distance_relative_error <= kDistanceRelativeTolerance);

            const double weight_absolute_error = std::abs(cpu_weight - gpu_weight);
            const double weight_relative_error = weight_absolute_error /
                std::max(std::abs(cpu_weight), 1e-12);
            result.maximum_weight_absolute_error = std::max(
                result.maximum_weight_absolute_error, weight_absolute_error);
            result.maximum_weight_relative_error = std::max(
                result.maximum_weight_relative_error, weight_relative_error);
            result.weights_within_tolerance =
                result.weights_within_tolerance &&
                (weight_absolute_error <= kWeightAbsoluteTolerance ||
                 weight_relative_error <= kWeightRelativeTolerance);
        }
    }

    const auto retrieved =
        compare_vectors(cpu.retrieved_output, gpu_retrieved_output);
    const auto blended = compare_vectors(cpu.blended_output, gpu_blended_output);
    result.retrieved_cosine = retrieved.cosine;
    result.maximum_retrieved_absolute_error = retrieved.maximum_absolute_error;
    result.blended_cosine = blended.cosine;
    result.maximum_blended_absolute_error = blended.maximum_absolute_error;
    result.finite = result.finite && retrieved.finite && blended.finite;
    const auto expected = static_cast<std::uint64_t>(query_count) * k;
    result.passed = result.finite && result.positional_matches == expected &&
                    result.set_overlap == expected &&
                    result.distance_within_tolerance &&
                    result.weights_within_tolerance &&
                    result.maximum_retrieved_absolute_error <=
                        kRetrievedAbsoluteTolerance &&
                    result.retrieved_cosine >= kRetrievedCosineMinimum &&
                    result.maximum_blended_absolute_error <=
                        kBlendedAbsoluteTolerance &&
                    result.blended_cosine >= kBlendedCosineMinimum;
    return result;
}

std::string json_escape(std::string_view text) {
    std::ostringstream output;
    for (const unsigned char character : text) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(character) << std::dec;
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

std::string cuda_version_string(int version) {
    return std::to_string(version / 1000) + "." +
           std::to_string((version % 1000) / 10);
}

std::string operating_system() {
    struct utsname information {};
    if (uname(&information) != 0) return "unknown";
    return std::string(information.sysname) + " " + information.release + " " +
           information.machine;
}

void print_json(const Options& options, const MatrixSource& database,
                std::string_view source_kind, std::string_view query_source,
                const std::optional<std::string>& query_sha256,
                const std::optional<std::uint32_t>& query_seed,
                const cudaDeviceProp& device_properties, int runtime_version,
                int driver_version, const LatencyStats& gpu_latency,
                const LatencyStats& wall_latency, double initialization_ms,
                double cpu_reference_ms,
                std::uint64_t persistent_bytes, const Comparison& comparison) {
    const auto expected_matches =
        static_cast<std::uint64_t>(options.queries) * options.k;
    std::cout << std::fixed << std::setprecision(9)
              << "{\"schema\":\"resonantia.realtime_index_benchmark.v1\""
              << ",\"diagnostic_only\":true"
              << ",\"product_acceptance\":false"
              << ",\"backend\":\"persistent_gpu_fp32_sgemm_exact_topk\""
              << ",\"profile\":\"r0_index_exact_weighted_blend\""
              << ",\"environment\":{\"os\":\"" << json_escape(operating_system())
              << "\",\"gpu_device\":" << options.device
              << ",\"gpu_name\":\"" << json_escape(device_properties.name)
              << "\",\"compute_capability\":\"" << device_properties.major << "."
              << device_properties.minor
              << "\",\"cuda_runtime\":\"" << cuda_version_string(runtime_version)
              << "\",\"cuda_driver\":\"" << cuda_version_string(driver_version)
              << "\",\"gpu_global_memory_bytes\":" << device_properties.totalGlobalMem
              << "}"
              << ",\"index\":{\"source\":\"" << source_kind
              << "\",\"rows\":" << database.rows
              << ",\"dimension\":" << database.dimension
              << ",\"k\":" << options.k
              << ",\"index_rate\":" << options.index_rate
              << ",\"sha256\":";
    if (database.sha256.empty()) std::cout << "null";
    else std::cout << "\"" << database.sha256 << "\"";
    std::cout << ",\"synthetic_seed\":";
    if (source_kind == "synthetic") std::cout << options.seed;
    else std::cout << "null";
    std::cout << "}"
              << ",\"queries\":{\"source\":\"" << query_source
              << "\",\"batch\":" << options.queries
              << ",\"sha256\":";
    if (query_sha256.has_value()) std::cout << "\"" << *query_sha256 << "\"";
    else std::cout << "null";
    std::cout << ",\"synthetic_seed\":";
    if (query_seed.has_value()) std::cout << *query_seed;
    else std::cout << "null";
    std::cout << "}"
              << ",\"run\":{\"warmup_calls\":" << options.warmups
              << ",\"repetitions\":" << options.repetitions
              << ",\"runner_owned_device_allocations_per_call\":0"
              << ",\"initialization_ms\":" << initialization_ms << "}"
              << ",\"latency_ms\":{\"gpu\":{\"p50\":" << gpu_latency.p50
              << ",\"p95\":" << gpu_latency.p95
              << ",\"p99\":" << gpu_latency.p99
              << ",\"max\":" << gpu_latency.maximum
              << "},\"wall\":{\"p50\":" << wall_latency.p50
              << ",\"p95\":" << wall_latency.p95
              << ",\"p99\":" << wall_latency.p99
              << ",\"max\":" << wall_latency.maximum
              << "},\"cpu_reference_total\":" << cpu_reference_ms << "}"
              << ",\"memory_bytes\":{\"database\":"
              << database.values.size() * sizeof(float)
              << ",\"cublas_workspace\":" << kCublasWorkspaceBytes
              << ",\"persistent_device_total\":" << persistent_bytes
              << ",\"distance_workspace\":"
              << database.rows * options.queries * sizeof(float) << "}"
              << ",\"validation\":{\"passed\":"
              << (comparison.passed ? "true" : "false")
              << ",\"finite\":" << (comparison.finite ? "true" : "false")
              << ",\"topk_positional_matches\":" << comparison.positional_matches
              << ",\"topk_set_overlap\":" << comparison.set_overlap
              << ",\"topk_expected\":" << expected_matches
              << ",\"index_rate_independent_retrieval\":true"
              << ",\"thresholds\":{\"distance_rule\":\"absolute_or_relative\""
              << ",\"distance_absolute\":" << kDistanceAbsoluteTolerance
              << ",\"distance_relative\":" << kDistanceRelativeTolerance
              << ",\"weight_rule\":\"absolute_or_relative\""
              << ",\"weight_absolute\":" << kWeightAbsoluteTolerance
              << ",\"weight_relative\":" << kWeightRelativeTolerance
              << ",\"retrieved_absolute\":" << kRetrievedAbsoluteTolerance
              << ",\"retrieved_cosine_minimum\":" << kRetrievedCosineMinimum
              << ",\"blended_absolute\":" << kBlendedAbsoluteTolerance
              << ",\"blended_cosine_minimum\":" << kBlendedCosineMinimum << "}"
              << ",\"distance_within_tolerance\":"
              << (comparison.distance_within_tolerance ? "true" : "false")
              << ",\"max_distance_absolute_error\":"
              << comparison.maximum_distance_absolute_error
              << ",\"max_distance_relative_error\":"
              << comparison.maximum_distance_relative_error
              << ",\"weights_within_tolerance\":"
              << (comparison.weights_within_tolerance ? "true" : "false")
              << ",\"max_weight_absolute_error\":"
              << comparison.maximum_weight_absolute_error
              << ",\"max_weight_relative_error\":"
              << comparison.maximum_weight_relative_error
              << ",\"retrieved_cosine\":" << comparison.retrieved_cosine
              << ",\"max_retrieved_absolute_error\":"
              << comparison.maximum_retrieved_absolute_error
              << ",\"weighted_blend_cosine\":" << comparison.blended_cosine
              << ",\"max_weighted_blend_absolute_error\":"
              << comparison.maximum_blended_absolute_error << "}}\n";
}

int run(int argc, char** argv) {
    const auto options = parse_options(argc, argv);

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (options.device < 0 || options.device >= device_count) {
        fail("--device is not available");
    }
    check_cuda(cudaSetDevice(options.device), "cudaSetDevice");
    cudaDeviceProp device_properties{};
    check_cuda(cudaGetDeviceProperties(&device_properties, options.device),
               "cudaGetDeviceProperties");
    int runtime_version = 0;
    int driver_version = 0;
    check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");

    MatrixSource database;
    std::string source_kind;
    if (options.synthetic_rows.has_value()) {
        database = make_synthetic(*options.synthetic_rows, options.dimension, options.seed);
        source_kind = "synthetic";
    } else {
        database = load_flat_index(*options.index_path, std::nullopt);
        source_kind = "flat_index";
    }
    validate_envelope(options, database.rows, database.dimension);

    std::vector<float> query_values;
    std::string query_source;
    std::optional<std::string> query_sha256;
    std::optional<std::uint32_t> query_seed;
    if (options.query_index_path.has_value()) {
        const auto query_index = load_flat_index(*options.query_index_path,
                                                 options.queries);
        if (query_index.dimension != database.dimension ||
            query_index.rows < options.queries) {
            fail("query Index must match dimension and contain --nq rows");
        }
        query_values = query_index.values;
        query_source = "flat_index";
        query_sha256 = query_index.sha256;
    } else if (source_kind == "synthetic") {
        const std::uint32_t derived_seed = options.seed ^ 0x9e3779b9U;
        query_values = make_synthetic_queries(options.queries, database.dimension,
                                              derived_seed);
        query_source = "synthetic";
        query_seed = derived_seed;
    } else {
        if (database.rows < options.queries) {
            fail("database does not contain enough rows for default queries");
        }
        query_values.resize(static_cast<std::size_t>(options.queries) *
                            database.dimension);
        std::copy_n(database.values.data(), query_values.size(), query_values.data());
        for (std::uint32_t query = 0; query < options.queries; ++query) {
            query_values[static_cast<std::size_t>(query) * database.dimension] += 1e-4F;
        }
        query_source = "database_rows_perturbed";
    }
    const double maximum_query_component =
        maximum_l2_safe_component(database.dimension);
    for (const float value : query_values) {
        if (!std::isfinite(value)) fail("query input contains a non-finite value");
        if (std::abs(static_cast<double>(value)) > maximum_query_component) {
            fail("query input contains a value that can overflow FP32 squared L2");
        }
    }

    const auto initialization_begin = Clock::now();
    PersistentGpuExactIndex gpu_index(
        database.values, query_values, database.rows, database.dimension,
        options.queries, options.k, options.index_rate);
    const double initialization_ms =
        std::chrono::duration<double, std::milli>(Clock::now() -
                                                  initialization_begin).count();
    for (std::uint32_t warmup = 0; warmup < options.warmups; ++warmup) {
        gpu_index.run();
    }
    gpu_index.synchronize();

    CudaEvent begin_event;
    CudaEvent end_event;
    std::vector<double> gpu_samples;
    std::vector<double> wall_samples;
    gpu_samples.reserve(options.repetitions);
    wall_samples.reserve(options.repetitions);
    for (std::uint32_t repetition = 0; repetition < options.repetitions;
         ++repetition) {
        const auto wall_begin = Clock::now();
        check_cuda(cudaEventRecord(begin_event, gpu_index.stream()),
                   "record begin event");
        gpu_index.run();
        check_cuda(cudaEventRecord(end_event, gpu_index.stream()),
                   "record end event");
        check_cuda(cudaEventSynchronize(end_event), "wait for timed retrieval");
        const auto wall_end = Clock::now();
        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, begin_event, end_event),
                   "measure timed retrieval");
        gpu_samples.push_back(elapsed_ms);
        wall_samples.push_back(
            std::chrono::duration<double, std::milli>(wall_end - wall_begin).count());
    }

    std::vector<float> gpu_top_distances(
        static_cast<std::size_t>(options.queries) * options.k);
    std::vector<std::int32_t> gpu_top_indices(
        static_cast<std::size_t>(options.queries) * options.k);
    std::vector<double> gpu_top_weights(
        static_cast<std::size_t>(options.queries) * options.k);
    std::vector<float> gpu_retrieved_output(
        static_cast<std::size_t>(options.queries) * database.dimension);
    std::vector<float> gpu_blended_output(
        static_cast<std::size_t>(options.queries) * database.dimension);
    gpu_index.copy_results(gpu_top_distances, gpu_top_indices, gpu_top_weights,
                           gpu_retrieved_output, gpu_blended_output);
    const auto cpu_reference = compute_cpu_reference(
        database.values, query_values, database.rows, database.dimension,
        options.queries, options.k, options.index_rate);
    const auto comparison = compare_results(
        cpu_reference, gpu_top_distances, gpu_top_indices, gpu_top_weights,
        gpu_retrieved_output, gpu_blended_output, options.queries, options.k);

    print_json(options, database, source_kind, query_source, query_sha256,
               query_seed, device_properties, runtime_version, driver_version,
               summarize(gpu_samples), summarize(wall_samples),
               initialization_ms, cpu_reference.elapsed_ms,
               gpu_index.persistent_bytes(), comparison);
    return comparison.passed ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
