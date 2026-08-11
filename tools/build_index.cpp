/**
 * @file build_index.cpp
 * @brief Build a feature retrieval index from a directory of audio files.
 *
 * Extracts HuBERT features (v2, 768-dim) from each audio file and writes a
 * flat index binary: [int64 ntotal][int64 dim][float32 data].
 * Pure C++/CUDA — no Python at runtime.
 *
 * Usage:
 *   build_index --hubert <path> --input-dir <dir> --output <index.bin>
 */
#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "voxmutatio/content/hubert_encoder.h"
#include "voxmutatio/io/audio_io.h"

namespace fs = std::filesystem;

namespace {

static_assert(std::endian::native == std::endian::little,
              "the memory-mapped Flat Index payload requires a little-endian host");

void write_u64_le(std::uint8_t* output, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        output[i] = static_cast<std::uint8_t>((value >> (8 * i)) & 0xffu);
    }
}

bool write_all(int fd, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::size_t offset = 0;
    while (offset < size) {
        const std::size_t request = std::min<std::size_t>(size - offset, 1U << 20);
        const ssize_t count = ::write(fd, bytes + offset, request);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        offset += static_cast<std::size_t>(count);
    }
    return true;
}

class StagedIndexFile {
public:
    StagedIndexFile() = default;
    ~StagedIndexFile() {
        if (fd_ >= 0) ::close(fd_);
        if (!path_.empty()) ::unlink(path_.c_str());
    }
    StagedIndexFile(const StagedIndexFile&) = delete;
    StagedIndexFile& operator=(const StagedIndexFile&) = delete;

    bool create(const fs::path& target) {
        std::error_code absolute_error;
        target_ = fs::absolute(target, absolute_error).lexically_normal();
        if (absolute_error || target_.filename().empty()) return false;
        const fs::path parent = target_.parent_path();
        std::error_code directory_error;
        if (!fs::is_directory(parent, directory_error) || directory_error) return false;

        for (unsigned attempt = 0; attempt < 128; ++attempt) {
            path_ = parent /
                    ("." + target_.filename().string() + ".stage." +
                     std::to_string(static_cast<long long>(::getpid())) + "." +
                     std::to_string(attempt));
            fd_ = ::open(path_.c_str(),
                         O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
            if (fd_ >= 0) return true;
            if (errno != EEXIST) break;
        }
        path_.clear();
        return false;
    }

    [[nodiscard]] int fd() const noexcept { return fd_; }

    bool publish() {
        if (fd_ < 0 || path_.empty() || target_.empty()) return false;
        if (::fsync(fd_) != 0) return false;
        const int descriptor = std::exchange(fd_, -1);
        if (::close(descriptor) != 0) return false;
        if (::rename(path_.c_str(), target_.c_str()) != 0) return false;
        path_.clear();

        const int directory = ::open(target_.parent_path().c_str(),
                                     O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (directory >= 0) {
            (void)::fsync(directory);
            ::close(directory);
        }
        return true;
    }

private:
    int fd_ = -1;
    fs::path path_;
    fs::path target_;
};

bool write_index_atomically(const fs::path& target, std::uint64_t rows,
                            std::uint64_t dimensions,
                            const std::vector<float>& features) {
    if (rows == 0 || dimensions == 0 ||
        rows > std::numeric_limits<std::uint64_t>::max() / dimensions ||
        rows * dimensions != features.size()) {
        return false;
    }
    std::size_t payload_bytes = 0;
    if (features.size() >
        std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return false;
    }
    payload_bytes = features.size() * sizeof(float);

    StagedIndexFile output;
    if (!output.create(target)) return false;
    std::array<std::uint8_t, 16> header {};
    write_u64_le(header.data(), rows);
    write_u64_le(header.data() + 8, dimensions);
    if (!write_all(output.fd(), header.data(), header.size()) ||
        !write_all(output.fd(), features.data(), payload_bytes)) {
        return false;
    }
    return output.publish();
}

void print_usage() {
    std::cout << "Usage:\n"
              << "  build_index --hubert <path> --input-dir <dir> --output <index.bin>\n\n"
              << "Extracts HuBERT v2 features from all .wav files in the\n"
              << "directory and writes a flat feature index.\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string hubert_path, input_dir, output_path;

    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--help" || argument == "-h") {
            print_usage();
            return 0;
        }
    }

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        const auto value = [&](std::string& destination) {
            if (i + 1 >= argc) {
                std::cerr << "error: missing value for " << a << "\n";
                return false;
            }
            destination = argv[++i];
            return true;
        };
        if (a == "--hubert") {
            if (!value(hubert_path)) return 1;
        } else if (a == "--input-dir") {
            if (!value(input_dir)) return 1;
        } else if (a == "--output") {
            if (!value(output_path)) return 1;
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return 1;
        }
    }

    if (hubert_path.empty() || input_dir.empty() || output_path.empty()) {
        print_usage();
        return 1;
    }
    std::error_code input_error;
    if (!fs::is_directory(input_dir, input_error) || input_error) {
        std::cerr << "Input directory is unavailable: " << input_dir << "\n";
        return 1;
    }

    voxmutatio::content::HubertConfig cfg;
    cfg.model_path = hubert_path;
    cfg.output_dim = 768;
    cfg.use_final_proj = false;
    cfg.num_layers = 12;

    voxmutatio::content::HubertEncoder encoder;
    if (!encoder.init(cfg)) {
        std::cerr << "Failed to init HuBERT\n";
        return 1;
    }

    std::vector<float> all_feats;
    const int dim = 768;
    int64_t ntotal = 0;
    int file_count = 0;

    std::vector<fs::path> audio_paths;
    std::error_code iteration_error;
    fs::directory_iterator iterator(input_dir, iteration_error), end;
    for (; !iteration_error && iterator != end; iterator.increment(iteration_error)) {
        const auto& entry = *iterator;
        std::error_code type_error;
        if (!entry.is_regular_file(type_error) || type_error) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (ext != ".wav") continue;

        audio_paths.push_back(entry.path());
    }
    if (iteration_error) {
        std::cerr << "Failed while scanning input directory: "
                  << iteration_error.message() << "\n";
        return 1;
    }
    std::sort(audio_paths.begin(), audio_paths.end(),
              [](const fs::path& left, const fs::path& right) {
                  return left.generic_string() < right.generic_string();
              });

    for (const auto& audio_path : audio_paths) {
        auto audio = voxmutatio::io::read_audio(audio_path.string(), 16000);
        if (!audio.has_value() || audio->data.empty()) {
            std::cerr << "  skip (read failed): " << audio_path << "\n";
            continue;
        }
        if (audio->data.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            std::cerr << "  skip (too many samples): " << audio_path << "\n";
            continue;
        }

        auto feats = encoder.extract(audio->data.data(),
                                     static_cast<int>(audio->data.size()));
        if (feats.empty()) continue;
        if (feats.size() % static_cast<std::size_t>(dim) != 0 ||
            feats.size() / static_cast<std::size_t>(dim) >
                static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            std::cerr << "Invalid HuBERT feature shape for: " << audio_path << "\n";
            return 1;
        }
        const int frames =
            static_cast<int>(feats.size() / static_cast<std::size_t>(dim));
        if (frames <= 0 || ntotal > std::numeric_limits<int64_t>::max() - frames) {
            std::cerr << "Index row count overflow\n";
            return 1;
        }

        all_feats.insert(all_feats.end(), feats.begin(), feats.end());
        ntotal += frames;
        ++file_count;
        std::cout << "  " << audio_path.filename().string() << ": "
                  << frames << " frames\n";
    }

    if (ntotal == 0) {
        std::cerr << "No features extracted\n";
        return 1;
    }

    if (!write_index_atomically(output_path,
                                static_cast<std::uint64_t>(ntotal),
                                static_cast<std::uint64_t>(dim), all_feats)) {
        std::cerr << "Failed while publishing index: " << output_path << "\n";
        return 1;
    }

    std::cout << "Wrote index: " << output_path << " (" << file_count
              << " files, " << ntotal << " vectors x " << dim << ")\n";
    return 0;
}
