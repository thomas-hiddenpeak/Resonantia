#pragma once

#include "voxmutatio/io/safetensors.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace resonantia_test::f0_fixture {

inline bool read_header(const std::filesystem::path& path,
                        std::array<std::uint8_t, 8>& prefix,
                        std::string& header,
                        std::uint64_t& file_size) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return false;
    const std::streamoff end = input.tellg();
    if (end < static_cast<std::streamoff>(prefix.size())) return false;
    file_size = static_cast<std::uint64_t>(end);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(prefix.data()),
               static_cast<std::streamsize>(prefix.size()));
    if (!input) return false;

    std::uint64_t header_size = 0;
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        header_size |= static_cast<std::uint64_t>(prefix[i]) << (8 * i);
    }
    if (header_size == 0 || header_size > file_size - prefix.size() ||
        header_size > static_cast<std::uint64_t>(
                          std::numeric_limits<std::size_t>::max())) {
        return false;
    }
    header.resize(static_cast<std::size_t>(header_size));
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    return static_cast<bool>(input);
}

// Preserve the complete production header and virtual file length while using
// filesystem holes for the payload. The source asset remains read-only.
inline bool create_zero_payload_sparse_clone(const std::filesystem::path& source,
                                             const std::filesystem::path& destination) {
    std::array<std::uint8_t, 8> prefix{};
    std::string header;
    std::uint64_t file_size = 0;
    if (!read_header(source, prefix, header, file_size) ||
        file_size > static_cast<std::uint64_t>(
                        std::numeric_limits<std::streamoff>::max())) {
        return false;
    }

    std::ofstream output(destination, std::ios::binary | std::ios::trunc);
    if (!output) return false;
    output.write(reinterpret_cast<const char*>(prefix.data()),
                 static_cast<std::streamsize>(prefix.size()));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!output) return false;
    if (file_size > prefix.size() + header.size()) {
        output.seekp(static_cast<std::streamoff>(file_size - 1));
        output.put('\0');
    }
    output.close();
    return static_cast<bool>(output) &&
           std::filesystem::file_size(destination) == file_size;
}

inline bool write_float_values(
        const std::filesystem::path& path,
        const std::vector<std::pair<std::size_t, float>>& values) {
    const std::uintmax_t file_size = std::filesystem::file_size(path);
    std::fstream output(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!output) return false;
    for (const auto& [offset, value] : values) {
        if (offset > file_size || sizeof(float) > file_size - offset ||
            offset > static_cast<std::size_t>(
                         std::numeric_limits<std::streamoff>::max())) {
            return false;
        }
        output.seekp(static_cast<std::streamoff>(offset));
        output.write(reinterpret_cast<const char*>(&value), sizeof(value));
        if (!output) return false;
    }
    output.flush();
    return static_cast<bool>(output);
}

inline bool write_tensor_value(const std::filesystem::path& path,
                               std::string_view tensor_name,
                               std::size_t element_index, float value) {
    std::size_t offset = 0;
    {
        voxmutatio::io::SafetensorsLoader loader;
        if (!loader.load(path.string())) return false;
        const auto* tensor = loader.get_tensor(std::string(tensor_name));
        if (!tensor || element_index >= tensor->data_nbytes / sizeof(float) ||
            element_index > (std::numeric_limits<std::size_t>::max() -
                             tensor->data_offset) / sizeof(float)) {
            return false;
        }
        offset = tensor->data_offset + element_index * sizeof(float);
    }
    return write_float_values(path, {{offset, value}});
}

inline bool prepare_fcpe_sparse_payload(const std::filesystem::path& path) {
    std::size_t cent_offset = 0;
    std::size_t projection_offset = 0;
    {
        voxmutatio::io::SafetensorsLoader loader;
        if (!loader.load(path.string())) return false;
        const auto* cent = loader.get_tensor("cent_table");
        const auto* projection = loader.get_tensor(
            "output_proj.parametrizations.weight.original1");
        if (!cent || cent->shape != std::vector<std::int64_t>{360} ||
            !projection ||
            projection->shape != std::vector<std::int64_t>{360, 512}) {
            return false;
        }
        cent_offset = cent->data_offset;
        projection_offset = projection->data_offset;
    }

    std::vector<std::pair<std::size_t, float>> values;
    values.reserve(720);
    for (std::size_t row = 0; row < 360; ++row) {
        values.emplace_back(cent_offset + row * sizeof(float),
                            1997.3794f + 20.0f * static_cast<float>(row));
        values.emplace_back(projection_offset + row * 512 * sizeof(float), 1.0f);
    }
    return write_float_values(path, values);
}

inline bool replace_header_bytes(const std::filesystem::path& path,
                                 std::string_view expected,
                                 std::string_view replacement) {
    if (expected.size() != replacement.size()) return false;
    std::array<std::uint8_t, 8> prefix{};
    std::string header;
    std::uint64_t file_size = 0;
    if (!read_header(path, prefix, header, file_size)) return false;
    const std::size_t position = header.find(expected);
    if (position == std::string::npos ||
        header.find(expected, position + 1) != std::string::npos) {
        return false;
    }
    header.replace(position, expected.size(), replacement);

    std::fstream output(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!output) return false;
    output.seekp(static_cast<std::streamoff>(prefix.size()));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.flush();
    return static_cast<bool>(output);
}

}  // namespace resonantia_test::f0_fixture
