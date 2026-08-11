#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

namespace resonantia_test::vits_fixture {

struct TensorSpec {
    std::string name;
    std::vector<std::int64_t> shape;
    std::size_t data_offset = 0;
    std::size_t element_count = 0;
};

struct SparseCheckpoint {
    std::size_t data_start = 0;
    std::vector<TensorSpec> tensors;

    bool write_value(const std::filesystem::path& path,
                     const std::string& name,
                     std::size_t element_index,
                     float value) const {
        const TensorSpec* selected = nullptr;
        for (const auto& tensor : tensors) {
            if (tensor.name == name) {
                selected = &tensor;
                break;
            }
        }
        if (!selected || element_index >= selected->element_count ||
            selected->data_offset >
                std::numeric_limits<std::size_t>::max() - data_start ||
            element_index >
                (std::numeric_limits<std::size_t>::max() - data_start -
                 selected->data_offset) /
                    sizeof(float)) {
            return false;
        }

        const std::size_t offset =
            data_start + selected->data_offset + element_index * sizeof(float);
        if (offset > static_cast<std::size_t>(
                         std::numeric_limits<std::streamoff>::max())) {
            return false;
        }
        std::fstream output(path, std::ios::binary | std::ios::in | std::ios::out);
        if (!output) return false;
        output.seekp(static_cast<std::streamoff>(offset));
        output.write(reinterpret_cast<const char*>(&value), sizeof(value));
        output.flush();
        return static_cast<bool>(output);
    }
};

inline bool checked_multiply(std::size_t lhs, std::size_t rhs,
                             std::size_t& result) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

inline bool add_tensor(std::vector<TensorSpec>& tensors,
                       std::size_t& payload_size,
                       std::string name,
                       std::initializer_list<std::int64_t> shape) {
    std::size_t elements = 1;
    for (const std::int64_t dimension : shape) {
        if (dimension <= 0 ||
            static_cast<std::uint64_t>(dimension) >
                std::numeric_limits<std::size_t>::max() ||
            !checked_multiply(elements, static_cast<std::size_t>(dimension),
                              elements)) {
            return false;
        }
    }
    std::size_t bytes = 0;
    if (!checked_multiply(elements, sizeof(float), bytes) ||
        bytes > std::numeric_limits<std::size_t>::max() - payload_size) {
        return false;
    }
    tensors.push_back({"model." + std::move(name),
                       std::vector<std::int64_t>(shape), payload_size, elements});
    payload_size += bytes;
    return true;
}

// Construct the complete currently-used VITS tensor manifest, while leaving
// the large payload as filesystem holes. One element per weight-norm row is
// materialized so every row has a finite, non-zero norm.
inline bool create_sparse_checkpoint(const std::filesystem::path& path,
                                     SparseCheckpoint& checkpoint,
                                     int feature_dim = 768) {
    constexpr int kHidden = 192;
    constexpr int kInter = 192;
    constexpr int kHeads = 2;
    constexpr int kLayers = 6;
    constexpr int kFilter = 768;
    constexpr int kWindow = 10;
    constexpr int kGin = 256;

    if (feature_dim != 256 && feature_dim != 768) return false;
    checkpoint = {};
    std::size_t payload_size = 0;
    const auto add = [&](std::string name,
                         std::initializer_list<std::int64_t> shape) {
        return add_tensor(checkpoint.tensors, payload_size, std::move(name), shape);
    };

    if (!add("enc_p.emb_phone.weight", {kHidden, feature_dim}) ||
        !add("enc_p.emb_phone.bias", {kHidden}) ||
        !add("enc_p.emb_pitch.weight", {256, kHidden})) {
        return false;
    }
    for (int layer = 0; layer < kLayers; ++layer) {
        const std::string attention =
            "enc_p.encoder.attn_layers." + std::to_string(layer) + ".";
        if (!add(attention + "emb_rel_k", {1, 2 * kWindow + 1, kHidden / kHeads}) ||
            !add(attention + "emb_rel_v", {1, 2 * kWindow + 1, kHidden / kHeads})) {
            return false;
        }
        for (const char* projection : {"conv_q", "conv_k", "conv_v", "conv_o"}) {
            if (!add(attention + projection + ".weight", {kHidden, kHidden, 1}) ||
                !add(attention + projection + ".bias", {kHidden})) {
                return false;
            }
        }

        const std::string feed_forward =
            "enc_p.encoder.ffn_layers." + std::to_string(layer) + ".";
        if (!add(feed_forward + "conv_1.weight", {kFilter, kHidden, 3}) ||
            !add(feed_forward + "conv_1.bias", {kFilter}) ||
            !add(feed_forward + "conv_2.weight", {kHidden, kFilter, 3}) ||
            !add(feed_forward + "conv_2.bias", {kHidden})) {
            return false;
        }
        for (const char* group : {"norm_layers_1", "norm_layers_2"}) {
            const std::string norm = "enc_p.encoder." + std::string(group) + "." +
                                     std::to_string(layer) + ".";
            if (!add(norm + "gamma", {kHidden}) ||
                !add(norm + "beta", {kHidden})) {
                return false;
            }
        }
    }
    if (!add("enc_p.proj.weight", {2 * kInter, kHidden, 1}) ||
        !add("enc_p.proj.bias", {2 * kInter})) {
        return false;
    }

    for (int coupling = 0; coupling < 4; ++coupling) {
        const std::string flow =
            "flow.flows." + std::to_string(coupling * 2) + ".";
        if (!add(flow + "pre.weight", {kHidden, kInter / 2, 1}) ||
            !add(flow + "pre.bias", {kHidden}) ||
            !add(flow + "post.weight", {kInter / 2, kHidden, 1}) ||
            !add(flow + "post.bias", {kInter / 2})) {
            return false;
        }
        const std::string encoder = flow + "enc.";
        if (!add(encoder + "cond_layer.weight_g", {2 * kHidden * 3, 1, 1}) ||
            !add(encoder + "cond_layer.weight_v", {2 * kHidden * 3, kGin, 1}) ||
            !add(encoder + "cond_layer.bias", {2 * kHidden * 3})) {
            return false;
        }
        for (int layer = 0; layer < 3; ++layer) {
            const std::string input =
                encoder + "in_layers." + std::to_string(layer) + ".";
            if (!add(input + "weight_g", {2 * kHidden, 1, 1}) ||
                !add(input + "weight_v", {2 * kHidden, kHidden, 5}) ||
                !add(input + "bias", {2 * kHidden})) {
                return false;
            }
            const int channels = layer < 2 ? 2 * kHidden : kHidden;
            const std::string residual =
                encoder + "res_skip_layers." + std::to_string(layer) + ".";
            if (!add(residual + "weight_g", {channels, 1, 1}) ||
                !add(residual + "weight_v", {channels, kHidden, 1}) ||
                !add(residual + "bias", {channels})) {
                return false;
            }
        }
    }

    if (!add("dec.m_source.l_linear.weight", {1, 1}) ||
        !add("dec.m_source.l_linear.bias", {1}) ||
        !add("dec.conv_pre.weight", {512, kInter, 7}) ||
        !add("dec.conv_pre.bias", {512}) ||
        !add("dec.cond.weight", {512, kGin, 1}) ||
        !add("dec.cond.bias", {512})) {
        return false;
    }

    constexpr int rates[4] = {10, 10, 2, 2};
    constexpr int kernels[4] = {16, 16, 4, 4};
    constexpr int resblock_kernels[3] = {3, 7, 11};
    int channels = 512;
    for (int stage = 0; stage < 4; ++stage) {
        const int output_channels = channels / 2;
        const std::string upsample = "dec.ups." + std::to_string(stage) + ".";
        if (!add(upsample + "weight_g", {channels, 1, 1}) ||
            !add(upsample + "weight_v",
                 {channels, output_channels, kernels[stage]}) ||
            !add(upsample + "bias", {output_channels})) {
            return false;
        }

        int source_kernel = 1;
        if (stage < 3) {
            int remaining_rate = 1;
            for (int next = stage + 1; next < 4; ++next) {
                remaining_rate *= rates[next];
            }
            source_kernel = remaining_rate * 2;
        }
        const std::string noise =
            "dec.noise_convs." + std::to_string(stage) + ".";
        if (!add(noise + "weight", {output_channels, 1, source_kernel}) ||
            !add(noise + "bias", {output_channels})) {
            return false;
        }

        for (int block = 0; block < 3; ++block) {
            const int block_index = stage * 3 + block;
            const int kernel = resblock_kernels[block];
            const std::string residual =
                "dec.resblocks." + std::to_string(block_index) + ".";
            for (int layer = 0; layer < 3; ++layer) {
                for (const char* group : {"convs1", "convs2"}) {
                    const std::string convolution =
                        residual + group + "." + std::to_string(layer) + ".";
                    if (!add(convolution + "weight_g", {output_channels, 1, 1}) ||
                        !add(convolution + "weight_v",
                             {output_channels, output_channels, kernel}) ||
                        !add(convolution + "bias", {output_channels})) {
                        return false;
                    }
                }
            }
        }
        channels = output_channels;
    }
    if (!add("dec.conv_post.weight", {1, channels, 7}) ||
        !add("emb_g.weight", {1, kGin})) {
        return false;
    }

    std::string header = "{";
    for (std::size_t i = 0; i < checkpoint.tensors.size(); ++i) {
        const auto& tensor = checkpoint.tensors[i];
        if (i != 0) header.push_back(',');
        header += "\"" + tensor.name + "\":{\"dtype\":\"F32\",\"shape\":[";
        for (std::size_t dimension = 0; dimension < tensor.shape.size(); ++dimension) {
            if (dimension != 0) header.push_back(',');
            header += std::to_string(tensor.shape[dimension]);
        }
        const std::size_t bytes = tensor.element_count * sizeof(float);
        header += "],\"data_offsets\":[" + std::to_string(tensor.data_offset) + "," +
                  std::to_string(tensor.data_offset + bytes) + "]}";
    }
    header.push_back('}');
    while ((sizeof(std::uint64_t) + header.size()) % alignof(float) != 0) {
        header.push_back(' ');
    }
    checkpoint.data_start = sizeof(std::uint64_t) + header.size();

    if (payload_size == 0 || checkpoint.data_start >
                                 std::numeric_limits<std::size_t>::max() - payload_size) {
        return false;
    }
    const std::size_t file_size = checkpoint.data_start + payload_size;
    if (file_size - 1 > static_cast<std::size_t>(
                            std::numeric_limits<std::streamoff>::max())) {
        return false;
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) return false;
    const std::uint64_t header_size = header.size();
    std::array<std::uint8_t, sizeof(header_size)> prefix{};
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        prefix[i] = static_cast<std::uint8_t>(header_size >> (8 * i));
    }
    output.write(reinterpret_cast<const char*>(prefix.data()),
                 static_cast<std::streamsize>(prefix.size()));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.seekp(static_cast<std::streamoff>(file_size - 1));
    output.put('\0');
    output.close();
    if (!output || std::filesystem::file_size(path) != file_size) return false;

    // Make each weight_v row non-degenerate without materializing its zero data.
    std::fstream values(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!values) return false;
    constexpr float kOne = 1.0f;
    for (const auto& tensor : checkpoint.tensors) {
        if (!tensor.name.ends_with(".weight_v") || tensor.shape.empty()) continue;
        const std::size_t rows = static_cast<std::size_t>(tensor.shape.front());
        const std::size_t inner = tensor.element_count / rows;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t offset = checkpoint.data_start + tensor.data_offset +
                                       row * inner * sizeof(float);
            values.seekp(static_cast<std::streamoff>(offset));
            values.write(reinterpret_cast<const char*>(&kOne), sizeof(kOne));
            if (!values) return false;
        }
    }
    values.flush();
    return static_cast<bool>(values);
}

}  // namespace resonantia_test::vits_fixture
