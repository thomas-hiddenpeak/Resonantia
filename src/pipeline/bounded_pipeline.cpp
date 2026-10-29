// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/pipeline/pipeline.h"

#include "../io/bounded_audio_io.h"
#include "bounded_pipeline_detail.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace voxmutatio::pipeline {

namespace {

namespace bounded_io = voxmutatio::io::detail;

constexpr std::uint64_t kInputSampleRate = 16'000;
constexpr std::uint64_t kOutputSampleRate = 40'000;
constexpr std::uint64_t kPitchHop = detail::kPitchHop;
constexpr std::uint64_t kContentMaximumSamples = 40 * kInputSampleRate;
constexpr std::size_t kIoBlockSamples = 8192;
constexpr std::size_t kNativeCacheSamples = 8192;
constexpr int kFilterOrder = 5;

constexpr std::array<double, 6> kHighpassB = {
    9.6996064518e-01, -4.8498032259e+00, 9.6996064518e+00,
    -9.6996064518e+00, 4.8498032259e+00, -9.6996064518e-01,
};
constexpr std::array<double, 6> kHighpassA = {
    1.0, -4.9390018192e+00, 9.7578635267e+00,
    -9.6395448494e+00, 4.7615067974e+00, -9.4082365321e-01,
};

bool checked_add(std::uint64_t left, std::uint64_t right,
                 std::uint64_t& output) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    output = left + right;
    return true;
}

bool checked_mul(std::uint64_t left, std::uint64_t right,
                 std::uint64_t& output) noexcept {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    output = left * right;
    return true;
}

bool narrow_size(std::uint64_t value, std::size_t& output) noexcept {
    if (value > std::numeric_limits<std::size_t>::max()) return false;
    output = static_cast<std::size_t>(value);
    return true;
}

bool narrow_int(std::uint64_t value, int& output) noexcept {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    output = static_cast<int>(value);
    return true;
}

bool seconds_to_samples(double seconds, std::uint64_t& output) noexcept {
    if (!std::isfinite(seconds) || seconds < 0.0) return false;
    const long double samples =
        static_cast<long double>(seconds) * static_cast<long double>(kInputSampleRate);
    if (samples < 0.0L ||
        samples > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        return false;
    }
    output = static_cast<std::uint64_t>(samples);
    return true;
}

bool fail(std::string& error, const char* message) {
    error = message;
    return false;
}

template <typename Sample>
bool append_repeated(bounded_io::RawSampleSpool<Sample>& spool, Sample value,
                     std::uint64_t count, std::string& error) {
    std::vector<Sample> block(kIoBlockSamples, value);
    while (count > 0) {
        const std::size_t take = static_cast<std::size_t>(
            std::min<std::uint64_t>(count, block.size()));
        if (!spool.append(std::span<const Sample>(block.data(), take))) {
            return fail(error, "bounded spool append failed");
        }
        count -= take;
    }
    return true;
}

template <typename Sample>
bool copy_spool(bounded_io::RawSampleSpool<Sample>& source,
                bounded_io::RawSampleSpool<Sample>& destination,
                std::string& error) {
    std::vector<Sample> block;
    for (std::uint64_t offset = 0; offset < source.sample_count();) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(kIoBlockSamples,
                                    source.sample_count() - offset));
        if (!source.read(offset, count, block) || !destination.append(block)) {
            return fail(error, "bounded spool copy failed");
        }
        offset += count;
    }
    return destination.sample_count() == source.sample_count() ||
           fail(error, "bounded spool copy length mismatch");
}

bool stable_sample_count(const bounded_io::WavMetadata& metadata,
                         std::uint64_t& output) noexcept {
    if (metadata.sample_rate == 0 || metadata.frame_count == 0) return false;
    std::uint64_t scaled = 0;
    if (!checked_mul(metadata.frame_count, kInputSampleRate, scaled)) return false;
    output = scaled / metadata.sample_rate;
    return output > 0;
}

bool build_stable_16k_snapshot(bounded_io::WavChunkReader& reader,
                               const std::filesystem::path& spool_directory,
                               bounded_io::RawFloatSpool& snapshot,
                               std::uint64_t expected_samples,
                               std::string& error) {
    if (!snapshot.create(spool_directory)) {
        return fail(error, "failed to create stable 16 kHz spool");
    }
    const auto metadata = reader.metadata();
    if (metadata.sample_rate == 0 || metadata.frame_count == 0) {
        return fail(error, "invalid bounded WAV metadata");
    }

    std::vector<float> native_cache;
    std::uint64_t cache_first = 0;
    std::uint64_t cache_end = 0;
    const auto sample_at = [&](std::uint64_t index, float& value) -> bool {
        if (index >= metadata.frame_count) return false;
        if (index < cache_first || index >= cache_end) {
            const std::size_t count = static_cast<std::size_t>(
                std::min<std::uint64_t>(kNativeCacheSamples,
                                        metadata.frame_count - index));
            if (!reader.read_mono_frames(index, count, native_cache)) return false;
            cache_first = index;
            cache_end = index + count;
        }
        value = native_cache[static_cast<std::size_t>(index - cache_first)];
        return std::isfinite(value);
    };

    std::vector<float> output;
    output.resize(kIoBlockSamples);
    for (std::uint64_t base = 0; base < expected_samples;) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(output.size(), expected_samples - base));
        for (std::size_t local = 0; local < count; ++local) {
            const std::uint64_t output_index = base + local;
            const long double position =
                static_cast<long double>(output_index) * metadata.sample_rate /
                static_cast<long double>(kInputSampleRate);
            std::uint64_t index = static_cast<std::uint64_t>(position);
            if (index >= metadata.frame_count) index = metadata.frame_count - 1;
            const long double fraction = position - static_cast<long double>(index);
            float first = 0.0f;
            if (!sample_at(index, first)) {
                return fail(error, "bounded WAV decode failed");
            }
            float value = first;
            if (index + 1 < metadata.frame_count) {
                float second = 0.0f;
                if (!sample_at(index + 1, second)) {
                    return fail(error, "bounded WAV decode failed");
                }
                value = static_cast<float>(
                    static_cast<long double>(first) * (1.0L - fraction) +
                    static_cast<long double>(second) * fraction);
            }
            if (!std::isfinite(value)) {
                return fail(error, "bounded resampling produced non-finite audio");
            }
            output[local] = value;
        }
        if (!snapshot.append(std::span<const float>(output.data(), count))) {
            return fail(error, "failed to append stable 16 kHz audio");
        }
        base += count;
    }
    if (!reader.verify_unchanged()) {
        return fail(error, "input WAV changed during bounded snapshot");
    }
    reader.close();
    return snapshot.sample_count() == expected_samples ||
           fail(error, "stable 16 kHz snapshot length mismatch");
}

class IirState {
public:
    double process(double input) noexcept {
        double output = kHighpassB[0] * input;
        for (int order = 1; order <= kFilterOrder; ++order) {
            output += kHighpassB[order] * input_history_[order - 1] -
                      kHighpassA[order] * output_history_[order - 1];
        }
        for (int index = kFilterOrder - 1; index > 0; --index) {
            input_history_[index] = input_history_[index - 1];
            output_history_[index] = output_history_[index - 1];
        }
        input_history_[0] = input;
        output_history_[0] = output;
        return output;
    }

private:
    std::array<double, kFilterOrder> input_history_{};
    std::array<double, kFilterOrder> output_history_{};
};

bool build_filtered_source(bounded_io::RawFloatSpool& source,
                           const std::filesystem::path& spool_directory,
                           bool apply_highpass,
                           bounded_io::RawFloatSpool& filtered,
                           std::string& error) {
    const std::uint64_t samples = source.sample_count();
    if (!apply_highpass || samples < 40) {
        filtered = std::move(source);
        return filtered.is_open();
    }
    if (!filtered.create(spool_directory)) {
        return fail(error, "failed to create filtered-audio spool");
    }

    constexpr std::uint64_t pad = 18;
    std::uint64_t extended_samples = 0;
    if (!checked_add(samples, 2 * pad, extended_samples)) {
        return fail(error, "high-pass padded length overflow");
    }
    bounded_io::RawDoubleSpool forward;
    bounded_io::RawDoubleSpool backward;
    if (!forward.create(spool_directory) || !backward.create(spool_directory)) {
        return fail(error, "failed to create high-pass double spool");
    }

    std::vector<float> first;
    std::vector<float> last;
    if (!source.read(0, static_cast<std::size_t>(pad + 1), first) ||
        !source.read(samples - (pad + 1), static_cast<std::size_t>(pad + 1), last)) {
        return fail(error, "failed to read high-pass reflection samples");
    }

    IirState forward_state;
    std::vector<double> forward_block;
    forward_block.reserve(kIoBlockSamples);
    const auto push_forward = [&](double input) -> bool {
        const double output = forward_state.process(input);
        if (!std::isfinite(output)) return false;
        forward_block.push_back(output);
        if (forward_block.size() == kIoBlockSamples) {
            if (!forward.append(forward_block)) return false;
            forward_block.clear();
        }
        return true;
    };

    for (std::uint64_t index = 0; index < pad; ++index) {
        if (!push_forward(2.0 * first[0] - first[pad - index])) {
            return fail(error, "high-pass forward pass failed");
        }
    }
    std::vector<float> source_block;
    for (std::uint64_t offset = 0; offset < samples;) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(kIoBlockSamples, samples - offset));
        if (!source.read(offset, count, source_block)) {
            return fail(error, "high-pass source read failed");
        }
        for (const float value : source_block) {
            if (!push_forward(value)) {
                return fail(error, "high-pass forward pass failed");
            }
        }
        offset += count;
    }
    for (std::uint64_t index = 0; index < pad; ++index) {
        const double value =
            2.0 * last[pad] - last[pad - 1 - static_cast<std::size_t>(index)];
        if (!push_forward(value)) {
            return fail(error, "high-pass forward pass failed");
        }
    }
    if (!forward_block.empty() && !forward.append(forward_block)) {
        return fail(error, "high-pass forward spool append failed");
    }
    if (forward.sample_count() != extended_samples) {
        return fail(error, "high-pass forward length mismatch");
    }

    IirState backward_state;
    std::vector<double> input_block;
    std::vector<double> backward_block;
    backward_block.reserve(kIoBlockSamples);
    for (std::uint64_t remaining = extended_samples; remaining > 0;) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(kIoBlockSamples, remaining));
        const std::uint64_t first_index = remaining - count;
        if (!forward.read(first_index, count, input_block)) {
            return fail(error, "high-pass reverse read failed");
        }
        for (auto iterator = input_block.rbegin(); iterator != input_block.rend(); ++iterator) {
            const double value = backward_state.process(*iterator);
            if (!std::isfinite(value)) {
                return fail(error, "high-pass reverse pass produced non-finite audio");
            }
            backward_block.push_back(value);
            if (backward_block.size() == kIoBlockSamples) {
                if (!backward.append(backward_block)) {
                    return fail(error, "high-pass reverse spool append failed");
                }
                backward_block.clear();
            }
        }
        remaining -= count;
    }
    if (!backward_block.empty() && !backward.append(backward_block)) {
        return fail(error, "high-pass reverse spool append failed");
    }
    if (backward.sample_count() != extended_samples) {
        return fail(error, "high-pass reverse length mismatch");
    }

    std::vector<double> reversed;
    std::vector<float> converted;
    converted.resize(kIoBlockSamples);
    for (std::uint64_t output_offset = 0; output_offset < samples;) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(converted.size(), samples - output_offset));
        const std::uint64_t highest = samples + pad - 1 - output_offset;
        const std::uint64_t lowest = highest - (count - 1);
        if (!backward.read(lowest, count, reversed)) {
            return fail(error, "high-pass output read failed");
        }
        for (std::size_t index = 0; index < count; ++index) {
            const float value = static_cast<float>(reversed[count - 1 - index]);
            if (!std::isfinite(value)) {
                return fail(error, "high-pass output is not finite");
            }
            converted[index] = value;
        }
        if (!filtered.append(std::span<const float>(converted.data(), count))) {
            return fail(error, "filtered-audio spool append failed");
        }
        output_offset += count;
    }
    return filtered.sample_count() == samples ||
           fail(error, "filtered-audio length mismatch");
}

bool append_reversed_source(bounded_io::RawFloatSpool& source,
                            std::uint64_t first_source, std::size_t count,
                            std::vector<float>& output, std::string& error) {
    std::vector<float> block;
    if (!source.read(first_source, count, block)) {
        return fail(error, "reflected-audio spool read failed");
    }
    output.insert(output.end(), block.rbegin(), block.rend());
    return true;
}

bool read_reflected(bounded_io::RawFloatSpool& source,
                    std::uint64_t source_samples, std::uint64_t pad,
                    std::uint64_t first_virtual, std::size_t count,
                    std::vector<float>& output, std::string& error) {
    output.clear();
    std::uint64_t virtual_samples = 0;
    std::uint64_t end_virtual = 0;
    if (source_samples < 2 || pad >= source_samples ||
        !checked_add(source_samples, 2 * pad, virtual_samples) ||
        !checked_add(first_virtual, count, end_virtual) ||
        end_virtual > virtual_samples) {
        return fail(error, "invalid reflected-audio range");
    }
    output.reserve(count);
    std::uint64_t cursor = first_virtual;
    const std::uint64_t middle_end = pad + source_samples;
    while (cursor < end_virtual) {
        if (cursor < pad) {
            const std::uint64_t region_end = std::min(end_virtual, pad);
            const std::uint64_t region_count = region_end - cursor;
            std::size_t region_size = 0;
            if (!narrow_size(region_count, region_size)) {
                return fail(error, "reflected-audio range exceeds host size");
            }
            const std::uint64_t lowest = pad - (region_end - 1);
            if (!append_reversed_source(source, lowest, region_size, output, error)) {
                return false;
            }
            cursor = region_end;
        } else if (cursor < middle_end) {
            const std::uint64_t region_end = std::min(end_virtual, middle_end);
            const std::uint64_t region_count = region_end - cursor;
            std::size_t region_size = 0;
            if (!narrow_size(region_count, region_size)) {
                return fail(error, "reflected-audio range exceeds host size");
            }
            std::vector<float> block;
            if (!source.read(cursor - pad, region_size, block)) {
                return fail(error, "reflected-audio source read failed");
            }
            output.insert(output.end(), block.begin(), block.end());
            cursor = region_end;
        } else {
            const std::uint64_t region_end = end_virtual;
            const std::uint64_t region_count = region_end - cursor;
            std::size_t region_size = 0;
            if (!narrow_size(region_count, region_size)) {
                return fail(error, "reflected-audio range exceeds host size");
            }
            const std::uint64_t right_offset = cursor - middle_end;
            const std::uint64_t lowest =
                source_samples - 2 - (right_offset + region_count - 1);
            if (!append_reversed_source(source, lowest, region_size, output, error)) {
                return false;
            }
            cursor = region_end;
        }
    }
    return output.size() == count || fail(error, "reflected-audio length mismatch");
}

class QuietCorePlanner {
public:
    QuietCorePlanner(bounded_io::RawFloatSpool& source, std::uint64_t samples,
                     std::uint64_t chunk, std::uint64_t query,
                     std::uint64_t hard_core)
        : source_(source), samples_(samples), chunk_(chunk), query_(query),
          hard_core_(hard_core), target_(chunk) {}

    bool next(std::uint64_t& start, std::uint64_t& end, bool& done,
              std::string& error) {
        done = false;
        if (previous_ >= samples_) {
            done = true;
            return true;
        }
        while (target_ < samples_) {
            const std::uint64_t energy_frames = samples_ / kPitchHop;
            if (energy_frames == 0) {
                return fail(error, "audio is too short for quiet planning");
            }
            const std::uint64_t low = target_ > query_
                ? std::max<std::uint64_t>(1, target_ - query_) : 1;
            const std::uint64_t high = std::min(samples_ - 1, target_ + query_);
            const std::uint64_t low_frame = low / kPitchHop;
            const std::uint64_t high_frame =
                std::min(energy_frames - 1, high / kPitchHop);

            std::uint64_t next_target = 0;
            if (!checked_add(target_, chunk_, next_target)) {
                return fail(error, "quiet-planner target overflow");
            }
            target_ = next_target;
            if (low_frame > high_frame) continue;

            const std::uint64_t first_sample = low_frame * kPitchHop;
            const std::uint64_t frame_count = high_frame - low_frame + 1;
            const std::uint64_t query_samples = frame_count * kPitchHop;
            std::size_t query_size = 0;
            if (!narrow_size(query_samples, query_size)) {
                return fail(error, "quiet-planner query exceeds host size");
            }
            std::vector<float> audio;
            if (!source_.read(first_sample, query_size, audio)) {
                return fail(error, "quiet-planner source read failed");
            }

            std::uint64_t best_frame = low_frame;
            float best_energy = std::numeric_limits<float>::max();
            for (std::uint64_t frame = 0; frame < frame_count; ++frame) {
                double accumulated_energy = 0.0;
                const std::size_t base = static_cast<std::size_t>(frame * kPitchHop);
                for (std::size_t sample = 0; sample < kPitchHop; ++sample) {
                    accumulated_energy += std::fabs(audio[base + sample]);
                }
                // Preserve legacy convert_long semantics: the double sum was
                // stored in a float energy vector before strict comparison.
                const float energy = static_cast<float>(accumulated_energy);
                if (energy < best_energy) {
                    best_energy = energy;
                    best_frame = low_frame + frame;
                }
            }
            const std::uint64_t candidate = best_frame * kPitchHop;
            if (candidate <= previous_ || candidate >= samples_) continue;
            if (candidate - previous_ > hard_core_) {
                return fail(error, "quiet planner exceeded the core hard bound");
            }
            start = previous_;
            end = candidate;
            previous_ = candidate;
            return true;
        }

        if (samples_ - previous_ > hard_core_) {
            return fail(error, "quiet planner final core exceeded the hard bound");
        }
        start = previous_;
        end = samples_;
        previous_ = samples_;
        return true;
    }

private:
    bounded_io::RawFloatSpool& source_;
    std::uint64_t samples_ = 0;
    std::uint64_t chunk_ = 0;
    std::uint64_t query_ = 0;
    std::uint64_t hard_core_ = 0;
    std::uint64_t previous_ = 0;
    std::uint64_t target_ = 0;
};

template <typename Infer>
bool build_raw_f0(bounded_io::RawFloatSpool& filtered,
                  std::uint64_t source_samples, std::uint64_t pad,
                  std::uint64_t context_samples,
                  const std::filesystem::path& spool_directory,
                  Infer&& infer, bounded_io::RawFloatSpool& raw_f0,
                  std::uint64_t& maximum_input_samples,
                  std::string& error,const ProgressObserver& observer,io::BlockCheckpoint* checkpoint) {
    if (!raw_f0.create(spool_directory)) {
        return fail(error, "failed to create raw F0 spool");
    }
    std::uint64_t virtual_samples = 0;
    if (!checked_add(source_samples, 2 * pad, virtual_samples)) {
        return fail(error, "F0 virtual input length overflow");
    }
    const std::uint64_t total_frames = virtual_samples / kPitchHop + 1;
    report_progress(observer,"f0",0,total_frames,"frames100hz");

    for (std::uint64_t first_frame = 0; first_frame < total_frames;) {
        detail::F0WindowPlan plan;
        if (!detail::plan_f0_window(
                virtual_samples, first_frame, context_samples, plan)) {
            return fail(error, "F0 window is not aligned to the global lattice");
        }
        const std::uint64_t input_samples = plan.input_samples;
        maximum_input_samples = std::max(maximum_input_samples, input_samples);
        const std::uint64_t hard_input =
            detail::kF0CoreSamples + 2 * context_samples;
        if (input_samples > hard_input) {
            return fail(error, "F0 window exceeded the profile hard bound");
        }
        std::size_t input_size = 0;
        int input_count = 0;
        if (!narrow_size(input_samples, input_size) ||
            !narrow_int(input_samples, input_count)) {
            return fail(error, "F0 window exceeds the extractor range");
        }
        const auto block_key="f0_"+std::to_string(first_frame);
        if(checkpoint) {
            checkpoint->boundary();
            if(auto cached=checkpoint->load(block_key,first_frame,plan.crop_frame_count)) {
                if(!raw_f0.append(*cached))return fail(error,"cached F0 spool append failed");
                first_frame=plan.end_frame;
                report_progress(observer,"f0",first_frame,total_frames,"frames100hz");
                checkpoint->boundary();
                continue;
            }
        }
        std::vector<float> window;
        if (!read_reflected(filtered, source_samples, pad, plan.input_start,
                            input_size, window, error)) {
            return false;
        }
        std::vector<float> local_f0 = infer(window.data(), input_count);
        if (local_f0.size() != plan.expected_local_frames) {
            return fail(error, "bounded F0 extraction returned an unexpected frame count");
        }
        if (plan.crop_first_frame > local_f0.size() ||
            plan.crop_frame_count > local_f0.size() - plan.crop_first_frame) {
            return fail(error, "bounded F0 crop exceeded its window");
        }
        const auto* crop_data = local_f0.data() +
            static_cast<std::size_t>(plan.crop_first_frame);
        if (!raw_f0.append(std::span<const float>(
                crop_data, static_cast<std::size_t>(plan.crop_frame_count)))) {
            return fail(error, "raw F0 spool append failed");
        }
        if(checkpoint)checkpoint->save(block_key,first_frame,std::span<const float>(
            crop_data,static_cast<std::size_t>(plan.crop_frame_count)));
        first_frame = plan.end_frame;
        report_progress(observer,"f0",first_frame,total_frames,"frames100hz");
        if(checkpoint)checkpoint->boundary();
    }
    return raw_f0.sample_count() == total_frames ||
           fail(error, "raw F0 spool length mismatch");
}

bool find_first_voiced(bounded_io::RawFloatSpool& raw,
                       std::uint64_t& index, float& value,
                       bool& found, std::string& error) {
    found = false;
    std::vector<float> block;
    for (std::uint64_t offset = 0; offset < raw.sample_count();) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(kIoBlockSamples,
                                    raw.sample_count() - offset));
        if (!raw.read(offset, count, block)) {
            return fail(error, "raw F0 spool read failed");
        }
        for (std::size_t local = 0; local < block.size(); ++local) {
            if (block[local] > 0.0f) {
                index = offset + local;
                value = block[local];
                found = true;
                return true;
            }
        }
        offset += count;
    }
    return true;
}

bool append_linear_gap(bounded_io::RawFloatSpool& spool,
                       std::uint64_t left_index, float left_value,
                       std::uint64_t right_index, float right_value,
                       std::string& error) {
    if (right_index <= left_index) return fail(error, "invalid F0 interpolation gap");
    const std::uint64_t distance = right_index - left_index;
    std::vector<float> block;
    block.resize(kIoBlockSamples);
    for (std::uint64_t first = left_index + 1; first <= right_index;) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(block.size(), right_index - first + 1));
        for (std::size_t local = 0; local < count; ++local) {
            const std::uint64_t index = first + local;
            const float fraction = static_cast<float>(index - left_index) /
                                   static_cast<float>(distance);
            block[local] = left_value + (right_value - left_value) * fraction;
        }
        if (!spool.append(std::span<const float>(block.data(), count))) {
            return fail(error, "interpolated F0 spool append failed");
        }
        first += count;
    }
    return true;
}

bool build_processed_f0(bounded_io::RawFloatSpool& raw,
                        const std::filesystem::path& spool_directory,
                        bool interpolate_unvoiced, int filter_radius,
                        int pitch_shift,
                        bounded_io::RawFloatSpool& processed,
                        std::string& error) {
    bounded_io::RawFloatSpool interpolated;
    if (!interpolated.create(spool_directory)) {
        return fail(error, "failed to create interpolated F0 spool");
    }

    if (!interpolate_unvoiced) {
        if (!copy_spool(raw, interpolated, error)) return false;
    } else {
        std::uint64_t first_index = 0;
        float first_value = 0.0f;
        bool found = false;
        if (!find_first_voiced(raw, first_index, first_value, found, error)) return false;
        if (!found) {
            if (!copy_spool(raw, interpolated, error)) return false;
        } else {
            if (!append_repeated(interpolated, first_value, first_index + 1, error)) {
                return false;
            }
            std::uint64_t left_index = first_index;
            float left_value = first_value;
            std::vector<float> block;
            for (std::uint64_t offset = first_index + 1;
                 offset < raw.sample_count();) {
                const std::size_t count = static_cast<std::size_t>(
                    std::min<std::uint64_t>(kIoBlockSamples,
                                            raw.sample_count() - offset));
                if (!raw.read(offset, count, block)) {
                    return fail(error, "raw F0 interpolation read failed");
                }
                for (std::size_t local = 0; local < block.size(); ++local) {
                    if (block[local] <= 0.0f) continue;
                    const std::uint64_t right_index = offset + local;
                    if (!append_linear_gap(interpolated, left_index, left_value,
                                           right_index, block[local], error)) {
                        return false;
                    }
                    left_index = right_index;
                    left_value = block[local];
                }
                offset += count;
            }
            if (!append_repeated(interpolated, left_value,
                                 raw.sample_count() - left_index - 1, error)) {
                return false;
            }
        }
    }
    if (interpolated.sample_count() != raw.sample_count()) {
        return fail(error, "interpolated F0 spool length mismatch");
    }
    if (!processed.create(spool_directory)) {
        return fail(error, "failed to create processed F0 spool");
    }

    const int radius = filter_radius >= 3 ? (filter_radius | 1) / 2 : 0;
    const int kernel = 2 * radius + 1;
    if (kernel > 31) return fail(error, "F0 median filter exceeds the supported radius");
    const float pitch_factor = std::pow(2.0f, pitch_shift / 12.0f);
    if (!std::isfinite(pitch_factor)) return fail(error, "invalid F0 pitch factor");

    std::vector<float> halo;
    std::vector<float> output;
    output.resize(kIoBlockSamples);
    std::array<float, 31> median_window{};
    const std::uint64_t total = interpolated.sample_count();
    for (std::uint64_t offset = 0; offset < total;) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(output.size(), total - offset));
        const std::uint64_t halo_start = offset > static_cast<std::uint64_t>(radius)
            ? offset - radius : 0;
        const std::uint64_t halo_end = std::min<std::uint64_t>(
            total, offset + count + static_cast<std::uint64_t>(radius));
        const std::size_t halo_count = static_cast<std::size_t>(halo_end - halo_start);
        if (!interpolated.read(halo_start, halo_count, halo)) {
            return fail(error, "processed F0 halo read failed");
        }
        for (std::size_t local = 0; local < count; ++local) {
            const std::uint64_t global = offset + local;
            float value = halo[static_cast<std::size_t>(global - halo_start)];
            if (radius > 0) {
                for (int delta = -radius; delta <= radius; ++delta) {
                    const std::uint64_t clamped = delta < 0 &&
                            static_cast<std::uint64_t>(-delta) > global
                        ? 0
                        : std::min<std::uint64_t>(
                              total - 1,
                              delta < 0 ? global - static_cast<std::uint64_t>(-delta)
                                        : global + static_cast<std::uint64_t>(delta));
                    median_window[static_cast<std::size_t>(delta + radius)] =
                        halo[static_cast<std::size_t>(clamped - halo_start)];
                }
                std::sort(median_window.begin(), median_window.begin() + kernel);
                value = median_window[static_cast<std::size_t>(radius)];
            }
            if (value > 0.0f) value *= pitch_factor;
            if (!std::isfinite(value)) {
                return fail(error, "processed F0 is not finite");
            }
            output[local] = value;
        }
        if (!processed.append(std::span<const float>(output.data(), count))) {
            return fail(error, "processed F0 spool append failed");
        }
        offset += count;
    }
    return processed.sample_count() == raw.sample_count() ||
           fail(error, "processed F0 spool length mismatch");
}

bool compute_rms_records(bounded_io::RawFloatSpool& audio,
                         std::uint64_t hop, std::uint64_t frame,
                         const std::filesystem::path& spool_directory,
                         bounded_io::RawFloatSpool& records,
                         std::string& error) {
    if (!records.create(spool_directory) || hop == 0 || frame == 0 ||
        audio.sample_count() == 0) {
        return fail(error, "failed to initialize RMS record spool");
    }
    const std::uint64_t frames =
        audio.sample_count() / hop + 1;
    std::vector<float> samples;
    for (std::uint64_t index = 0; index < frames; ++index) {
        const std::uint64_t center = index * hop;
        const std::uint64_t left = frame / 2;
        const std::uint64_t start = center >= left ? center - left : 0;
        const std::uint64_t end = center +
            std::min(frame - left, audio.sample_count() - center);
        if (end <= start) return fail(error, "invalid RMS record range");
        std::size_t count = 0;
        if (!narrow_size(end - start, count) || !audio.read(start, count, samples)) {
            return fail(error, "RMS source read failed");
        }
        double sum = 0.0;
        for (const float value : samples) sum += static_cast<double>(value) * value;
        const float rms = static_cast<float>(std::sqrt(sum / static_cast<double>(frame)));
        if (!std::isfinite(rms) ||
            !records.append(std::span<const float>(&rms, 1))) {
            return fail(error, "RMS record append failed");
        }
    }
    return records.sample_count() == frames ||
           fail(error, "RMS record count mismatch");
}

class EnvelopeCursor {
public:
    EnvelopeCursor(bounded_io::RawFloatSpool& records,
                   std::uint64_t output_samples)
        : records_(records), output_samples_(output_samples) {}

    bool value(std::uint64_t output_index, float& result,
               std::string& error) {
        const std::uint64_t record_count = records_.sample_count();
        if (record_count == 0 || output_index >= output_samples_) {
            return fail(error, "invalid RMS envelope lookup");
        }
        // PyTorch linear interpolation with align_corners=false. RMS records
        // are centered with zero padding, so neither end is shifted forward.
        const double position = std::clamp(
            (static_cast<double>(output_index) + 0.5) *
                static_cast<double>(record_count) / static_cast<double>(output_samples_) - 0.5,
            0.0, static_cast<double>(record_count - 1));
        const std::uint64_t first = static_cast<std::uint64_t>(position);
        if (first != cached_first_) {
            const std::size_t count = first + 1 < record_count ? 2 : 1;
            if (!records_.read(first, count, cached_)) {
                return fail(error, "RMS envelope record read failed");
            }
            cached_first_ = first;
        }
        const float fraction = static_cast<float>(position - first);
        const float left = cached_[0];
        const float right = cached_.size() > 1 ? cached_[1] : cached_[0];
        result = left + (right - left) * fraction;
        return std::isfinite(result) || fail(error, "RMS envelope is not finite");
    }

private:
    bounded_io::RawFloatSpool& records_;
    std::uint64_t output_samples_ = 0;
    std::uint64_t cached_first_ = std::numeric_limits<std::uint64_t>::max();
    std::vector<float> cached_;
};

bool write_final_output(bounded_io::RawFloatSpool& filtered_source,
                        bounded_io::RawFloatSpool& raw_output,
                        const std::filesystem::path& spool_directory,
                        double rms_mix_rate,
                        bounded_io::AtomicWavStreamWriter& writer,
                        std::string& error) {
    if (raw_output.sample_count() != writer.expected_samples()) {
        return fail(error, "raw synthesis length does not match the WAV contract");
    }

    bounded_io::RawFloatSpool source_rms;
    bounded_io::RawFloatSpool output_rms;
    if (rms_mix_rate < 1.0) {
        if (!compute_rms_records(filtered_source, 8'000, 16'000,
                                 spool_directory, source_rms, error) ||
            !compute_rms_records(raw_output, 20'000, 40'000,
                                 spool_directory, output_rms, error)) {
            return false;
        }
    }

    EnvelopeCursor source_envelope(source_rms, raw_output.sample_count());
    EnvelopeCursor output_envelope(output_rms, raw_output.sample_count());
    std::vector<float> block;
    for (std::uint64_t offset = 0; offset < raw_output.sample_count();) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(kIoBlockSamples,
                                    raw_output.sample_count() - offset));
        if (!raw_output.read(offset, count, block)) {
            return fail(error, "raw synthesis spool read failed");
        }
        if (rms_mix_rate < 1.0) {
            for (std::size_t local = 0; local < block.size(); ++local) {
                float source_rms_value = 0.0f;
                float output_rms_value = 0.0f;
                if (!source_envelope.value(offset + local, source_rms_value, error) ||
                    !output_envelope.value(offset + local, output_rms_value, error)) {
                    return false;
                }
                const float denominator = std::max(output_rms_value, 1e-6f);
                block[local] *=
                    std::pow(source_rms_value, static_cast<float>(1.0 - rms_mix_rate)) *
                    std::pow(denominator, static_cast<float>(rms_mix_rate - 1.0));
                if (!std::isfinite(block[local])) {
                    return fail(error, "global RMS produced non-finite audio");
                }
            }
        }
        if (!writer.append(block)) return fail(error, "incremental WAV append failed");
        offset += count;
    }
    return writer.samples_written() == writer.expected_samples() ||
           fail(error, "incremental WAV sample count mismatch");
}

}  // namespace

bool detail::resample_wav_to_stable_snapshot(
        bounded_io::WavChunkReader& reader,
        const std::filesystem::path& spool_directory,
        bounded_io::RawFloatSpool& snapshot,
        std::uint64_t expected_samples,
        std::string& error) {
    return build_stable_16k_snapshot(
        reader, spool_directory, snapshot, expected_samples, error);
}

bool detail::filter_stable_source(
        bounded_io::RawFloatSpool& source,
        const std::filesystem::path& spool_directory,
        bool apply_highpass,
        bounded_io::RawFloatSpool& filtered,
        std::string& error) {
    return build_filtered_source(
        source, spool_directory, apply_highpass, filtered, error);
}

bool detail::plan_first_quiet_core(
        bounded_io::RawFloatSpool& source,
        std::uint64_t source_samples,
        std::uint64_t chunk_samples,
        std::uint64_t query_samples,
        std::uint64_t hard_core_samples,
        std::uint64_t& core_start,
        std::uint64_t& core_end,
        bool& done,
        std::string& error) {
    QuietCorePlanner planner(source, source_samples, chunk_samples,
                             query_samples, hard_core_samples);
    return planner.next(core_start, core_end, done, error);
}

bool detail::process_f0(
        bounded_io::RawFloatSpool& raw,
        const std::filesystem::path& spool_directory,
        const VCConfig& config,
        bounded_io::RawFloatSpool& processed,
        std::string& error, double pad_seconds, double duration_seconds) {
    if (!valid_pitch_edits(config.pitch_edits)) return fail(error, "invalid pitch edits");
    if (!build_processed_f0(raw, spool_directory, config.interp_unvoiced,
                              config.filter_radius, config.f0_up_key,
                              processed, error)) return false;
    if (config.pitch_edits.empty()) return true;
    // The opt-in correction path has a fixed 600s ceiling; two 61k-frame host
    // arrays do not change the duration-independent waveform working set.
    if (!std::isfinite(duration_seconds) || duration_seconds <= 0 || duration_seconds > 600 ||
        !std::isfinite(pad_seconds) || pad_seconds < 0 || pad_seconds > 5 ||
        raw.sample_count() > 61100) return fail(error, "pitch edits require input <=600 seconds");
    std::vector<float> values, original;
    if (!processed.read(0, processed.sample_count(), values) ||
        !raw.read(0, raw.sample_count(), original)) return fail(error, "pitch edit read failed");
    for (const auto& edit : config.pitch_edits) {
        if (edit.end > duration_seconds) return fail(error, "pitch edit exceeds input duration");
        const auto first = static_cast<std::size_t>(std::llround((edit.start + pad_seconds) * 100));
        const auto last = static_cast<std::size_t>(std::llround((edit.end + pad_seconds) * 100));
        if (last >= values.size() || last <= first) return fail(error, "pitch edit outside F0 timeline");
        if (edit.mode == PitchEditMode::kSmooth) {
            for (auto i = first; i <= last; ++i)
                if (original[i] <= 0 || values[i] <= 0)
                    return fail(error, "smooth pitch edit crosses unvoiced audio; narrow the interval");
            const double a = std::log(values[first]), b = std::log(values[last]);
            for (auto i = first + 1; i < last; ++i)
                values[i] = std::exp(a + (b-a) * static_cast<double>(i-first)/(last-first));
        } else {
            const float factor = std::pow(2.0f, edit.semitones / 12.0);
            for (auto i = first; i < last; ++i) if (original[i] > 0) values[i] *= factor;
        }
    }
    bounded_io::RawFloatSpool corrected;
    if (!corrected.create(spool_directory) || !corrected.append(values))
        return fail(error, "pitch edit write failed or nonfinite result");
    processed = std::move(corrected);
    return true;
}

bool detail::write_global_rms_output(
        bounded_io::RawFloatSpool& filtered_source,
        bounded_io::RawFloatSpool& raw_output,
        const std::filesystem::path& spool_directory,
        double rms_mix_rate,
        bounded_io::AtomicWavStreamWriter& writer,
        std::string& error) {
    return write_final_output(filtered_source, raw_output, spool_directory,
                              rms_mix_rate, writer, error);
}

VCResult VoiceConversionPipeline::convert_file_bounded(
        const std::string& input_path, const std::string& output_path,
        int speaker_id,const ProgressObserver& observer,io::BlockCheckpoint* checkpoint) try {
    using clock = std::chrono::high_resolution_clock;
    const auto start_time = clock::now();
    VCResult result;

    const auto finish_failure = [&](const char* fallback) {
        result.success = false;
        if (result.error_message.empty()) result.error_message = fallback;
        result.total_ms = std::chrono::duration<double, std::milli>(
            clock::now() - start_time).count();
        return result;
    };

    if (!initialized_) {
        result.error_message = "pipeline not initialized";
        return finish_failure("pipeline not initialized");
    }
    if (speaker_id < 0 || speaker_id >= synth_.num_speakers()) {
        result.error_message = "speaker_id out of range";
        return finish_failure("speaker_id out of range");
    }
    if (config_.index_rate > 0.0 &&
        (!index_loaded_ || feature_index_.dim() != feature_dim_)) {
        result.error_message = "requested feature index is unavailable";
        return finish_failure("requested feature index is unavailable");
    }
    // No repository asset currently pairs the WavLM feature space with a
    // matching trained VITS checkpoint.  Legacy owning APIs retain their
    // existing behavior, but the new bounded profile must not treat a stock
    // HuBERT/ContentVec generator comparison as qualification.  Reject before
    // resolving either payload path or creating staging/spool files.
    if (config_.content_encoder == ContentEncoder::kWavlm) {
        result.error_message = "bounded WavLM profile is not qualified";
        return finish_failure("bounded WavLM profile is not qualified");
    }
    if (config_.content_encoder != ContentEncoder::kHubert ||
        (config_.f0_method != F0Method::kRmvpe &&
         config_.f0_method != F0Method::kFcpe)) {
        result.error_message = "unsupported bounded conversion profile";
        return finish_failure("unsupported bounded conversion profile");
    }

    const std::filesystem::path target(output_path);
    const std::filesystem::path spool_directory = target.parent_path().empty()
        ? std::filesystem::path(".") : target.parent_path();
    bounded_io::WavChunkReader reader;
    if (!reader.open(input_path)) {
        result.error_message = "failed to open bounded WAV input";
        return finish_failure("failed to open bounded WAV input");
    }
    std::uint64_t source_samples = 0;
    if (!stable_sample_count(reader.metadata(), source_samples) || source_samples < 2) {
        result.error_message = "invalid bounded WAV sample count";
        return finish_failure("invalid bounded WAV sample count");
    }
    std::uint64_t scaled_output = 0;
    if (!checked_mul(source_samples, kOutputSampleRate, scaled_output)) {
        result.error_message = "bounded output length overflow";
        return finish_failure("bounded output length overflow");
    }
    const std::uint64_t output_samples = scaled_output / kInputSampleRate;

    bounded_io::AtomicWavStreamWriter writer;
    if (!writer.open(target, synth_cfg_.sample_rate, output_samples)) {
        result.error_message = "failed to create incremental WAV staging";
        return finish_failure("failed to create incremental WAV staging");
    }
    report_progress(observer,"input-snapshot",0,0,"unknown");
    bounded_io::RawFloatSpool stable;
    if (!detail::resample_wav_to_stable_snapshot(
            reader, spool_directory, stable, source_samples,
            result.error_message)) {
        return finish_failure("bounded input snapshot failed");
    }

    report_progress(observer,"highpass",0,0,"unknown");
    bounded_io::RawFloatSpool filtered;
    if (!detail::filter_stable_source(
            stable, spool_directory, config_.apply_highpass, filtered,
            result.error_message)) {
        return finish_failure("bounded high-pass failed");
    }

    std::uint64_t requested_chunk = 0;
    std::uint64_t query = 0;
    std::uint64_t pad = 0;
    if (!seconds_to_samples(config_.max_chunk_sec, requested_chunk) ||
        !seconds_to_samples(config_.chunk_query_sec, query) ||
        !seconds_to_samples(config_.chunk_pad_sec, pad)) {
        result.error_message = "invalid bounded chunk configuration";
        return finish_failure("invalid bounded chunk configuration");
    }
    requested_chunk = std::max(requested_chunk, kPitchHop);
    query = std::max(query, kPitchHop);
    pad = std::max(pad, kPitchHop);
    pad = std::min(pad, source_samples - 1);

    std::uint64_t fixed_overhead = 0;
    std::uint64_t query_twice = 0;
    std::uint64_t pad_twice = 0;
    if (!checked_mul(query, 2, query_twice) || !checked_mul(pad, 2, pad_twice) ||
        !checked_add(query_twice, pad_twice, fixed_overhead) ||
        !checked_add(fixed_overhead, kPitchHop, fixed_overhead) ||
        fixed_overhead >= kContentMaximumSamples) {
        result.error_message = "bounded chunk context exceeds the content model limit";
        return finish_failure("bounded chunk context exceeds the content model limit");
    }
    const std::uint64_t effective_chunk = std::min(
        requested_chunk, kContentMaximumSamples - fixed_overhead);
    if (effective_chunk < kPitchHop) {
        result.error_message = "bounded chunk core is too small";
        return finish_failure("bounded chunk core is too small");
    }
    std::uint64_t hard_core = 0;
    std::uint64_t hard_segment = 0;
    if (!checked_add(effective_chunk, query_twice, hard_core) ||
        !checked_add(hard_core, pad_twice, hard_segment) ||
        !checked_add(hard_segment, kPitchHop, hard_segment) ||
        hard_segment > kContentMaximumSamples) {
        result.error_message = "bounded segment hard bound is invalid";
        return finish_failure("bounded segment hard bound is invalid");
    }

    const std::uint64_t f0_context =
        detail::f0_context_samples(config_.f0_method);
    bounded_io::RawFloatSpool raw_f0;
    std::uint64_t maximum_f0_input = 0;
    const auto f0_start = clock::now();
    if (!build_raw_f0(
            filtered, source_samples, pad, f0_context, spool_directory,
            [this](const float* audio, int samples) {
                return extract_f0(audio, samples);
            },
            raw_f0, maximum_f0_input, result.error_message,observer,checkpoint)) {
        return finish_failure("bounded F0 extraction failed");
    }
    bounded_io::RawFloatSpool processed_f0;
    if (!detail::process_f0(raw_f0, spool_directory, config_,
                            processed_f0, result.error_message,
                            static_cast<double>(pad)/16000, static_cast<double>(source_samples)/16000)) {
        return finish_failure("bounded F0 post-processing failed");
    }
    result.f0_ms = std::chrono::duration<double, std::milli>(
        clock::now() - f0_start).count();
    const std::uint64_t f0_hard_input =
        detail::kF0CoreSamples + 2 * f0_context;
    if (maximum_f0_input == 0 || maximum_f0_input > f0_hard_input) {
        result.error_message = "bounded F0 resource observation exceeded its profile";
        return finish_failure("bounded F0 resource observation exceeded its profile");
    }

    report_progress(observer,"synthesis",0,source_samples,"samples16k");
    bounded_io::RawFloatSpool raw_output;
    if (!raw_output.create(spool_directory)) {
        result.error_message = "failed to create raw synthesis spool";
        return finish_failure("failed to create raw synthesis spool");
    }
    QuietCorePlanner planner(filtered, source_samples, effective_chunk, query,
                             hard_core);
    const std::uint64_t left_output_context =
        pad * kOutputSampleRate / kInputSampleRate;
    std::vector<float> zero_block(kIoBlockSamples, 0.0f);
    while (true) {
        std::uint64_t core_start = 0;
        std::uint64_t core_end = 0;
        bool done = false;
        if (!planner.next(core_start, core_end, done, result.error_message)) {
            return finish_failure("bounded quiet planning failed");
        }
        if (done) break;

        const auto block_key="synth_"+std::to_string(core_start);
        if(checkpoint) {
            checkpoint->boundary();
            const auto start=core_start*kOutputSampleRate/kInputSampleRate;
            const auto end=core_end*kOutputSampleRate/kInputSampleRate;
            if(raw_output.sample_count()!=start||end<=start)
                return finish_failure("cached synthesis absolute alignment mismatch");
            if(auto cached=checkpoint->load(block_key,core_start,end-start)) {
                if(!raw_output.append(*cached))return finish_failure("cached synthesis spool append failed");
                report_progress(observer,"synthesis",core_end,source_samples,"samples16k");
                checkpoint->boundary();
                continue;
            }
        }

        std::uint64_t virtual_samples = 0;
        std::uint64_t right_context = 0;
        if (!checked_add(source_samples, 2 * pad, virtual_samples) ||
            !checked_add(core_end, 2 * pad, right_context) ||
            !checked_add(right_context, kPitchHop, right_context)) {
            result.error_message = "bounded segment range overflow";
            return finish_failure("bounded segment range overflow");
        }
        const std::uint64_t segment_end = std::min(virtual_samples, right_context);
        if (segment_end <= core_start ||
            segment_end - core_start > hard_segment ||
            segment_end - core_start > kContentMaximumSamples) {
            result.error_message = "bounded content segment exceeded its hard bound";
            return finish_failure("bounded content segment exceeded its hard bound");
        }
        const std::uint64_t segment_samples = segment_end - core_start;
        std::size_t segment_size = 0;
        int segment_count = 0;
        if (!narrow_size(segment_samples, segment_size) ||
            !narrow_int(segment_samples, segment_count)) {
            result.error_message = "bounded content segment exceeds the model range";
            return finish_failure("bounded content segment exceeds the model range");
        }
        std::vector<float> segment;
        if (!read_reflected(filtered, source_samples, pad, core_start,
                            segment_size, segment, result.error_message)) {
            return finish_failure("bounded reflected segment read failed");
        }

        const std::uint64_t first_f0 = core_start / kPitchHop;
        const std::uint64_t end_f0 =
            std::min(processed_f0.sample_count(), segment_end / kPitchHop);
        if (end_f0 <= first_f0) {
            result.error_message = "bounded F0 segment is empty";
            return finish_failure("bounded F0 segment is empty");
        }
        const std::uint64_t f0_samples = end_f0 - first_f0;
        std::size_t f0_size = 0;
        int f0_count = 0;
        if (!narrow_size(f0_samples, f0_size) || !narrow_int(f0_samples, f0_count)) {
            result.error_message = "bounded F0 segment exceeds the model range";
            return finish_failure("bounded F0 segment exceeds the model range");
        }
        std::vector<float> f0;
        std::vector<float> raw;
        if (!processed_f0.read(first_f0, f0_size, f0) ||
            !raw_f0.read(first_f0, f0_size, raw)) {
            result.error_message = "bounded F0 segment read failed";
            return finish_failure("bounded F0 segment read failed");
        }
        std::vector<std::uint8_t> uv(f0_size);
        for (std::size_t index = 0; index < f0_size; ++index) {
            uv[index] = raw[index] <= 0.0f ? 1 : 0;
        }

        auto piece = synth_segment(segment.data(), segment_count,
                                   f0.data(), uv.data(), f0_count,
                                   speaker_id, result, first_f0);
        if (piece.data.empty()) {
            return finish_failure("bounded chunk synthesis failed");
        }
        if (!std::all_of(piece.data.begin(), piece.data.end(),
                         [](float value) { return std::isfinite(value); })) {
            result.error_message = "bounded synthesis produced non-finite audio";
            return finish_failure("bounded synthesis produced non-finite audio");
        }

        const std::uint64_t output_start =
            core_start * kOutputSampleRate / kInputSampleRate;
        const std::uint64_t output_end =
            core_end * kOutputSampleRate / kInputSampleRate;
        if (raw_output.sample_count() != output_start || output_end < output_start) {
            result.error_message = "bounded synthesis lost absolute output alignment";
            return finish_failure("bounded synthesis lost absolute output alignment");
        }
        const std::uint64_t core_output = output_end - output_start;
        const std::uint64_t available = piece.data.size() > left_output_context
            ? piece.data.size() - left_output_context : 0;
        const std::uint64_t take = std::min(core_output, available);
        if (take > 0) {
            const auto* data = piece.data.data() +
                static_cast<std::size_t>(left_output_context);
            if (!raw_output.append(std::span<const float>(
                    data, static_cast<std::size_t>(take)))) {
                result.error_message = "raw synthesis spool append failed";
                return finish_failure("raw synthesis spool append failed");
            }
        }
        std::uint64_t remaining = core_output - take;
        while (remaining > 0) {
            const std::size_t count = static_cast<std::size_t>(
                std::min<std::uint64_t>(zero_block.size(), remaining));
            if (!raw_output.append(std::span<const float>(zero_block.data(), count))) {
                result.error_message = "raw synthesis zero-fill failed";
                return finish_failure("raw synthesis zero-fill failed");
            }
            remaining -= count;
        }
        if(checkpoint) {
            std::vector<float> committed;
            if(!raw_output.read(output_start,static_cast<std::size_t>(core_output),committed))
                return finish_failure("cannot read completed synthesis block");
            checkpoint->save(block_key,core_start,committed);
        }
        report_progress(observer,"synthesis",core_end,source_samples,"samples16k");
        if(checkpoint)checkpoint->boundary();
    }
    if (raw_output.sample_count() != output_samples) {
        result.error_message = "bounded synthesis output length mismatch";
        return finish_failure("bounded synthesis output length mismatch");
    }

    report_progress(observer,"output",0,0,"unknown");
    if (!detail::write_global_rms_output(
            filtered, raw_output, spool_directory, config_.rms_mix_rate,
            writer, result.error_message)) {
        return finish_failure("bounded global RMS or WAV write failed");
    }
    if (!writer.commit()) {
        result.error_message = "failed to commit bounded WAV output";
        return finish_failure("failed to commit bounded WAV output");
    }

    result.audio.data.clear();
    result.audio.sample_rate = synth_cfg_.sample_rate;
    result.success = true;
    result.error_message.clear();
    result.total_ms = std::chrono::duration<double, std::milli>(
        clock::now() - start_time).count();
    return result;
} catch (const io::OperationPaused& error) {
    VCResult result;
    result.paused=checkpoint!=nullptr;
    result.error_message=error.what();
    return result;
} catch (const std::exception& error) {
    VCResult result;
    result.error_message = "bounded file conversion runtime exception: ";
    result.error_message += error.what();
    return result;
} catch (...) {
    VCResult result;
    result.error_message = "bounded file conversion failed with an unknown exception";
    return result;
}

}  // namespace voxmutatio::pipeline
