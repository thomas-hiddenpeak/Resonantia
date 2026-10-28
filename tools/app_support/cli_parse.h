// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

// Shared strict parsing for numeric command-line options. This header is
// private to tools/: it deliberately does not add to the library API.

#pragma once

#include <charconv>
#include <cmath>
#include <concepts>
#include <ostream>
#include <string_view>
#include "voxmutatio/core/types.h"

namespace voxmutatio::cli {

inline bool parse_synthesis_mode(std::string_view value, SynthesisMode& mode) {
    if (value == "sampled") { mode = SynthesisMode::kSampled; return true; }
    if (value == "mean") { mode = SynthesisMode::kMean; return true; }
    return false;
}

inline const char* option_value(int argc, char* const argv[], int& index,
                                std::string_view option, std::ostream& error) {
    if (index + 1 >= argc) {
        error << "error: missing value for " << option << '\n';
        return nullptr;
    }
    return argv[++index];
}

template <std::integral T>
[[nodiscard]] bool parse_integer(std::string_view text, T minimum, T maximum,
                                 T& output) noexcept {
    if (text.empty() || minimum > maximum) return false;
    // from_chars intentionally omits a leading '+'. Accept it explicitly to
    // preserve conventional CLI spellings such as --pitch +12, while still
    // requiring at least one digit and full consumption below.
    if (text.front() == '+') {
        text.remove_prefix(1);
        if (text.empty() || text.front() == '+' || text.front() == '-') return false;
    }
    T staged {};
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto result = std::from_chars(begin, end, staged, 10);
    if (result.ec != std::errc {} || result.ptr != end ||
        staged < minimum || staged > maximum) {
        return false;
    }
    output = staged;
    return true;
}

template <std::floating_point T>
[[nodiscard]] bool parse_real(std::string_view text, T minimum, T maximum,
                              T& output) noexcept {
    if (text.empty() || minimum > maximum) return false;
    if (text.front() == '+') {
        text.remove_prefix(1);
        if (text.empty() || text.front() == '+' || text.front() == '-') return false;
    }
    T staged {};
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto result = std::from_chars(begin, end, staged, std::chars_format::general);
    if (result.ec != std::errc {} || result.ptr != end || !std::isfinite(staged) ||
        staged < minimum || staged > maximum) {
        return false;
    }
    output = staged;
    return true;
}

template <std::integral T>
[[nodiscard]] bool parse_integer_option(int argc, char* const argv[], int& index,
                                        std::string_view option, T minimum,
                                        T maximum, T& output,
                                        std::ostream& error) {
    const char* const value = option_value(argc, argv, index, option, error);
    if (!value) return false;
    if (!parse_integer<T>(value, minimum, maximum, output)) {
        error << "error: invalid value for " << option << ": '" << value << "'\n";
        return false;
    }
    return true;
}

template <std::floating_point T>
[[nodiscard]] bool parse_real_option(int argc, char* const argv[], int& index,
                                     std::string_view option, T minimum,
                                     T maximum, T& output,
                                     std::ostream& error) {
    const char* const value = option_value(argc, argv, index, option, error);
    if (!value) return false;
    if (!parse_real<T>(value, minimum, maximum, output)) {
        error << "error: invalid value for " << option << ": '" << value << "'\n";
        return false;
    }
    return true;
}

}  // namespace voxmutatio::cli
