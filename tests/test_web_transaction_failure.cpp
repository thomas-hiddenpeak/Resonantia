// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <fcntl.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <string>
#include <vector>

#include "tools/web/workflow_support.h"

// Isolated process: inject one allocation failure immediately after an actual
// failed publish rename. No production fault hooks or HTTP worker interposition.
namespace {
const char* fail_source = nullptr;
bool fail_next_allocation = false;
int allocation_failures = 0;
int observed_rename_failures = 0;
}

void* operator new(std::size_t size) {
    if (fail_next_allocation) {
        fail_next_allocation = false;
        ++allocation_failures;
        throw std::bad_alloc();
    }
    if (void* result = std::malloc(size == 0 ? 1 : size)) return result;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

extern "C" int rename(const char* source, const char* destination) noexcept {
    const int result = ::renameat(AT_FDCWD, source, AT_FDCWD, destination);
    if (result != 0 && fail_source && std::strcmp(source, fail_source) == 0) {
        ++observed_rename_failures;
        fail_next_allocation = true;
    }
    return result;
}

namespace {
namespace fs = std::filesystem;

struct TemporaryTree {
    fs::path path;
    TemporaryTree() {
        std::string pattern = (fs::temp_directory_path() / "resonantia-web-rollback-XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        char* made = ::mkdtemp(buffer.data());
        if (!made) throw std::runtime_error("mkdtemp failed");
        path = made;
    }
    ~TemporaryTree() {
        fail_next_allocation = false;
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void write_file(const fs::path& path, const std::string& bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output << bytes;
    output.close();
    if (!output) throw std::runtime_error("fixture write failed");
}

bool contents_match(const fs::path& path, const std::string& bytes) {
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), {}) == bytes;
}

bool old_artifact_preserved(const fs::path& directory, const std::string& prefix,
                            const std::string& relative) {
    const std::string expected = "old:" + relative;
    if (contents_match(directory / relative, expected)) return true;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().filename().string().rfind(prefix, 0) == 0 &&
            contents_match(entry.path() / relative, expected)) return true;
    }
    return false;
}
}

int main() {
    try {
        TemporaryTree tree;
        int failures = 0;
        for (const std::string kind : {"material", "step", "training"}) {
            const fs::path directory = tree.path / kind;
            const fs::path staged = directory / "staged";
            const std::vector<std::string> originals = kind == "material"
                ? std::vector<std::string>{"raw/old.wav", "work/old.wav", "steps.txt"}
                : kind == "step" ? std::vector<std::string>{"work/old.wav", "steps.txt"}
                                 : std::vector<std::string>{"clips/old.wav", "model.safetensors", "model.index"};
            for (const std::string& relative : originals)
                write_file(directory / relative, "old:" + relative);
            if (kind == "material") write_file(staged / "raw/new.wav", "new");
            else if (kind == "step") write_file(staged / "work/new.wav", "new");
            else {
                write_file(staged / "clips/new.wav", "new");
                write_file(staged / "model.safetensors", "new");
            }
            const fs::path first = staged / (kind == "material" ? "raw" : kind == "step" ? "work" : "clips");
            const fs::path second = staged / (kind == "material" ? "work" : kind == "step" ? "steps.txt" : "model.safetensors");
            const fs::path third = staged / "model.index";
            const std::string missing = (kind == "training" ? third : second).string();
            std::string error;
            allocation_failures = observed_rename_failures = 0;
            fail_source = missing.c_str();
            bool committed = false;
            try {
                using namespace resonantia::web;
                committed = kind == "material" ? replace_material_dirs(directory, first, second, error)
                    : kind == "step" ? replace_step_artifacts(directory, first, second, error)
                                     : replace_training_artifacts(directory, first, second, third, error);
            } catch (const std::bad_alloc&) {
                // Propagation is allowed; destroying the last old copy is not.
            }
            fail_source = nullptr;
            fail_next_allocation = false;
            bool preserved = !committed && observed_rename_failures == 1 && allocation_failures == 1;
            const std::string prefix = kind == "material" ? ".material_backup_"
                                     : kind == "step" ? ".step_backup_" : ".train_backup_";
            for (const std::string& relative : originals) {
                if (!old_artifact_preserved(directory, prefix, relative)) {
                    std::cerr << "LOST " << kind << "/" << relative << '\n';
                    preserved = false;
                }
            }
            std::cout << kind << ": rename failures=" << observed_rename_failures
                      << ", one-shot allocation failures=" << allocation_failures
                      << ", old artifacts preserved=" << preserved << '\n';
            if (!preserved) ++failures;
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        fail_source = nullptr;
        fail_next_allocation = false;
        std::cerr << error.what() << '\n';
        return 1;
    }
}
