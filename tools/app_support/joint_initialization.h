#pragma once
#include <filesystem>
#include <string_view>
namespace voxmutatio::app_support::joint_full {
// Called with the training directory lock held. Only an interrupted step-zero
// initialization is eligible: no published checkpoint, candidate or unknown file.
void validate_initial_training_directory(const std::filesystem::path& directory);
// Verify any committed identity before removing only recognized partial writes.
void recover_initial_training_directory(const std::filesystem::path& directory,
                                        std::string_view identity);
}
