include_guard(GLOBAL)

# Prefer a system package, but keep a reproducible fallback for developer and CI
# machines. The archive is content-addressed; no branch or mutable Git tag is run.
find_package(Catch2 3.4 CONFIG QUIET)
if(NOT Catch2_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.4.0.tar.gz
        URL_HASH SHA256=122928b814b75717316c71af69bd2b43387643ba076a6ec16e7882bfb2dfacbb
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    )
    FetchContent_MakeAvailable(Catch2)
endif()
