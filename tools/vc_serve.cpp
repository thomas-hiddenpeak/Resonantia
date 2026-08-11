/**
 * @file vc_serve.cpp
 * @brief Startup configuration and assembly for the Resonantia Web service.
 */

#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include "tools/web/http_transport.h"
#include "tools/web/web_service.h"
#include "tools/web/workflow_support.h"

namespace fs = std::filesystem;

namespace {

struct StartupConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string webroot = "webui";
    std::string repo;
    std::string auth_token;
};

bool parse_startup_config(int argc, char** argv, StartupConfig& config, int& exit_code) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            std::printf("Usage: vc_serve [--host IPv4] [--port 8080] [--webroot webui] "
                        "[--repo PATH] [--auth-token TOKEN]\n");
            exit_code = 0;
            return false;
        }
        if (argument != "--host" && argument != "--port" && argument != "--webroot" &&
            argument != "--repo" && argument != "--auth-token") {
            std::fprintf(stderr, "Unknown argument: %s\n", argument.c_str());
            exit_code = 1;
            return false;
        }
        if (index + 1 >= argc) {
            std::fprintf(stderr, "Missing value for %s\n", argument.c_str());
            exit_code = 1;
            return false;
        }
        const std::string value = argv[++index];
        if (argument == "--host") {
            config.host = value;
        } else if (argument == "--port") {
            if (!resonantia::web::parse_int_field(value, 1, 65535, config.port)) {
                std::fprintf(stderr, "Invalid port: %s\n", value.c_str());
                exit_code = 1;
                return false;
            }
        } else if (argument == "--webroot") {
            config.webroot = value;
        } else if (argument == "--repo") {
            config.repo = value;
        } else {
            config.auth_token = value;
        }
    }
    if (config.auth_token.size() > 4096) {
        std::fprintf(stderr, "Auth token is too long\n");
        exit_code = 1;
        return false;
    }
    if (!resonantia::web::valid_ipv4_address(config.host)) {
        std::fprintf(stderr, "Invalid IPv4 address: %s\n", config.host.c_str());
        exit_code = 1;
        return false;
    }
    return true;
}

bool resolve_paths(const StartupConfig& config, resonantia::web::Paths& paths) {
    std::error_code error;
    const fs::path canonical_webroot = fs::canonical(fs::absolute(config.webroot), error);
    if (error || !fs::is_directory(canonical_webroot)) {
        std::fprintf(stderr, "Invalid webroot: %s\n", config.webroot.c_str());
        return false;
    }

    paths.webroot = canonical_webroot.string();
    paths.repo = config.repo.empty() ? canonical_webroot.parent_path().string()
                                     : fs::absolute(config.repo).string();
    paths.models = paths.repo + "/models";
    paths.build = paths.repo + "/build";
    paths.runs = paths.repo + "/runs";
    paths.hubert = paths.models + "/hubert_base/model.safetensors";
    paths.rmvpe = paths.models + "/rmvpe.safetensors";
    paths.gmodel = paths.models + "/pretrained_v2/pretrained_v2/f0G40k.safetensors";
    paths.dmodel = paths.models + "/pretrained_v2/pretrained_v2/f0D40k.safetensors";

    fs::create_directories(paths.runs, error);
    if (error) {
        std::fprintf(stderr, "Cannot create runs directory: %s\n", error.message().c_str());
        return false;
    }
    return true;
}

int run_server(int argc, char** argv) {
    StartupConfig config;
    int exit_code = 0;
    if (!parse_startup_config(argc, argv, config, exit_code)) return exit_code;

    resonantia::web::Paths paths;
    if (!resolve_paths(config, paths)) return 1;
    const std::string repo = paths.repo;

    auto process_runner = std::make_shared<resonantia::web::PosixProcessRunner>();
    auto service = std::make_shared<resonantia::web::WebService>(
        std::move(paths), std::move(process_runner));
    resonantia::web::HttpTransport transport({config.host, config.port, config.auth_token});

    return transport.serve(
        [service](int fd, const resonantia::web::Request& request) {
            service->handle_request(fd, request);
        },
        [&config, &repo] {
            std::printf("Resonantia WebUI on http://%s:%d\n  repo=%s\n  authentication=%s\n",
                        config.host.c_str(), config.port, repo.c_str(),
                        config.auth_token.empty() ? "disabled" : "enabled");
            if (config.host == "0.0.0.0" && config.auth_token.empty()) {
                std::fprintf(stderr,
                             "[vc_serve] warning: listening on all IPv4 interfaces without "
                             "authentication\n");
            }
        });
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run_server(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "vc_serve fatal error: %s\n", error.what());
    } catch (...) {
        std::fprintf(stderr, "vc_serve fatal error\n");
    }
    return 1;
}
