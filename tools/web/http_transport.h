#pragma once

#include <cstddef>
#include <functional>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace resonantia::web {

inline constexpr std::size_t kMaxHeaderBytes = 64u << 10;
inline constexpr std::size_t kMaxBodyBytes = 128u << 20;
inline constexpr std::size_t kMaxInflightBodyBytes = 256u << 20;
inline constexpr std::size_t kMaxFileBytes = 64u << 20;
inline constexpr std::size_t kMaxFieldBytes = 16u << 10;
inline constexpr std::size_t kMaxMultipartParts = 128;
inline constexpr int kMaxConnections = 32;
inline constexpr int kSocketTimeoutSeconds = 15;
inline constexpr int kHeaderReadTimeoutSeconds = 10;
inline constexpr int kBodyReadTimeoutSeconds = 120;

namespace detail {
struct HttpRuntimeState;
}

struct Request {
    Request();
    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    Request(Request&&) = delete;
    Request& operator=(Request&&) = delete;
    ~Request();

    std::string method;
    std::string path;
    std::string query;
    std::string content_type;
    std::string authorization;
    std::string range;
    std::string body;

    // Internal transport hook: records the aggregate body reservation so its
    // lifetime follows this request on every handler exit path.
    void hold_body_reservation(std::shared_ptr<detail::HttpRuntimeState> runtime,
                               std::size_t bytes) noexcept;

private:
    std::shared_ptr<detail::HttpRuntimeState> runtime_;
    std::size_t reserved_body_bytes_ = 0;
};

struct Part {
    std::string name;
    std::string filename;
    std::string data;
};

struct HttpTransportConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string auth_token;
};

using RequestHandler = std::function<void(int, const Request&)>;
using ListeningHandler = std::function<void()>;

class HttpTransport {
public:
    explicit HttpTransport(HttpTransportConfig config);
    HttpTransport(const HttpTransport&) = delete;
    HttpTransport& operator=(const HttpTransport&) = delete;

    [[nodiscard]] int serve(RequestHandler handler,
                            ListeningHandler on_listening = {});

private:
    HttpTransportConfig config_;
    std::shared_ptr<detail::HttpRuntimeState> runtime_;
};

[[nodiscard]] bool valid_ipv4_address(const std::string& address);

void send_response(int fd, int status, const char* text, const std::string& content_type,
                   const std::string& body, const std::string& extra_headers = "");
void send_json(int fd, const std::string& json);
// Regular immutable task artifacts only. Sends bounded 64 KiB blocks and one
// optional byte range; malformed/unsatisfiable ranges return 416, never full audio.
void send_audio_file(int fd, const std::filesystem::path& path, const std::string& range);

// Parses multipart/form-data and emits the same HTTP error contract used by all
// upload routes. The returned Part views own their strings independently.
[[nodiscard]] bool request_parts(int fd, const Request& request,
                                 std::vector<Part>& parts);

}  // namespace resonantia::web
