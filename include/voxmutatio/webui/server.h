#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace voxmutatio::webui {

/// HTTP request method
enum class HttpMethod : std::uint8_t {
    kGet,
    kPost,
    kPut,
    kDelete,
    kOptions,
};

/// HTTP request
struct HttpRequest {
    HttpMethod method;
    std::string path;
    std::unordered_map<std::string, std::string> headers;
    std::unordered_map<std::string, std::string> query_params;
    std::vector<uint8_t> body;
};

/// HTTP response
struct HttpResponse {
    int status_code = 200;
    std::unordered_map<std::string, std::string> headers;
    std::vector<uint8_t> body;
    
    /// Helper: set JSON body
    void set_json(const std::string& json_str);
    
    /// Helper: set text body
    void set_text(const std::string& text);
    
    /// Helper: set file body
    void set_file(const std::string& file_path, 
                  const std::string& content_type);
};

/// Route handler function type
using RouteHandler = std::function<HttpResponse(const HttpRequest&)>;

/// Web server configuration
struct WebServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    int max_connections = 100;
    std::string static_files_dir;  // Directory for static assets (frontend)
};

/// Deprecated placeholder for the former embedded HTTP server surface.
///
/// This class never implemented an HTTP listener. It remains source-compatible
/// for one migration window; use the `vc_serve` executable and its documented
/// HTTP API for all supported Web service behavior.
class [[deprecated("Embedded WebServer is unsupported; use the vc_serve executable")]] WebServer {
public:
    WebServer();
    ~WebServer();

    WebServer(WebServer&&) noexcept;
    WebServer& operator=(WebServer&&) noexcept;

    WebServer(const WebServer&) = delete;
    WebServer& operator=(const WebServer&) = delete;

    /// Initialize server
    bool init(const WebServerConfig& config);
    
    /// Register a route handler
    void register_route(HttpMethod method, const std::string& path,
                       RouteHandler handler);
    
    /// Fail fast because the legacy embedded server is unsupported.
    ///
    /// Always throws std::logic_error with migration guidance for `vc_serve`.
    void start();
    
    /// Stop server
    void stop();
    
    /// Check if server is running
    [[nodiscard]] bool is_running() const noexcept { return running_; }

private:
    WebServerConfig config_;
    std::unordered_map<std::string, RouteHandler> routes_;
    bool running_ = false;
    
    // Server implementation details
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/// Legacy placeholder routes retained only for source compatibility.
[[deprecated("Legacy embedded routes are unsupported; use the vc_serve HTTP API")]]
void register_vc_api(WebServer& server);

}  // namespace voxmutatio::webui
