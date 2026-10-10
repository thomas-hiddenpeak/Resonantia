#include "tools/web/http_transport.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <sstream>
#include <string_view>
#include <thread>
#include <utility>

namespace resonantia::web {

namespace detail {

struct HttpRuntimeState {
    std::atomic<int> active_connections{0};
    std::atomic<std::size_t> inflight_body_bytes{0};
};

}  // namespace detail

namespace {

enum class ReadResult {
    kOk,
    kBadRequest,
    kPayloadTooLarge,
    kUnauthorized,
    kServerBusy,
};

enum class MultipartResult {
    kOk,
    kMalformed,
    kTooLarge,
    kTooManyParts,
};

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool parse_size(const std::string& value, std::size_t& output) {
    if (value.empty()) return false;
    std::size_t parsed = 0;
    for (unsigned char c : value) {
        if (!std::isdigit(c)) return false;
        const std::size_t digit = static_cast<std::size_t>(c - '0');
        if (parsed > (std::numeric_limits<std::size_t>::max() - digit) / 10) return false;
        parsed = parsed * 10 + digit;
    }
    output = parsed;
    return true;
}

bool constant_time_equal(const std::string& left, const std::string& right) {
    const std::size_t count = std::max(left.size(), right.size());
    std::size_t difference = left.size() ^ right.size();
    for (std::size_t i = 0; i < count; ++i) {
        const unsigned char a = i < left.size() ? static_cast<unsigned char>(left[i]) : 0;
        const unsigned char b = i < right.size() ? static_cast<unsigned char>(right[i]) : 0;
        difference |= static_cast<std::size_t>(a ^ b);
    }
    return difference == 0;
}

bool authorized(const Request& request, const std::string& token) {
    if (token.empty()) return true;
    constexpr char kBearer[] = "Bearer ";
    if (request.authorization.size() < sizeof(kBearer) - 1) return false;
    for (std::size_t i = 0; i < sizeof(kBearer) - 1; ++i) {
        if (std::tolower(static_cast<unsigned char>(request.authorization[i])) !=
            std::tolower(static_cast<unsigned char>(kBearer[i]))) {
            return false;
        }
    }
    return constant_time_equal(request.authorization.substr(sizeof(kBearer) - 1), token);
}

ssize_t receive_before(int fd, char* data, std::size_t size,
                       std::chrono::steady_clock::time_point deadline) {
    for (;;) {
        const auto remaining = std::chrono::duration_cast<std::chrono::microseconds>(
            deadline - std::chrono::steady_clock::now());
        if (remaining.count() <= 0) {
            errno = EAGAIN;
            return -1;
        }
        timeval timeout{
            static_cast<time_t>(remaining.count() / 1'000'000),
            static_cast<suseconds_t>(remaining.count() % 1'000'000),
        };
        if (timeout.tv_sec == 0 && timeout.tv_usec == 0) timeout.tv_usec = 1;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        const ssize_t received = recv(fd, data, size, 0);
        if (received < 0 && errno == EINTR) continue;
        return received;
    }
}

ReadResult read_request(int fd, Request& request, const std::string& auth_token,
                        const std::shared_ptr<detail::HttpRuntimeState>& runtime) {
    std::string buffer;
    char chunk[8192];
    std::size_t header_end = std::string::npos;
    const auto header_deadline = std::chrono::steady_clock::now() +
                                 std::chrono::seconds(kHeaderReadTimeoutSeconds);
    while (header_end == std::string::npos) {
        const ssize_t received = receive_before(fd, chunk, sizeof(chunk), header_deadline);
        if (received <= 0) return ReadResult::kBadRequest;
        buffer.append(chunk, static_cast<std::size_t>(received));
        header_end = buffer.find("\r\n\r\n");
        if (header_end == std::string::npos && buffer.size() > kMaxHeaderBytes) {
            return ReadResult::kPayloadTooLarge;
        }
    }
    if (header_end > kMaxHeaderBytes) return ReadResult::kPayloadTooLarge;

    const std::string head = buffer.substr(0, header_end);
    std::string body = buffer.substr(header_end + 4);
    std::istringstream headers(head);
    std::string line;
    std::getline(headers, line);
    {
        std::istringstream first_line(line);
        std::string target;
        std::string version;
        std::string extra;
        if (!(first_line >> request.method >> target >> version) || (first_line >> extra) ||
            (version != "HTTP/1.0" && version != "HTTP/1.1") || target.empty()) {
            return ReadResult::kBadRequest;
        }
        const std::size_t query = target.find('?');
        if (query == std::string::npos) {
            request.path = target;
        } else {
            request.path = target.substr(0, query);
            request.query = target.substr(query + 1);
        }
        if (request.path.empty() || request.path.front() != '/') {
            return ReadResult::kBadRequest;
        }
    }

    std::size_t content_length = 0;
    bool have_content_length = false;
    bool have_content_type = false;
    bool have_authorization = false;
    while (std::getline(headers, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.front() == ' ' || line.front() == '\t') {
            return ReadResult::kBadRequest;
        }
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos || colon == 0) return ReadResult::kBadRequest;
        const std::string key = lower(line.substr(0, colon));
        std::string value = line.substr(colon + 1);
        for (unsigned char header_char : key) {
            if (!std::isalnum(header_char) && header_char != '-') {
                return ReadResult::kBadRequest;
            }
        }
        while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
            value.erase(value.begin());
        }
        while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
            value.pop_back();
        }
        if (key == "content-length") {
            std::size_t parsed = 0;
            if (have_content_length || !parse_size(value, parsed)) {
                return ReadResult::kBadRequest;
            }
            content_length = parsed;
            have_content_length = true;
        } else if (key == "content-type") {
            if (have_content_type) return ReadResult::kBadRequest;
            request.content_type = value;
            have_content_type = true;
        } else if (key == "authorization") {
            if (have_authorization) return ReadResult::kBadRequest;
            request.authorization = value;
            have_authorization = true;
        } else if (key == "transfer-encoding") {
            return ReadResult::kBadRequest;
        }
    }

    if (content_length > kMaxBodyBytes) return ReadResult::kPayloadTooLarge;
    if ((request.method == "POST" || request.method == "PUT") && !have_content_length) {
        return ReadResult::kBadRequest;
    }
    if (request.path.rfind("/api/", 0) == 0 && !authorized(request, auth_token)) {
        return ReadResult::kUnauthorized;
    }

    std::size_t inflight = runtime->inflight_body_bytes.load(std::memory_order_relaxed);
    for (;;) {
        if (inflight > kMaxInflightBodyBytes ||
            content_length > kMaxInflightBodyBytes - inflight) {
            return ReadResult::kServerBusy;
        }
        if (runtime->inflight_body_bytes.compare_exchange_weak(
                inflight, inflight + content_length, std::memory_order_relaxed)) {
            break;
        }
    }
    request.hold_body_reservation(runtime, content_length);

    if (body.size() > content_length) body.resize(content_length);
    const auto body_deadline = std::chrono::steady_clock::now() +
                               std::chrono::seconds(kBodyReadTimeoutSeconds);
    while (body.size() < content_length) {
        const std::size_t wanted = std::min(sizeof(chunk), content_length - body.size());
        const ssize_t received = receive_before(fd, chunk, wanted, body_deadline);
        if (received <= 0) return ReadResult::kBadRequest;
        body.append(chunk, static_cast<std::size_t>(received));
    }
    request.body = std::move(body);
    return ReadResult::kOk;
}

bool mime_parameter_token(unsigned char c) {
    return c > 32 && c < 127 &&
           std::string_view("()<>@,;:\\\"/[]?=").find(c) == std::string_view::npos;
}

bool parse_disposition(std::string_view value, Part& part) {
    std::size_t position = 0;
    // MIME comments are whitespace between tokens, not parameter sources.
    // Scan nesting iteratively; the complete part header already has a size cap.
    const auto skip_space = [&] {
        while (position < value.size()) {
            if (value[position] == ' ' || value[position] == '\t') {
                ++position;
            } else if (value[position] == '(') {
                std::size_t depth = 1;
                ++position;
                while (position < value.size() && depth != 0) {
                    const char c = value[position++];
                    if (c == '\\') {
                        if (position == value.size()) return false;
                        ++position;
                    } else if (c == '(') {
                        ++depth;
                    } else if (c == ')') {
                        --depth;
                    }
                }
                if (depth != 0) return false;
            } else {
                break;
            }
        }
        return true;
    };
    const auto token = [&] {
        const std::size_t start = position;
        while (position < value.size() && mime_parameter_token(value[position])) ++position;
        return value.substr(start, position - start);
    };
    if (!skip_space() || lower(std::string(token())) != "form-data" || !skip_space()) return false;
    bool have_name = false, have_filename = false;
    while (position < value.size()) {
        if (value[position++] != ';' || !skip_space()) return false;
        const std::string name = lower(std::string(token()));
        if (name.empty() || !skip_space() || position == value.size() ||
            value[position++] != '=' || !skip_space()) return false;
        std::string parameter;
        if (position < value.size() && value[position] == '"') {
            ++position;
            bool closed = false;
            while (position < value.size()) {
                char c = value[position++];
                if (c == '"') {
                    closed = true;
                    break;
                }
                if (c == '\\') {
                    if (position == value.size()) return false;
                    c = value[position++];
                }
                parameter.push_back(c);
            }
            if (!closed) return false;
        } else {
            parameter = std::string(token());
            if (parameter.empty()) return false;
        }
        if (!skip_space() || (position < value.size() && value[position] != ';')) return false;
        if (name == "name") {
            if (have_name) return false;
            part.name = std::move(parameter);
            have_name = true;
        } else if (name == "filename") {
            if (have_filename) return false;
            part.filename = std::move(parameter);
            have_filename = true;
        } else if (name == "filename*") {
            // RFC 7578 explicitly excludes this encoding in multipart/form-data.
            return false;
        }
    }
    return have_name;
}

bool parse_part_headers(std::string_view headers, Part& part) {
    std::string key, value;
    bool have_disposition = false;
    const auto finish_header = [&] {
        if (key != "content-disposition") return true;
        if (have_disposition) return false;
        have_disposition = true;
        return parse_disposition(value, part);
    };
    std::size_t position = 0;
    while (position < headers.size()) {
        const std::size_t end = headers.find("\r\n", position);
        const std::string_view line = headers.substr(
            position, end == std::string_view::npos ? headers.size() - position : end - position);
        if (line.empty()) return false;
        for (unsigned char c : line) {
            if ((c < 32 && c != '\t') || c == 127) return false;
        }
        if (line.front() == ' ' || line.front() == '\t') {
            if (key.empty()) return false;
            // Unfold CRLF only. Keep whitespace so tokens cannot be concatenated.
            value.append(line.data(), line.size());
        } else {
            if (!finish_header()) return false;
            const std::size_t colon = line.find(':');
            if (colon == 0 || colon == std::string_view::npos) return false;
            for (unsigned char c : line.substr(0, colon)) {
                if (c < 33 || c > 126) return false;
            }
            key = lower(std::string(line.substr(0, colon)));
            value = std::string(line.substr(colon + 1));
        }
        position = end == std::string_view::npos ? headers.size() : end + 2;
    }
    return finish_header() && have_disposition;
}

MultipartResult parse_multipart(const std::string& body, const std::string& boundary,
                                std::vector<Part>& parts) {
    if (boundary.empty() || boundary.size() > 200 ||
        boundary.find('\r') != std::string::npos || boundary.find('\n') != std::string::npos) {
        return MultipartResult::kMalformed;
    }
    const std::string delimiter = "--" + boundary;
    std::size_t position = 0;
    if (body.compare(position, delimiter.size(), delimiter) != 0) {
        return MultipartResult::kMalformed;
    }
    while (position != std::string::npos) {
        position += delimiter.size();
        if (body.compare(position, 2, "--") == 0) {
            position += 2;
            if (position == body.size() ||
                (body.compare(position, 2, "\r\n") == 0 && position + 2 == body.size())) {
                return MultipartResult::kOk;
            }
            return MultipartResult::kMalformed;
        }
        if (body.compare(position, 2, "\r\n") == 0) {
            position += 2;
        } else {
            return MultipartResult::kMalformed;
        }
        const std::size_t header_end = body.find("\r\n\r\n", position);
        if (header_end == std::string::npos || header_end - position > kMaxHeaderBytes) {
            return MultipartResult::kMalformed;
        }
        const std::string headers = body.substr(position, header_end - position);
        const std::size_t content_start = header_end + 4;
        const std::string marker = "\r\n" + delimiter;
        std::size_t next_marker = body.find(marker, content_start);
        while (next_marker != std::string::npos) {
            const std::size_t suffix = next_marker + marker.size();
            const bool closing = body.compare(suffix, 2, "--") == 0 &&
                (suffix + 2 == body.size() || body.compare(suffix + 2, 2, "\r\n") == 0);
            if (body.compare(suffix, 2, "\r\n") == 0 || closing) break;
            // Binary content can contain a delimiter prefix. Only a complete
            // delimiter line ends the part; leave every other byte untouched.
            next_marker = body.find(marker, suffix);
        }
        const std::size_t next = next_marker == std::string::npos
                                     ? std::string::npos
                                     : next_marker + 2;
        if (next == std::string::npos) return MultipartResult::kMalformed;
        if (parts.size() >= kMaxMultipartParts) return MultipartResult::kTooManyParts;

        Part part;
        if (!parse_part_headers(headers, part)) return MultipartResult::kMalformed;
        const std::size_t part_size = next_marker - content_start;
        const std::size_t limit = part.filename.empty() ? kMaxFieldBytes : kMaxFileBytes;
        if (part_size > limit) return MultipartResult::kTooLarge;
        part.data = body.substr(content_start, part_size);
        parts.push_back(std::move(part));
        position = next;
    }
    return MultipartResult::kMalformed;
}

bool parameter_token(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') ||
           std::string_view("!#$%&'*+-.^_`|~").find(c) != std::string_view::npos;
}

bool extract_boundary(const std::string& content_type, std::string& boundary) {
    const std::string normalized = lower(content_type);
    const std::size_t media_end = normalized.find(';');
    std::string media = normalized.substr(0, media_end);
    while (!media.empty() && (media.front() == ' ' || media.front() == '\t')) {
        media.erase(media.begin());
    }
    while (!media.empty() && (media.back() == ' ' || media.back() == '\t')) {
        media.pop_back();
    }
    if (media != "multipart/form-data" || media_end == std::string::npos) return false;

    // Parse parameter boundaries and quoted strings rather than searching for a
    // substring that could belong to another parameter's name or value.
    bool have_boundary = false;
    std::size_t position = media_end;
    const auto skip_space = [&] {
        while (position < content_type.size() &&
               (content_type[position] == ' ' || content_type[position] == '\t')) {
            ++position;
        }
    };
    while (position < content_type.size()) {
        if (content_type[position++] != ';') return false;
        skip_space();
        const std::size_t name_start = position;
        while (position < content_type.size() && parameter_token(content_type[position])) {
            ++position;
        }
        const std::string name = normalized.substr(name_start, position - name_start);
        skip_space();
        if (name.empty() || position == content_type.size() ||
            content_type[position++] != '=') return false;
        skip_space();
        std::string value;
        if (position < content_type.size() && content_type[position] == '"') {
            ++position;
            bool closed = false;
            while (position < content_type.size()) {
                unsigned char c = content_type[position++];
                if (c == '"') {
                    closed = true;
                    break;
                }
                if (c == '\\') {
                    if (position == content_type.size()) return false;
                    c = content_type[position++];
                }
                if ((c < 32 && c != '\t') || c == 127) return false;
                value.push_back(static_cast<char>(c));
            }
            if (!closed) return false;
        } else {
            const std::size_t value_start = position;
            while (position < content_type.size() && parameter_token(content_type[position])) {
                ++position;
            }
            if (position == value_start) return false;
            value = content_type.substr(value_start, position - value_start);
        }
        skip_space();
        if (position < content_type.size() && content_type[position] != ';') return false;
        if (name == "boundary") {
            if (have_boundary || value.empty() || value.size() > 200) return false;
            for (unsigned char c : value) {
                if (c < 32 || c >= 127) return false;
            }
            boundary = std::move(value);
            have_boundary = true;
        }
    }
    return have_boundary;
}

bool send_all(int fd, const char* data, std::size_t size) {
    while (size > 0) {
        const ssize_t sent = send(fd, data, size, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) return false;
        data += sent;
        size -= static_cast<std::size_t>(sent);
    }
    return true;
}

class ConnectionGuard {
public:
    ConnectionGuard(int socket, std::shared_ptr<detail::HttpRuntimeState> runtime)
        : fd_(socket), runtime_(std::move(runtime)) {}
    ConnectionGuard(const ConnectionGuard&) = delete;
    ConnectionGuard& operator=(const ConnectionGuard&) = delete;
    ~ConnectionGuard() {
        close(fd_);
        runtime_->active_connections.fetch_sub(1, std::memory_order_relaxed);
    }

private:
    int fd_;
    std::shared_ptr<detail::HttpRuntimeState> runtime_;
};

void handle_connection(int fd, RequestHandler handler, const std::string& auth_token,
                       std::shared_ptr<detail::HttpRuntimeState> runtime) {
    ConnectionGuard guard(fd, runtime);
    timeval timeout{kSocketTimeoutSeconds, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    try {
        Request request;
        const ReadResult result = read_request(fd, request, auth_token, runtime);
        if (result == ReadResult::kBadRequest) {
            send_response(fd, 400, "Bad Request", "application/json",
                          "{\"error\":\"malformed request\"}");
            return;
        }
        if (result == ReadResult::kPayloadTooLarge) {
            send_response(fd, 413, "Payload Too Large", "application/json",
                          "{\"error\":\"request too large\"}");
            return;
        }
        if (result == ReadResult::kUnauthorized) {
            send_response(fd, 401, "Unauthorized", "application/json",
                          "{\"error\":\"authentication required\"}",
                          "WWW-Authenticate: Bearer\r\n");
            return;
        }
        if (result == ReadResult::kServerBusy) {
            send_response(fd, 503, "Service Unavailable", "application/json",
                          "{\"error\":\"request memory limit reached\"}");
            return;
        }
        handler(fd, request);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[vc_serve] request exception: %s\n", error.what());
        try {
            send_response(fd, 500, "Internal Server Error", "application/json",
                          "{\"error\":\"internal server error\"}");
        } catch (...) {
        }
    } catch (...) {
        std::fprintf(stderr, "[vc_serve] request exception\n");
        try {
            send_response(fd, 500, "Internal Server Error", "application/json",
                          "{\"error\":\"internal server error\"}");
        } catch (...) {
        }
    }
}

}  // namespace

Request::Request() = default;

void Request::hold_body_reservation(std::shared_ptr<detail::HttpRuntimeState> runtime,
                                    std::size_t bytes) noexcept {
    runtime_ = std::move(runtime);
    reserved_body_bytes_ = bytes;
}

Request::~Request() {
    if (runtime_ && reserved_body_bytes_ != 0) {
        runtime_->inflight_body_bytes.fetch_sub(reserved_body_bytes_, std::memory_order_relaxed);
    }
}

HttpTransport::HttpTransport(HttpTransportConfig config)
    : config_(std::move(config)), runtime_(std::make_shared<detail::HttpRuntimeState>()) {}

bool valid_ipv4_address(const std::string& address) {
    in_addr parsed{};
    return inet_pton(AF_INET, address.c_str(), &parsed) == 1;
}

void send_response(int fd, int status, const char* text, const std::string& content_type,
                   const std::string& body, const std::string& extra_headers) {
    std::ostringstream response;
    response << "HTTP/1.1 " << status << " " << text << "\r\nContent-Type: "
             << content_type << "\r\nContent-Length: " << body.size()
             << "\r\nX-Content-Type-Options: nosniff\r\nCache-Control: no-store\r\n"
             << extra_headers << "Connection: close\r\n\r\n";
    const std::string headers = response.str();
    if (!send_all(fd, headers.data(), headers.size())) return;
    if (!body.empty()) send_all(fd, body.data(), body.size());
}

void send_json(int fd, const std::string& json) {
    send_response(fd, 200, "OK", "application/json", json);
}

bool request_parts(int fd, const Request& request, std::vector<Part>& parts) {
    std::string boundary;
    if (!extract_boundary(request.content_type, boundary)) {
        send_response(fd, 400, "Bad Request", "application/json",
                      "{\"error\":\"invalid multipart boundary\"}");
        return false;
    }
    switch (parse_multipart(request.body, boundary, parts)) {
        case MultipartResult::kOk:
            return true;
        case MultipartResult::kTooLarge:
            send_response(fd, 413, "Payload Too Large", "application/json",
                          "{\"error\":\"multipart field or file too large\"}");
            return false;
        case MultipartResult::kTooManyParts:
            send_response(fd, 413, "Payload Too Large", "application/json",
                          "{\"error\":\"too many multipart parts\"}");
            return false;
        case MultipartResult::kMalformed:
            send_response(fd, 400, "Bad Request", "application/json",
                          "{\"error\":\"malformed multipart body\"}");
            return false;
    }
    return false;
}

int HttpTransport::serve(RequestHandler handler, ListeningHandler on_listening) {
    std::signal(SIGPIPE, SIG_IGN);
    const int server = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (server < 0) {
        std::fprintf(stderr, "socket failed: %s\n", std::strerror(errno));
        return 1;
    }

    const int option = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    inet_pton(AF_INET, config_.host.c_str(), &address.sin_addr);
    address.sin_port = htons(config_.port);
    if (bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
        std::fprintf(stderr, "bind failed on %s:%d: %s\n", config_.host.c_str(), config_.port,
                     std::strerror(errno));
        close(server);
        return 1;
    }
    if (listen(server, kMaxConnections) < 0) {
        std::fprintf(stderr, "listen failed: %s\n", std::strerror(errno));
        close(server);
        return 1;
    }
    if (on_listening) on_listening();

    for (;;) {
        const int fd = accept4(server, nullptr, nullptr, SOCK_CLOEXEC);
        if (fd < 0) {
            if (errno == EINTR) continue;
            std::fprintf(stderr, "accept failed: %s\n", std::strerror(errno));
            close(server);
            return 1;
        }
        const int previous = runtime_->active_connections.fetch_add(1, std::memory_order_relaxed);
        if (previous >= kMaxConnections) {
            runtime_->active_connections.fetch_sub(1, std::memory_order_relaxed);
            send_response(fd, 503, "Service Unavailable", "application/json",
                          "{\"error\":\"connection limit reached\"}");
            close(fd);
            continue;
        }
        try {
            std::thread worker(handle_connection, fd, handler, config_.auth_token, runtime_);
            worker.detach();
        } catch (const std::exception& error) {
            runtime_->active_connections.fetch_sub(1, std::memory_order_relaxed);
            close(fd);
            std::fprintf(stderr, "cannot start request worker: %s\n", error.what());
        }
    }
}

}  // namespace resonantia::web
