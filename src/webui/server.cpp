// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/webui/server.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

namespace voxmutatio::webui {

void HttpResponse::set_json(const std::string& json_str) {
    headers["Content-Type"] = "application/json";
    body.assign(json_str.begin(), json_str.end());
}

void HttpResponse::set_text(const std::string& text) {
    headers["Content-Type"] = "text/plain";
    body.assign(text.begin(), text.end());
}

void HttpResponse::set_file(const std::string& file_path,
                            const std::string& content_type) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        status_code = 404;
        set_text("File not found");
        return;
    }
    
    headers["Content-Type"] = content_type;
    body.assign(std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>());
}

namespace {

// Simple HTTP request parser
bool parse_http_request(const std::vector<uint8_t>& raw, HttpRequest& req) {
    if (raw.empty()) {
        return false;
    }
    
    std::string raw_str(raw.begin(), raw.end());
    std::istringstream stream(raw_str);
    
    // Parse request line
    std::string request_line;
    std::getline(stream, request_line);
    
    // Method
    size_t space_pos = request_line.find(' ');
    if (space_pos == std::string::npos) {
        return false;
    }
    
    std::string method_str = request_line.substr(0, space_pos);
    if (method_str == "GET") {
        req.method = HttpMethod::kGet;
    } else if (method_str == "POST") {
        req.method = HttpMethod::kPost;
    } else if (method_str == "PUT") {
        req.method = HttpMethod::kPut;
    } else if (method_str == "DELETE") {
        req.method = HttpMethod::kDelete;
    } else if (method_str == "OPTIONS") {
        req.method = HttpMethod::kOptions;
    } else {
        return false;
    }
    
    // Path
    size_t path_start = space_pos + 1;
    size_t path_end = request_line.find(' ', path_start);
    if (path_end == std::string::npos) {
        return false;
    }
    
    std::string full_path = request_line.substr(path_start, 
                                                path_end - path_start);
    
    // Parse query string
    size_t query_pos = full_path.find('?');
    if (query_pos != std::string::npos) {
        req.path = full_path.substr(0, query_pos);
        std::string query = full_path.substr(query_pos + 1);
        
        // Parse query parameters
        std::istringstream query_stream(query);
        std::string param;
        while (std::getline(query_stream, param, '&')) {
            size_t eq_pos = param.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = param.substr(0, eq_pos);
                std::string value = param.substr(eq_pos + 1);
                req.query_params[key] = value;
            }
        }
    } else {
        req.path = full_path;
    }
    
    // Parse headers
    std::string header_line;
    while (std::getline(stream, header_line) && !header_line.empty()) {
        size_t colon_pos = header_line.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = header_line.substr(0, colon_pos);
            std::string value = header_line.substr(colon_pos + 2);  // Skip ": "
            req.headers[key] = value;
        }
    }
    
    // Parse body (for POST/PUT requests)
    if (req.method == HttpMethod::kPost || req.method == HttpMethod::kPut) {
        std::string body_str((std::istreambuf_iterator<char>(stream)),
                            std::istreambuf_iterator<char>());
        req.body.assign(body_str.begin(), body_str.end());
    }
    
    return true;
}

std::string format_http_response(const HttpResponse& resp) {
    std::ostringstream oss;
    
    // Status line
    oss << "HTTP/1.1 " << resp.status_code << " ";
    switch (resp.status_code) {
        case 200: oss << "OK"; break;
        case 201: oss << "Created"; break;
        case 400: oss << "Bad Request"; break;
        case 404: oss << "Not Found"; break;
        case 500: oss << "Internal Server Error"; break;
        default: oss << "Unknown"; break;
    }
    oss << "\r\n";
    
    // Headers
    for (const auto& [key, value] : resp.headers) {
        oss << key << ": " << value << "\r\n";
    }
    oss << "Content-Length: " << resp.body.size() << "\r\n";
    oss << "Connection: close\r\n";
    oss << "\r\n";
    
    // Body
    oss.write(reinterpret_cast<const char*>(resp.body.data()),
              static_cast<std::streamsize>(resp.body.size()));
    
    return oss.str();
}

}  // namespace

bool WebServer::init(const WebServerConfig& config) {
    config_ = config;
    
    // TODO: Initialize TCP listener socket
    // For now, stub implementation
    
    return true;
}

void WebServer::register_route(HttpMethod method, const std::string& path,
                               RouteHandler handler) {
    std::string key;
    switch (method) {
        case HttpMethod::kGet: key = "GET"; break;
        case HttpMethod::kPost: key = "POST"; break;
        case HttpMethod::kPut: key = "PUT"; break;
        case HttpMethod::kDelete: key = "DELETE"; break;
        case HttpMethod::kOptions: key = "OPTIONS"; break;
    }
    key += " " + path;
    routes_[key] = std::move(handler);
}

void WebServer::start() {
    std::cout << "Starting WebServer on " << config_.host 
              << ":" << config_.port << std::endl;
    
    running_ = true;
    
    // TODO: Implement actual HTTP server with:
    // 1. TCP listener socket
    // 2. Connection accept loop
    // 3. Request parsing
    // 4. Route matching
    // 5. Response formatting
    
    // Stub: print registered routes
    std::cout << "Registered routes:" << std::endl;
    for (const auto& [key, _] : routes_) {
        std::cout << "  " << key << std::endl;
    }
    
    std::cout << "Server running. Press Ctrl+C to stop." << std::endl;
    
    // Block here (stub)
    while (running_) {
        // TODO: accept connections and handle requests
    }
}

void WebServer::stop() {
    running_ = false;
    std::cout << "Server stopped." << std::endl;
}

void register_vc_api(WebServer& server) {
    // POST /api/convert - Voice conversion endpoint
    server.register_route(HttpMethod::kPost, "/api/convert",
        [](const HttpRequest& req) -> HttpResponse {
            HttpResponse resp;
            
            // Parse JSON request body
            // TODO: Parse input_audio, speaker_id, f0_up_key, etc.
            
            resp.set_json(R"({
                "status": "ok",
                "message": "Voice conversion API endpoint (stub)"
            })");
            
            return resp;
        });
    
    // GET /api/models - List available models
    server.register_route(HttpMethod::kGet, "/api/models",
        [](const HttpRequest& req) -> HttpResponse {
            HttpResponse resp;
            resp.set_json(R"({
                "models": []
            })");
            return resp;
        });
    
    // GET /api/config - Get server configuration
    server.register_route(HttpMethod::kGet, "/api/config",
        [](const HttpRequest& req) -> HttpResponse {
            HttpResponse resp;
            resp.set_json(R"({
                "version": "0.1.0",
                "device": "cuda"
            })");
            return resp;
        });
}

}  // namespace voxmutatio::webui
