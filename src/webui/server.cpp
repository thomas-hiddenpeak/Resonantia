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

bool parse_http_request(const std::vector<uint8_t>& raw, HttpRequest& req) {
    if (raw.empty()) return false;
    
    std::string raw_str(raw.begin(), raw.end());
    std::istringstream stream(raw_str);
    
    std::string request_line;
    std::getline(stream, request_line);
    
    size_t space_pos = request_line.find(' ');
    if (space_pos == std::string::npos) return false;
    
    std::string method_str = request_line.substr(0, space_pos);
    if (method_str == "GET") req.method = HttpMethod::kGet;
    else if (method_str == "POST") req.method = HttpMethod::kPost;
    else if (method_str == "PUT") req.method = HttpMethod::kPut;
    else if (method_str == "DELETE") req.method = HttpMethod::kDelete;
    else if (method_str == "OPTIONS") req.method = HttpMethod::kOptions;
    else return false;
    
    size_t path_start = space_pos + 1;
    size_t path_end = request_line.find(' ', path_start);
    if (path_end == std::string::npos) return false;
    
    std::string full_path = request_line.substr(path_start, path_end - path_start);
    req.path = full_path;
    
    return true;
}

}  // namespace

bool WebServer::init(const WebServerConfig& config) {
    config_ = config;
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
    while (running_) {
        // TODO: accept connections
    }
}

void WebServer::stop() {
    running_ = false;
}

void register_vc_api(WebServer& server) {
    server.register_route(HttpMethod::kPost, "/api/convert",
        [](const HttpRequest&) -> HttpResponse {
            HttpResponse resp;
            resp.set_json("{\"status\":\"ok\"}");
            return resp;
        });
    
    server.register_route(HttpMethod::kGet, "/api/models",
        [](const HttpRequest&) -> HttpResponse {
            HttpResponse resp;
            resp.set_json("{\"models\":[]}");
            return resp;
        });
}

}  // namespace voxmutatio::webui
