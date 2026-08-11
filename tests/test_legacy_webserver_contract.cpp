// Copyright 2026 Resonantia Authors.
// Licensed under the MIT License.

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include "voxmutatio/webui/server.h"

namespace {

using voxmutatio::webui::WebServer;
using voxmutatio::webui::WebServerConfig;

static_assert(std::is_default_constructible_v<WebServer>);
static_assert(std::is_destructible_v<WebServer>);
static_assert(std::is_move_constructible_v<WebServer>);
static_assert(std::is_move_assignable_v<WebServer>);
static_assert(!std::is_copy_constructible_v<WebServer>);
static_assert(!std::is_copy_assignable_v<WebServer>);

TEST_CASE("legacy WebServer remains source-compatible during deprecation",
          "[legacy-webserver][contract]") {
    WebServerConfig config;
    config.host = "127.0.0.1";
    config.port = 18080;

    WebServer server;
    REQUIRE(server.init(config));
    CHECK_FALSE(server.is_running());

    WebServer moved(std::move(server));
    WebServer assigned;
    assigned = std::move(moved);
    voxmutatio::webui::register_vc_api(assigned);
    CHECK_FALSE(assigned.is_running());
}

TEST_CASE("legacy WebServer start fails fast with vc_serve migration guidance",
          "[legacy-webserver][contract]") {
    WebServer server;
    REQUIRE(server.init(WebServerConfig{}));

    bool threw_logic_error = false;
    try {
        server.start();
    } catch (const std::logic_error& error) {
        threw_logic_error = true;
        CHECK(std::string(error.what()).find("vc_serve") != std::string::npos);
    } catch (...) {
        FAIL("legacy WebServer::start() threw an unexpected exception type");
    }

    CHECK(threw_logic_error);
    CHECK_FALSE(server.is_running());
    server.stop();
    CHECK_FALSE(server.is_running());
}

}  // namespace

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
