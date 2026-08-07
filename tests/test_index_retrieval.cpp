/**
 * @file test_index_retrieval.cpp
 * @brief Feature index retrieval test using real HuBERT features.
 *
 * Builds a flat index from real HuBERT features (derived from real speech),
 * then verifies RVC-style weighted k-NN retrieval: querying with a vector
 * that exists in the database returns that vector (nearest dist ~0).
 */
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include "voxmutatio/index/cuda_flat_index.h"

namespace {

std::vector<float> load_bin(const std::string& path, int& rows, int& cols) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return {};
    int32_t nd; f.read(reinterpret_cast<char*>(&nd), 4);
    std::vector<int> shape;
    int64_t tot = 1;
    for (int i = 0; i < nd; ++i) { int32_t s; f.read(reinterpret_cast<char*>(&s), 4); shape.push_back(s); tot *= s; }
    std::vector<float> d(tot);
    f.read(reinterpret_cast<char*>(d.data()), tot * 4);
    rows = shape.size() > 0 ? shape[0] : 0;
    cols = shape.size() > 1 ? shape[1] : 0;
    return d;
}

// Write index binary: [int64 ntotal][int64 dim][float data]
bool write_index(const std::string& path, const std::vector<float>& data,
                 int64_t ntotal, int64_t dim) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.write(reinterpret_cast<const char*>(&ntotal), 8);
    f.write(reinterpret_cast<const char*>(&dim), 8);
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * 4);
    return f.good();
}

double cosine(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
    return dot / (std::sqrt(na * nb) + 1e-12);
}

}  // namespace

TEST_CASE("Feature index weighted retrieval (real HuBERT)", "[index][retrieval]") {
    using namespace voxmutatio::index;

    // Load real HuBERT features [T, 768]
    int T = 0, D = 0;
    auto feats = load_bin("../tests/fixtures/hubert_ref_layer12.bin", T, D);
    REQUIRE(!feats.empty());
    REQUIRE(D == 768);
    std::cout << "Database: " << T << " x " << D << std::endl;

    // Build index binary from the features
    std::string idx_path = "../tests/fixtures/test_index.bin";
    REQUIRE(write_index(idx_path, feats, T, D));

    CudaFlatIndex index;
    REQUIRE(index.load(idx_path));
    REQUIRE(index.total_vectors() == T);
    REQUIRE(index.dim() == 768);

    // Query with the first 10 database vectors themselves.
    int nq = 10;
    auto retrieved = index.retrieve_weighted(feats.data(), nq, 8, 768);
    REQUIRE(static_cast<int>(retrieved.size()) == nq * 768);

    // Since each query IS in the database, the nearest neighbor has distance 0,
    // its weight (1/eps)^2 dominates, so retrieved ~= the query vector.
    double min_cos = 1.0;
    for (int i = 0; i < nq; ++i) {
        double c = cosine(&retrieved[i * 768], &feats[i * 768], 768);
        min_cos = std::min(min_cos, c);
    }
    std::cout << "Min cosine(retrieved, query): " << min_cos << std::endl;
    CHECK(min_cos > 0.999);

    // Verify blend at rate 1.0 gives retrieved, rate 0.0 gives original.
    std::vector<float> blended(nq * 768);
    blend_features(blended.data(), feats.data(), retrieved.data(), nq, 768, 0.0);
    CHECK(cosine(blended.data(), feats.data(), nq * 768) > 0.9999);
    blend_features(blended.data(), feats.data(), retrieved.data(), nq, 768, 1.0);
    CHECK(cosine(blended.data(), retrieved.data(), nq * 768) > 0.9999);

    std::remove(idx_path.c_str());
}
