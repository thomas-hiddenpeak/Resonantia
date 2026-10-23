// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/training/generator_trainer.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#include "voxmutatio/io/safetensors.h"
#include "checked_weights.h"
#include "export_weights.h"

namespace voxmutatio::training {

namespace ag = voxmutatio::autograd;

namespace {

constexpr int kGin = 256;

// Reconstruct weight_norm weight: weight[o,inner] = g[o]*v[o,inner]/||v[o]||.
std::vector<float> reconstruct_wn(const io::SafetensorsLoader& L, const std::string& name,
                                  int out_dim, int inner) {
  return detail::reconstruct_weight_norm(L, name, out_dim, inner);
}

std::vector<float> load_plain(const io::SafetensorsLoader& L, const std::string& name,
                              int count) {
  return detail::load_f32(L, name, count);
}

}  // namespace

bool GeneratorTrainer::init(const std::string& path, int speaker_id) {
  // A failed re-initialization must never leave the previous checkpoint usable.
  // Reset before any I/O, and reset again if loading fails after partial GPU
  // allocation.
  *this = GeneratorTrainer{};
  io::SafetensorsLoader L;
  if (!L.load(path)) {
    fprintf(stderr, "GeneratorTrainer: cannot load %s\n", path.c_str());
    return false;
  }
  try {
  const float* speaker_embeddings = detail::require_speaker_embedding(L, speaker_id, kGin);
  const std::string P = "model.dec.";

  auto param = [&](std::vector<float> host, std::vector<int> shape) {
    auto t = ag::Tensor::from_host(host, std::move(shape), true);
    params_.push_back(t);
    return t;
  };

  // conv_pre (plain), fold cond(g) into bias.
  conv_pre_w_ = param(load_plain(L, P + "conv_pre.weight", 512 * 192 * 7), {512, 192, 7});
  {
    auto pre_b = load_plain(L, P + "conv_pre.bias", 512);
    // cond(g): Conv1d(256,512,1). g = emb_g[sid].
    const float* cw = detail::require_f32(L, P + "cond.weight", 512 * kGin);
    const float* cb = detail::require_f32(L, P + "cond.bias", 512);
    cond_g_init_.assign(512, 0.0f);
    for (int co = 0; co < 512; ++co) {
      float s = cb[co];
      for (int ci = 0; ci < kGin; ++ci)
        s += cw[co * kGin + ci] * speaker_embeddings[speaker_id * kGin + ci];
      if (!std::isfinite(s)) throw std::runtime_error("non-finite folded speaker condition");
      cond_g_init_[co] = s;
      pre_b[co] += s;
    }
    conv_pre_b_ = param(pre_b, {512});
  }

  const int rates[4] = {10, 10, 2, 2}, kk[4] = {16, 16, 4, 4};
  int C = 512;
  for (int i = 0; i < 4; ++i) {
    int Cout = C / 2, k = kk[i];
    // ups: ConvTranspose1d weight_norm [Cin, Cout, k].
    ups_w_[i] = param(reconstruct_wn(L, P + "ups." + std::to_string(i), C, Cout * k),
                      {C, Cout, k});
    ups_b_[i] = param(load_plain(L, P + "ups." + std::to_string(i) + ".bias", Cout), {Cout});
    // noise_convs (plain).
    int nk = (i < 3) ? 0 : 1;
    if (i < 3) { int sf = 1; for (int j = i + 1; j < 4; ++j) sf *= rates[j]; nk = sf * 2; }
    noise_w_[i] = param(load_plain(L, P + "noise_convs." + std::to_string(i) + ".weight", Cout * nk),
                        {Cout, 1, nk});
    noise_b_[i] = param(load_plain(L, P + "noise_convs." + std::to_string(i) + ".bias", Cout), {Cout});
    C = Cout;
  }

  // resblocks: 12 blocks, each convs1/convs2 x3 (weight_norm).
  const int rbk[3] = {3, 7, 11};
  int rb_ch[4] = {256, 128, 64, 32};
  for (int i = 0; i < 4; ++i) {
    int ch = rb_ch[i];
    for (int jr = 0; jr < 3; ++jr) {
      int idx = i * 3 + jr, kernel = rbk[jr];
      std::string rp = P + "resblocks." + std::to_string(idx) + ".";
      for (int j = 0; j < 3; ++j) {
        c1w_[idx][j] = param(reconstruct_wn(L, rp + "convs1." + std::to_string(j), ch, ch * kernel),
                             {ch, ch, kernel});
        c1b_[idx][j] = param(load_plain(L, rp + "convs1." + std::to_string(j) + ".bias", ch), {ch});
        c2w_[idx][j] = param(reconstruct_wn(L, rp + "convs2." + std::to_string(j), ch, ch * kernel),
                             {ch, ch, kernel});
        c2b_[idx][j] = param(load_plain(L, rp + "convs2." + std::to_string(j) + ".bias", ch), {ch});
      }
    }
  }

  conv_post_w_ = param(load_plain(L, P + "conv_post.weight", 32 * 7), {1, 32, 7});
  initialized_ = true;
  return true;
  } catch (const std::exception& error) {
    *this = GeneratorTrainer{};
    fprintf(stderr, "GeneratorTrainer: invalid checkpoint: %s\n", error.what());
    return false;
  }
}

ag::Tensor GeneratorTrainer::resblock(const ag::Tensor& xn, int idx, int C, int L,
                                      int kernel, const int* dil) {
  ag::Tensor x = xn;
  for (int j = 0; j < 3; ++j) {
    auto xt = ag::leaky_relu(x, 0.1f);
    int pad1 = dil[j] * (kernel - 1) / 2;
    xt = ag::conv1d(xt, c1w_[idx][j], c1b_[idx][j], C, L, C, kernel, 1, pad1, dil[j], 1);
    xt = ag::leaky_relu(xt, 0.1f);
    int pad2 = (kernel - 1) / 2;
    xt = ag::conv1d(xt, c2w_[idx][j], c2b_[idx][j], C, L, C, kernel, 1, pad2, 1, 1);
    x = ag::add(x, xt);
  }
  return x;
}

ag::Tensor GeneratorTrainer::decode(const ag::Tensor& z, const ag::Tensor& har, int T) {
  if (!initialized_) {
    throw std::logic_error("GeneratorTrainer is not initialized");
  }
  const int rates[4] = {10, 10, 2, 2}, kk[4] = {16, 16, 4, 4};
  const int rbk[3] = {3, 7, 11};
  const int rbd[3][3] = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
  int harlen = T * 400;

  auto x = ag::conv1d(z, conv_pre_w_, conv_pre_b_, 192, T, 512, 7, 1, 3, 1, 1);
  int C = 512, L = T;
  for (int i = 0; i < 4; ++i) {
    x = ag::leaky_relu(x, 0.1f);
    int Cout = C / 2, u = rates[i], k = kk[i], pad = (k - u) / 2;
    auto up = ag::conv_transpose1d(x, ups_w_[i], ups_b_[i], C, L, Cout, k, u, pad);
    int Lout = (L - 1) * u - 2 * pad + k;

    int nk, nstride, npad;
    if (i < 3) { int sf = 1; for (int j = i + 1; j < 4; ++j) sf *= rates[j];
                 nk = sf * 2; nstride = sf; npad = sf / 2; }
    else { nk = 1; nstride = 1; npad = 0; }
    auto xsrc = ag::conv1d(har, noise_w_[i], noise_b_[i], 1, harlen, Cout, nk, nstride, npad, 1, 1);
    auto xn = ag::add(up, xsrc);

    ag::Tensor acc;
    for (int jr = 0; jr < 3; ++jr) {
      auto rb = resblock(xn, i * 3 + jr, Cout, Lout, rbk[jr], rbd[jr]);
      acc = (jr == 0) ? rb : ag::add(acc, rb);
    }
    x = ag::scale(acc, 1.0f / 3.0f);
    C = Cout;
    L = Lout;
  }
  x = ag::leaky_relu(x, 0.1f);
  x = ag::conv1d(x, conv_post_w_, ag::Tensor{}, C, L, 1, 7, 1, 3, 1, 1);
  return ag::tanh_op(x);
}

namespace {

using Entry = detail::ExportTensor;
using detail::derive_wn;

bool write_safetensors(const std::string& path, const std::vector<Entry>& entries) {
  // Build JSON header with contiguous data offsets.
  std::string json = "{";
  std::size_t offset = 0;
  bool first = true;
  for (const auto& e : entries) {
    std::size_t n = 1;
    for (int s : e.shape) {
      if (s <= 0 || n > std::numeric_limits<std::size_t>::max() / s) return false;
      n *= static_cast<std::size_t>(s);
    }
    if (n != e.data.size() || n > std::numeric_limits<std::size_t>::max() / sizeof(float))
      return false;
    for (float value : e.data) if (!std::isfinite(value)) return false;
    std::size_t bytes = n * sizeof(float);
    if (offset > std::numeric_limits<std::size_t>::max() - bytes) return false;
    if (!first) json += ",";
    first = false;
    json += "\"" + e.name + "\":{\"dtype\":\"F32\",\"shape\":[";
    for (std::size_t i = 0; i < e.shape.size(); ++i) {
      if (i) json += ",";
      json += std::to_string(e.shape[i]);
    }
    json += "],\"data_offsets\":[" + std::to_string(offset) + "," +
            std::to_string(offset + bytes) + "]}";
    offset += bytes;
  }
  json += "}";
  json.append((8 - json.size() % 8) % 8, ' ');

  static_assert(std::endian::native == std::endian::little,
                "safetensors F32 export requires a little-endian host");
  std::string temporary = path + ".tmp.XXXXXX";
  std::vector<char> temporary_name(temporary.begin(), temporary.end());
  temporary_name.push_back('\0');
  const int fd = mkstemp(temporary_name.data());
  if (fd < 0) return false;
  std::FILE* f = fdopen(fd, "wb");
  if (!f) {
    close(fd);
    unlink(temporary_name.data());
    return false;
  }

  const auto write_exact = [&](const void* data, std::size_t bytes) {
    return bytes == 0 || std::fwrite(data, 1, bytes, f) == bytes;
  };
  const std::uint64_t hlen = json.size();
  unsigned char header_length[8];
  for (int i = 0; i < 8; ++i)
    header_length[i] = static_cast<unsigned char>((hlen >> (8 * i)) & 0xffu);

  bool ok = write_exact(header_length, sizeof(header_length)) &&
            write_exact(json.data(), json.size());
  for (const auto& e : entries) {
    if (e.data.size() > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        !write_exact(e.data.data(), e.data.size() * sizeof(float))) {
      ok = false;
      break;
    }
  }
  if (ok && std::fflush(f) != 0) ok = false;
  if (std::ferror(f)) ok = false;
  if (ok && fsync(fd) != 0) ok = false;
  if (std::fclose(f) != 0) ok = false;
  if (!ok || rename(temporary_name.data(), path.c_str()) != 0) {
    unlink(temporary_name.data());
    return false;
  }
  return true;
}

}  // namespace

bool GeneratorTrainer::export_model(const std::string& src, const std::string& out) {
  return export_with_weights(src, out, {});
}

bool GeneratorTrainer::export_with_weights(
    const std::string& src, const std::string& out,
    const std::vector<detail::ExportTensor>& additional) try {
  if (!initialized_) return false;
  io::SafetensorsLoader L;
  if (!L.load(src)) return false;

  // Copy all source tensors.
  std::vector<Entry> entries;
  std::unordered_map<std::string, int> idx;
  for (const auto& name : L.tensor_names()) {
    const io::Tensor* t = L.get_tensor(name);
    if (t->dtype != "F32" || t->data_nbytes % sizeof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(L.data(name)) % alignof(float) != 0)
      return false;
    for (auto dimension : t->shape)
      if (dimension <= 0 || dimension > std::numeric_limits<int>::max()) return false;
    std::size_t n = t->data_nbytes / sizeof(float);
    const auto* p = reinterpret_cast<const float*>(L.data(name));
    std::vector<int> shape(t->shape.begin(), t->shape.end());
    idx[name] = static_cast<int>(entries.size());
    entries.push_back({name, std::move(shape), std::vector<float>(p, p + n)});
  }
  auto set = [&](const std::string& name, std::vector<int> shape, std::vector<float> data) {
    auto it = idx.find(name);
    if (it != idx.end()) { entries[it->second].shape = std::move(shape); entries[it->second].data = std::move(data); }
    else { idx[name] = (int)entries.size(); entries.push_back({name, std::move(shape), std::move(data)}); }
  };
  const std::string P = "model.dec.";

  // conv_pre: remove folded cond(g) from the trained bias.
  set(P + "conv_pre.weight", {512, 192, 7}, conv_pre_w_.to_host());
  {
    auto b = conv_pre_b_.to_host();
    for (int i = 0; i < 512; ++i) b[i] -= cond_g_init_[i];
    set(P + "conv_pre.bias", {512}, b);
  }

  const int rates[4] = {10, 10, 2, 2}, kk[4] = {16, 16, 4, 4};
  int C = 512;
  for (int i = 0; i < 4; ++i) {
    int Cout = C / 2, k = kk[i];
    std::vector<float> g, v;
    derive_wn(ups_w_[i].to_host(), C, Cout * k, g, v);
    set(P + "ups." + std::to_string(i) + ".weight_g", {C, 1, 1}, g);
    set(P + "ups." + std::to_string(i) + ".weight_v", {C, Cout, k}, v);
    set(P + "ups." + std::to_string(i) + ".bias", {Cout}, ups_b_[i].to_host());
    int nk = (i < 3) ? 0 : 1;
    if (i < 3) { int sf = 1; for (int j = i + 1; j < 4; ++j) sf *= rates[j]; nk = sf * 2; }
    set(P + "noise_convs." + std::to_string(i) + ".weight", {Cout, 1, nk}, noise_w_[i].to_host());
    set(P + "noise_convs." + std::to_string(i) + ".bias", {Cout}, noise_b_[i].to_host());
    C = Cout;
  }

  const int rbk[3] = {3, 7, 11};
  int rb_ch[4] = {256, 128, 64, 32};
  for (int i = 0; i < 4; ++i) {
    int ch = rb_ch[i];
    for (int jr = 0; jr < 3; ++jr) {
      int id = i * 3 + jr, kernel = rbk[jr];
      std::string rp = P + "resblocks." + std::to_string(id) + ".";
      for (int j = 0; j < 3; ++j) {
        std::vector<float> g, v;
        derive_wn(c1w_[id][j].to_host(), ch, ch * kernel, g, v);
        set(rp + "convs1." + std::to_string(j) + ".weight_g", {ch, 1, 1}, g);
        set(rp + "convs1." + std::to_string(j) + ".weight_v", {ch, ch, kernel}, v);
        set(rp + "convs1." + std::to_string(j) + ".bias", {ch}, c1b_[id][j].to_host());
        derive_wn(c2w_[id][j].to_host(), ch, ch * kernel, g, v);
        set(rp + "convs2." + std::to_string(j) + ".weight_g", {ch, 1, 1}, g);
        set(rp + "convs2." + std::to_string(j) + ".weight_v", {ch, ch, kernel}, v);
        set(rp + "convs2." + std::to_string(j) + ".bias", {ch}, c2b_[id][j].to_host());
      }
    }
  }
  set(P + "conv_post.weight", {1, 32, 7}, conv_post_w_.to_host());
  for (const auto& tensor : additional) set(tensor.name, tensor.shape, tensor.data);

  return write_safetensors(out, entries);
} catch (const std::exception& error) {
  std::fprintf(stderr, "GeneratorTrainer: export failed: %s\n", error.what());
  return false;
}

}  // namespace voxmutatio::training
