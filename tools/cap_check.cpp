// SPDX-FileCopyrightText: 2026 yuunnn-w
// SPDX-License-Identifier: GPL-3.0-only
// 采样候选上限的独立验证：把 engine.cpp 里决定候选规模的那段循环原样搬过来，用合成 logits 看它
// 什么时候停在 4096、什么时候扩到整表。不需要卡，也不需要模型，一条 g++ 就能跑。
//
//   g++ -O2 -std=c++11 -o cap_check tools/cap_check.cpp && ./cap_check
//
// 期望结果：常见分布停在 4096 且被裁掉的概率在 1e-7 量级；分布很平（temperature=2.0 的重尾、
// logits 全相等）时扩到整表；客户端自己给了 top_k 就按它的值截断，不扩大。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

static int select_k(std::vector<std::pair<float, int> >& cand, float t, int req_top_k, float req_top_p, int* out_limit,
                    float* out_bound_frac) {
  const int CAP = 4096;
  int k = req_top_k > 0 ? std::min((int)cand.size(), req_top_k) : std::min((int)cand.size(), CAP);
  std::vector<float> p;
  float sum = 0.0f;
  int limit = k;
  float bound = 0.0f;
  for (;;) {
    std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
    p.assign((size_t)k, 0.0f);
    sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      p[i] = std::exp((cand[i].first - cand[0].first) / t);
      sum += p[i];
    }
    limit = k;
    if (req_top_p > 0.0f && req_top_p < 1.0f && sum > 0.0f) {
      float acc = 0.0f;
      for (int i = 0; i < k; ++i) {
        acc += p[i] / sum;
        if (acc >= req_top_p) {
          limit = i + 1;
          break;
        }
      }
    }
    bool grow = false;
    int rem = (int)cand.size() - k;
    bound = 0.0f;
    if (req_top_k <= 0 && rem > 0 && sum > 0.0f) {
      float tail = (float)rem * std::exp((cand[k - 1].first - cand[0].first) / t);
      bound = tail / sum;
      grow = tail > 0.01f * sum;
    }
    if (!grow) {
      break;
    }
    k = std::min((int)cand.size(), k * 4);
  }
  *out_limit = limit;
  *out_bound_frac = bound;
  return k;
}

// 常见形状：前几名很尖，后面是一条很低的长尾
static void make_llm(std::vector<std::pair<float, int> >& c, int V) {
  for (int i = 0; i < V; ++i) {
    float x;
    if (i < 5) {
      x = 12.0f - (float)i * 1.5f;
    } else if (i < 40) {
      x = 5.0f - (float)(i - 5) * 0.25f;
    } else {
      x = -6.0f - (float)(i - 40) * 0.0009f;
    }
    c.push_back(std::make_pair(x, i));
  }
}

// 很重的低概率尾巴（top_p=1.0 的客户端最容易撞到的形状）
static void make_heavy_tail(std::vector<std::pair<float, int> >& c, int V) {
  for (int i = 0; i < V; ++i) {
    float x = 10.0f / (float)(i + 1) - 1.0f * std::log((float)(i + 1));
    c.push_back(std::make_pair(x, i));
  }
}

static void make_flat(std::vector<std::pair<float, int> >& c, int V) {
  for (int i = 0; i < V; ++i) {
    c.push_back(std::make_pair(0.0f, i));
  }
}

static void run(const char* name, std::vector<std::pair<float, int> > c, float t, int top_k, float top_p) {
  int limit = 0;
  float bound = 0.0f;
  int k = select_k(c, t, top_k, top_p, &limit, &bound);
  // 用整表再算一遍，看被裁掉的真实质量有多大
  double full = 0.0;
  std::vector<double> e(c.size());
  float mx = c[0].first;
  for (size_t i = 0; i < c.size(); ++i) {
    e[i] = std::exp((c[i].first - mx) / (double)t);
    full += e[i];
  }
  double kept = 0.0;
  for (int i = 0; i < k; ++i) {
    kept += e[(size_t)i];
  }
  printf("%-22s t=%.2f top_k=%-5d top_p=%.2f -> 候选=%-7d 核采样截断=%-7d 上界/保留=%.3e 真实裁掉=%.3e\n",
         name, t, top_k, top_p, k, limit, bound, (full - kept) / full);
}

int main() {
  const int V = 131072;  // 本模型词表大小
  std::vector<std::pair<float, int> > c;
  c.reserve((size_t)V);

  make_llm(c, V);
  run("尖峰长尾/默认采样", c, 1.0f, 0, 0.95f);
  c.clear();
  make_llm(c, V);
  run("尖峰长尾/top_p=1.0", c, 1.0f, 0, 1.0f);
  c.clear();
  make_heavy_tail(c, V);
  run("重尾/top_p=1.0", c, 1.0f, 0, 1.0f);
  c.clear();
  make_heavy_tail(c, V);
  run("重尾/temperature=2.0", c, 2.0f, 0, 0.95f);
  c.clear();
  make_flat(c, V);
  run("logits 全相等", c, 1.0f, 0, 1.0f);
  c.clear();
  make_flat(c, V);
  run("平坦但客户端 top_k=8", c, 1.0f, 8, 1.0f);
  return 0;
}
