// FLS 逐星边缘（P1'）的行表构造：FlsWindow 与 gtest 共用，避免两边各写一份。
#pragma once

#include <cstdint>
#include <vector>

// 每星的行表（device 上传前为 host 向量）：
//   d00_*: Λ00 的贡献（CSR；每项一个 6 向量；src=该 6 元组的 A_val 起始、c=其首列 id）
//   d11_*: Λ11 的贡献（同上；含 dyn(0→1) 的节点 1；成对口径不含 dyn(1→2)）
//   pr_s1/pr_s0: Λ10 的配对（每星 6 对：dyn(0→1) 各行 n1 与 n0 的 6 元组起始）
struct FlsLists {
    int n = 0;
    std::vector<int64_t> d00_p{0}, d00_s, d00_c;
    std::vector<int64_t> d11_p{0}, d11_s, d11_c;
    std::vector<int64_t> pr_s1, pr_s0;
};

// n 星、L 帧、E 条 ISL、G 个 GTS；off_* 为装配结构里四组行组的起始偏移
// （ISL → GTS → dyn → prior，见 fls_struct_impl 与行组布局）。
inline FlsLists fls_make_lists(int n, int L, int E, int G,
                               const std::vector<int>& isl_i, const std::vector<int>& isl_j,
                               const std::vector<int>& gts,
                               int64_t off_isl, int64_t off_gts, int64_t off_dyn, int64_t off_damp) {
    FlsLists o;
    o.n = n;
    for (int i = 0; i < n; ++i) {
        // ---- Λ00,ii：prior@0 + 关联 ISL@0 + GTS@0 + dyn(0→1) 的节点 0 ----
        for (int r = 0; r < 6; ++r) {                             // prior@0：6×6 共 6 行
            o.d00_s.push_back(off_damp + (int64_t)i * 36 + (int64_t)r * 6);
            o.d00_c.push_back((int64_t)i * 6);
        }
        for (int e = 0; e < E; ++e) {
            if (isl_i[e] == i) {
                o.d00_s.push_back(off_isl + (int64_t)e * 12);
                o.d00_c.push_back((int64_t)i * 6);
            }
            if (isl_j[e] == i) {
                o.d00_s.push_back(off_isl + (int64_t)e * 12 + 6);
                o.d00_c.push_back((int64_t)i * 6);
            }
        }
        for (int g = 0; g < G; ++g)
            if (gts[g] == i) {
                o.d00_s.push_back(off_gts + (int64_t)g * 6);
                o.d00_c.push_back((int64_t)i * 6);
            }
        if (L >= 2) {
            for (int r = 0; r < 6; ++r) {                         // dyn(0→1) 节点 0：6 行
                o.d00_s.push_back(off_dyn + (int64_t)(i * 6 + r) * 12);
                o.d00_c.push_back((int64_t)i * 6);
            }
        } else {
            o.d00_s.push_back(off_dyn + (int64_t)i * 72);         // L=1：占位行，交接核跳过最后一行
            o.d00_c.push_back((int64_t)i * 6);
        }
        o.d00_p.push_back((int64_t)o.d00_s.size());
        // ---- Λ11,ii：关联 ISL@1 + GTS@1 + dyn(0→1) 节点 1（成对口径，不含未来帧）----
        for (int e = 0; e < E; ++e) {
            if (isl_i[e] == i) {
                o.d11_s.push_back(off_isl + ((int64_t)E + e) * 12);
                o.d11_c.push_back((int64_t)(n + i) * 6);
            }
            if (isl_j[e] == i) {
                o.d11_s.push_back(off_isl + ((int64_t)E + e) * 12 + 6);
                o.d11_c.push_back((int64_t)(n + i) * 6);
            }
        }
        for (int g = 0; g < G; ++g)
            if (gts[g] == i) {
                o.d11_s.push_back(off_gts + ((int64_t)G + g) * 6);
                o.d11_c.push_back((int64_t)(n + i) * 6);
            }
        if (L >= 2) {
            for (int r = 0; r < 6; ++r) {                         // dyn(0→1) 节点 1：6 行
                o.d11_s.push_back(off_dyn + (int64_t)(i * 6 + r) * 12 + 6);
                o.d11_c.push_back((int64_t)(n + i) * 6);
            }
        }
        // 成对（0,1）口径：不把 dyn(1→2) 当作节点 1 的硬信息；未来帧不进入该边缘。
        o.d11_p.push_back((int64_t)o.d11_s.size());
        // ---- Λ10：dyn(0→1) 的 6 行（n1 与 n0 逐行配对） ----
        for (int k = 0; k < 6; ++k) {
            o.pr_s1.push_back(off_dyn + (int64_t)(i * 6 + k) * 12 + 6);
            o.pr_s0.push_back(off_dyn + (int64_t)(i * 6 + k) * 12);
        }
    }
    return o;
}
