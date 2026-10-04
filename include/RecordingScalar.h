/* 路径 A：记录型标量（recording scalar）+ 反向磁带（tape）。
 *
 * 目的：不把参数塞进稠密 DA，也不手写 tape RHS；用一个标量类型包裹
 * (value, node, tape)，重载算术与初等函数。把既有 `template<typename T>`
 * RHS（如 TBPfull）直接以 T=Scalar 实例化即自动记录整条积分计算图；
 * 反向为对偶标量的标准反向模式（乘法伴随即乘法）。参数梯度代价 ~ 图上算子数，
 * 与 m 无关；不重写 DA、不新增 DACE 内核原语。
 *
 * 依赖：DACE 的 PromotionTrait（使 AlgebraicVector<Scalar> 的元素级运算正确提升）。
 */
#ifndef DACE_RECORDINGSCALAR_H_
#define DACE_RECORDINGSCALAR_H_

#include <cmath>
#include <cstddef>
#include <vector>

#include <dace/dace.h>

namespace DACE {

class RecordTape {
public:
    enum Kind { LEAF, ADD, SUB, MUL, DIV, NEG, SCALE, SIN, COS, EXP, LOG, SQRT, POW };
    struct Node { Kind kind; int a; int b; double s; };

    int leaf(double v) { vals_.push_back(v); nodes_.push_back({LEAF, -1, -1, 0.0}); return size() - 1; }
    int add(int a, int b) { vals_.push_back(vals_[a] + vals_[b]); nodes_.push_back({ADD, a, b, 0.0}); return size() - 1; }
    int sub(int a, int b) { vals_.push_back(vals_[a] - vals_[b]); nodes_.push_back({SUB, a, b, 0.0}); return size() - 1; }
    int mul(int a, int b) { vals_.push_back(vals_[a] * vals_[b]); nodes_.push_back({MUL, a, b, 0.0}); return size() - 1; }
    int div(int a, int b) { vals_.push_back(vals_[a] / vals_[b]); nodes_.push_back({DIV, a, b, 0.0}); return size() - 1; }
    int neg(int a) { vals_.push_back(-vals_[a]); nodes_.push_back({NEG, a, -1, 0.0}); return size() - 1; }
    int scale(int a, double s) { vals_.push_back(vals_[a] * s); nodes_.push_back({SCALE, a, -1, s}); return size() - 1; }
    int sin(int a) { vals_.push_back(std::sin(vals_[a])); nodes_.push_back({SIN, a, -1, 0.0}); return size() - 1; }
    int cos(int a) { vals_.push_back(std::cos(vals_[a])); nodes_.push_back({COS, a, -1, 0.0}); return size() - 1; }
    int exp(int a) { vals_.push_back(std::exp(vals_[a])); nodes_.push_back({EXP, a, -1, 0.0}); return size() - 1; }
    int log(int a) { vals_.push_back(std::log(vals_[a])); nodes_.push_back({LOG, a, -1, 0.0}); return size() - 1; }
    int sqrt(int a) { vals_.push_back(std::sqrt(vals_[a])); nodes_.push_back({SQRT, a, -1, 0.0}); return size() - 1; }
    int pow(int a, double p) { vals_.push_back(std::pow(vals_[a], p)); nodes_.push_back({POW, a, -1, p}); return size() - 1; }

    int size() const { return static_cast<int>(vals_.size()); }
    double value(int i) const { return vals_[i]; }

    // 反向：对若干输出节点给种子，返回每个节点的伴随。
    std::vector<double> backward(const std::vector<int> &outs,
                                 const std::vector<double> &seeds) const {
        std::vector<double> g(vals_.size(), 0.0);
        for (std::size_t k = 0; k < outs.size(); ++k) g[outs[k]] += seeds[k];
        for (int i = size() - 1; i >= 0; --i) {
            const Node &n = nodes_[i];
            const double gi = g[i];
            switch (n.kind) {
                case LEAF:  break;
                case ADD:   g[n.a] += gi; g[n.b] += gi; break;
                case SUB:   g[n.a] += gi; g[n.b] -= gi; break;
                case MUL:   g[n.a] += gi * vals_[n.b]; g[n.b] += gi * vals_[n.a]; break;
                case DIV:   g[n.a] += gi / vals_[n.b]; g[n.b] += -gi * vals_[i] / vals_[n.b]; break;
                case NEG:   g[n.a] -= gi; break;
                case SCALE: g[n.a] += gi * n.s; break;
                case SIN:   g[n.a] += gi * std::cos(vals_[n.a]); break;
                case COS:   g[n.a] += -gi * std::sin(vals_[n.a]); break;
                case EXP:   g[n.a] += gi * vals_[i]; break;
                case LOG:   g[n.a] += gi / vals_[n.a]; break;
                case SQRT:  g[n.a] += gi * 0.5 / (vals_[i] != 0.0 ? vals_[i] : 1.0); break;
                case POW:   g[n.a] += gi * n.s * std::pow(vals_[n.a], n.s - 1.0); break;
            }
        }
        return g;
    }

private:
    std::vector<Node> nodes_;
    std::vector<double> vals_;
};

inline RecordTape *&active_tape() { static thread_local RecordTape *t = nullptr; return t; }

class Scalar {
public:
    double v;
    int node;
    RecordTape *tape;
    Scalar() : v(0.0), node(-1), tape(nullptr) {}
    Scalar(double x) : v(x), node(-1), tape(nullptr) {}
};

inline Scalar make_leaf(double x) {
    Scalar s;
    s.v = x;
    s.tape = active_tape();
    s.node = s.tape ? s.tape->leaf(x) : -1;
    return s;
}

inline RecordTape *pick_tape(const Scalar &a, const Scalar &b) { return a.tape ? a.tape : b.tape; }
inline int node_on(RecordTape *t, const Scalar &s) {
    return (s.tape == t && s.node >= 0) ? s.node : t->leaf(s.v);
}

inline Scalar operator+(const Scalar &a, const Scalar &b) {
    RecordTape *t = pick_tape(a, b); if (!t) return Scalar(a.v + b.v);
    Scalar r; r.v = a.v + b.v; r.tape = t; r.node = t->add(node_on(t, a), node_on(t, b)); return r;
}
inline Scalar operator-(const Scalar &a, const Scalar &b) {
    RecordTape *t = pick_tape(a, b); if (!t) return Scalar(a.v - b.v);
    Scalar r; r.v = a.v - b.v; r.tape = t; r.node = t->sub(node_on(t, a), node_on(t, b)); return r;
}
inline Scalar operator*(const Scalar &a, const Scalar &b) {
    RecordTape *t = pick_tape(a, b); if (!t) return Scalar(a.v * b.v);
    Scalar r; r.v = a.v * b.v; r.tape = t; r.node = t->mul(node_on(t, a), node_on(t, b)); return r;
}
inline Scalar operator/(const Scalar &a, const Scalar &b) {
    RecordTape *t = pick_tape(a, b); if (!t) return Scalar(a.v / b.v);
    Scalar r; r.v = a.v / b.v; r.tape = t; r.node = t->div(node_on(t, a), node_on(t, b)); return r;
}
inline Scalar operator-(const Scalar &a) {
    if (!a.tape) return Scalar(-a.v);
    Scalar r; r.v = -a.v; r.tape = a.tape; r.node = a.tape->neg(node_on(a.tape, a)); return r;
}
inline Scalar operator+(const Scalar &a) { return a; }

inline Scalar &operator+=(Scalar &a, const Scalar &b) { a = a + b; return a; }
inline Scalar &operator-=(Scalar &a, const Scalar &b) { a = a - b; return a; }
inline Scalar &operator*=(Scalar &a, const Scalar &b) { a = a * b; return a; }
inline Scalar &operator/=(Scalar &a, const Scalar &b) { a = a / b; return a; }

inline Scalar sin(const Scalar &a) {
    if (!a.tape) return Scalar(std::sin(a.v));
    Scalar r; r.v = std::sin(a.v); r.tape = a.tape; r.node = a.tape->sin(node_on(a.tape, a)); return r;
}
inline Scalar cos(const Scalar &a) {
    if (!a.tape) return Scalar(std::cos(a.v));
    Scalar r; r.v = std::cos(a.v); r.tape = a.tape; r.node = a.tape->cos(node_on(a.tape, a)); return r;
}
inline Scalar exp(const Scalar &a) {
    if (!a.tape) return Scalar(std::exp(a.v));
    Scalar r; r.v = std::exp(a.v); r.tape = a.tape; r.node = a.tape->exp(node_on(a.tape, a)); return r;
}
inline Scalar log(const Scalar &a) {
    if (!a.tape) return Scalar(std::log(a.v));
    Scalar r; r.v = std::log(a.v); r.tape = a.tape; r.node = a.tape->log(node_on(a.tape, a)); return r;
}
inline Scalar sqrt(const Scalar &a) {
    if (!a.tape) return Scalar(std::sqrt(a.v));
    Scalar r; r.v = std::sqrt(a.v); r.tape = a.tape; r.node = a.tape->sqrt(node_on(a.tape, a)); return r;
}
inline Scalar pow(const Scalar &a, double p) {
    if (!a.tape) return Scalar(std::pow(a.v, p));
    Scalar r; r.v = std::pow(a.v, p); r.tape = a.tape; r.node = a.tape->pow(node_on(a.tape, a), p); return r;
}
inline Scalar pow(const Scalar &a, int p) { return pow(a, static_cast<double>(p)); }
inline Scalar sqr(const Scalar &a) { return a * a; }

inline double cons(const Scalar &a) { return a.v; }

// PromotionTrait：Scalar 与 double 的混合提升仍为 Scalar（供 AlgebraicVector<Scalar> 用）。
template <> class PromotionTrait<Scalar, Scalar> { public: typedef Scalar returnType; };
template <> class PromotionTrait<Scalar, double> { public: typedef Scalar returnType; };
template <> class PromotionTrait<double, Scalar> { public: typedef Scalar returnType; };
template <> class PromotionTrait<Scalar, int> { public: typedef Scalar returnType; };
template <> class PromotionTrait<int, Scalar> { public: typedef Scalar returnType; };
template <> class PromotionTrait<Scalar, unsigned int> { public: typedef Scalar returnType; };
template <> class PromotionTrait<unsigned int, Scalar> { public: typedef Scalar returnType; };

}  // namespace DACE

#endif /* DACE_RECORDINGSCALAR_H_ */
