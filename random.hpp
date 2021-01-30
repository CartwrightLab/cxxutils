/*
MIT License

Copyright (c) 2019-2020 Reed A. Cartwright <reed@cartwrig.ht>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#ifndef RACUTILS_RANDOM_HPP
#define RACUTILS_RANDOM_HPP

#include <array>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

#if defined(_WIN64) || defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include <chrono>
#include <random>
#include <string>
#include <thread>

namespace racutils::random {

template <size_t count>
class SeedSeq;
class Random;

namespace details {

/*
A Fast 128-bit, Lehmer-style PRNG
Copyright (c) 2018 Melissa E. O'Neill
The MIT License (MIT)
https://gist.github.com/imneme/aeae7628565f15fb3fef54be8533e39c
https://www.pcg-random.org/posts/does-it-beat-the-minimal-standard.html
*/

class Lehmer64Fast {
   public:
    using result_type = uint64_t;
    using state_type = __uint128_t;
    using seed_type = std::array<uint32_t, sizeof(state_type) / sizeof(uint32_t)>;

   private:
    state_type state_;
    static constexpr auto MCG_MULT = 0xda942042e4dd58b5ULL;
    static constexpr unsigned int STYPE_BITS = 8 * sizeof(state_type);
    static constexpr unsigned int RTYPE_BITS = 8 * sizeof(result_type);

   public:
    static constexpr result_type min() { return static_cast<result_type>(0); }
    static constexpr result_type max() { return static_cast<result_type>(~static_cast<result_type>(0)); }

    explicit Lehmer64Fast(state_type state = state_type(0x9f57c403d06c42fcULL)) { SetState(state); }
    explicit Lehmer64Fast(seed_type seed) { Seed(seed); }

    void Seed(state_type state) { SetState(state); }

    void Seed(seed_type seed) {
        state_type state;
        std::memcpy(&state, &seed, sizeof(seed));
        Seed(state);
    }

    void Advance() { state_ *= MCG_MULT; }

    void Discard(size_t count) {
        for(size_t k = 0; k < count; ++k) {
            Advance();
        }
    }

    state_type GetState() const { return state_; }
    seed_type GetSeed() const {
        seed_type seed;
        std::memcpy(&seed, &state_, sizeof(seed));
        return seed;
    }

    result_type operator()() {
        Advance();
        return static_cast<result_type>(state_ >> (STYPE_BITS - RTYPE_BITS));
    }

    bool operator==(const Lehmer64Fast &rhs) { return (state_ == rhs.state_); }

    bool operator!=(const Lehmer64Fast &rhs) { return !operator==(rhs); }

   private:
    void SetState(state_type state) {
        // state must be odd.
        state_ = state | 1;
    }
};

using RandomEngine = Lehmer64Fast;

inline int64_t random_i63(uint64_t u) { return u >> 1; }
inline uint32_t random_u32(uint64_t u) { return u >> 32; }
inline int32_t random_i31(uint64_t u) { return u >> 33; }

// uniformly distributed between [0,range)
// Algorithm 5 from Lemire (2018) https://arxiv.org/pdf/1805.10941.pdf
// Modified by M.E. O'Neill (2018) https://www.pcg-random.org/posts/bounded-rands.html
template <typename callback>
uint64_t random_u64_range(uint64_t range, callback &get) {
    uint64_t x = get();
    __uint128_t m = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(range);
    auto l = static_cast<uint64_t>(m);
    if(l < range) {
        // Calculate {uint64_t t = -range % range} avoiding divisions
        // as much as possible.
        uint64_t t = -range;
        if(t >= range) {
            t -= range;
            if(t >= range) {
                t %= range;
            }
        }
        while(l < t) {
            x = get();
            m = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(range);
            l = static_cast<uint64_t>(m);
        }
    }
    return m >> 64;
}

inline std::pair<uint32_t, uint32_t> random_u32_pair(uint64_t u) { return {u, u >> 32}; }

inline double random_f53(uint64_t u) {
    auto n = static_cast<int64_t>(u >> 11);
    return n / 9007199254740992.0;
}

inline double random_f52(uint64_t u) {
    auto n = 1 | static_cast<int64_t>(u >> 11);
    return n / 9007199254740992.0;
}


inline double random_exp_inv(double f) { return -log(f); }

extern const std::array<int64_t, 256> ek;
extern const std::array<double, 256> ew;
extern const std::array<double, 256> ef;

template <typename callback>
double random_exp_zig_internal(int64_t a, int b, callback &get) {
    constexpr double r = 7.69711747013104972;
    do {
        if(b == 0) {
            return r + random_exp_inv(random_f52(get()));
        }
        double x = a * ew[b];
        if(ef[b - 1] + random_f52(get()) * (ef[b] - ef[b - 1]) < exp(-x)) {
            return x;
        }
        a = random_i63(get());
        b = static_cast<int>(a & 255);
    } while(a > ek[b]);
    return a * ew[b];
}

template <typename callback>
inline double random_exp_zig(callback &get) {
    int64_t a = random_i63(get());
    auto b = static_cast<int>(a & 255);
    if(a <= ek[b]) {
        return a * ew[b];
    }
    return random_exp_zig_internal(a, b, get);
}

}  // namespace details

// code sanity check
static_assert(std::is_same<uint64_t, details::RandomEngine::result_type>::value,
              "The result type of RandomEngine is not a uint64_t.");

class Random : public details::RandomEngine {
   public:
    using engine_type = details::RandomEngine;
    // import constructor
    using engine_type::engine_type;
    // import seed_type
    using seed_type = engine_type::seed_type;

    uint64_t bits();
    uint64_t bits(int b);

    uint64_t u64();
    uint64_t u64(uint64_t range);

    uint32_t u32();
    std::pair<uint32_t, uint32_t> u32_pair();

    double f52();
    double f53();

    double exp(double mean = 1.0);

    template <size_t count>
    void Seed(const SeedSeq<count> &ss);
    void Seed(const uint32_t s);
    using engine_type::Seed;
};

// uniformly distributed between [0,2^64)
inline uint64_t Random::bits() { return details::RandomEngine::operator()(); }
// uniformly distributed between [0,2^b)
inline uint64_t Random::bits(int b) { return bits() >> (64 - b); }

// uniformly distributed between [0,2^64)
inline uint64_t Random::u64() { return bits(); }

// uniformly distributed between [0,range)
inline uint64_t Random::u64(uint64_t range) { return details::random_u64_range(range, *this); }

// uniformly distributed between [0,2^32)
inline uint32_t Random::u32() { return details::random_u32(bits()); }

// uniformly distributed pair between [0,2^32)
inline std::pair<uint32_t, uint32_t> Random::u32_pair() { return details::random_u32_pair(bits()); }

// uniformly distributed between (0,1.0)
inline double Random::f52() { return details::random_f52(bits()); }

// uniformly distributed between [0,1.0)
inline double Random::f53() { return details::random_f53(bits()); }

// exponential random value with specified mean. mean=1.0/rate
inline double Random::exp(double mean) { return details::random_exp_zig(*this) * mean; }

// Think about using https://gist.github.com/imneme/540829265469e673d045
// https://www.pcg-random.org/posts/simple-portable-cpp-seed-entropy.html
// https://www.pcg-random.org/posts/cpps-random_device.html

namespace details {
// Multilinear hash (https://arxiv.org/pdf/1202.4961.pdf)
// Hash is based on a sequence of 64-bit numbers generated by Weyl sequence
// Multilinear hash is (m_0 + sum(m_i*u_i) mod 2^64) / 2^32
// m = buffer of 64-bit unsigned random values
// u = 32-bit input values that are being hashed
template <uint64_t INC, uint64_t INIT>
struct hash_impl_t {
    template <typename In1, typename In2, typename Out1, typename Out2>
    void operator()(In1 it1, In2 it2, Out1 itA, Out2 itB) {
        uint64_t w = INIT;
        auto next_num = [&w]() {
            w += INC;
            return w;
        };

        for(auto out = itA; out != itB; ++out) {
            // hash input
            uint64_t sum = next_num();
            for(auto it = it1; it != it2; ++it) {
                uint32_t u = *it;
                sum += next_num() * u;
            }
            // If input ends in a zero, the hash is not unique.
            // Add a final value to ensure that this doesn't happen.
            sum += next_num() * 1;
            // final value
            *out = static_cast<uint32_t>(sum >> 32);
        }
    }
};

using hash_implA = hash_impl_t<0x9e3779b97f4a7c15ULL, 0x3423da0b87484307ULL>;
using hash_implB = hash_impl_t<0x9e3779b97f4a7c15ULL, 0xdf8b06c40fa44478ULL>;
}  // namespace details

// SeedSeq is a finite entropy seed sequence.
// Inspiration: https://www.pcg-random.org/posts/developing-a-seed_seq-alternative.html
template <size_t count>
class SeedSeq {
   public:
    using result_type = uint32_t;

   private:
    std::array<result_type, count> state_;

   public:
    template <typename It1, typename It2>
    SeedSeq(It1 begin, It2 end) {
        Seed(begin, end);
    }

    template <typename T>
    SeedSeq(std::initializer_list<T> init) {
        Seed(init.begin(), init.end());
    }

    // Generates an internal state based on provided seeds
    template <typename It1, typename It2>
    void Seed(It1 begin, It2 end) {
        details::hash_implA hash;
        hash(begin, end, state_.begin(), state_.end());
    }

    // Generates an external state based on the internal state
    template <typename It1, typename It2>
    void Generate(It1 begin, It2 end) const {
        details::hash_implB hash;
        hash(state_.begin(), state_.end(), begin, end);
    }
};

using SeedSeq256 = SeedSeq<8>;

inline void Random::Seed(uint32_t s) {
    SeedSeq256 ss({s});
    Seed(ss);
}

template <size_t count>
inline void Random::Seed(const SeedSeq<count> &ss) {
    seed_type seed;
    ss.Generate(seed.begin(), seed.end());
    Seed(seed);
}

namespace details {
inline std::string base58_encode(uint32_t u) {
    const char *base58_alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    std::string buffer(6, base58_alphabet[0]);
    for(int i = 0; i < 6 && u != 0; ++i) {
        buffer[5 - i] = base58_alphabet[u % 58];
        u = u / 58;
    }
    return buffer;
}
}  // namespace details

template <size_t COUNT>
std::string encode_seed(const std::array<uint32_t, COUNT> &seed) {
    if(seed.empty()) {
        return {};
    }
    std::string str = details::base58_encode(seed[0]);
    for(int i=1;i<seed.size(); ++i) {
        str += "-";
        str += details::base58_encode(seed[i]);
    }
    return str;
}

namespace details {
static constexpr uint32_t fnv(uint32_t hash, const char *pos) {
    return *pos == '\0' ? hash : fnv((hash * 16777619U) ^ *pos, pos + 1);
}

template <typename T>
static uint64_t hash(T &&value) {
    auto h =
        std::hash<typename std::remove_reference<typename std::remove_cv<T>::type>::type>{}(std::forward<T>(value));
    return static_cast<uint64_t>(h);
}

template <typename T>
static uint32_t crushto32(T &&value) {
    uint64_t u = details::hash(value);
    // Multilinear hash
    uint64_t result = 0x80e25f91f5ba47eaULL;
    result += 0x6db4dd6c7a89963cULL * static_cast<uint32_t>(u);
    result += 0xd35f3cdd31f49ad8ULL * static_cast<uint32_t>(u >> 32);
    return static_cast<uint32_t>(result >> 32);
};
}  // namespace details

// Based on ideas from https://www.pcg-random.org/posts/simple-portable-cpp-seed-entropy.html
// Based on code from https://gist.github.com/imneme/540829265469e673d045
inline SeedSeq256 auto_seed_seq() {
    using details::crushto32;

    // Constant that changes every time we compile the code
    constexpr uint32_t compile_stamp = details::fnv(2166136261U, __DATE__ __TIME__ __FILE__);

    // get 32-bits of system-wide entropy once
    static uint32_t random_int = std::random_device{}();
    // increment it every call and don't worry about race conditions
    random_int += 0xedf19156;

    // heap randomness
    void *malloc_addr = malloc(sizeof(int));  // NOLINT
    free(malloc_addr);                        // NOLINT
    auto heap = crushto32(malloc_addr);
    auto stack = crushto32(&malloc_addr);

    // High-resolution time information
    auto hitime = crushto32(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    // The address of the couple of functions.
    auto time_func = crushto32(&std::chrono::high_resolution_clock::now);
    auto exit_func = crushto32(&_Exit);
    auto self_func = crushto32(&auto_seed_seq);

    // Thread ID
    auto thread_id = crushto32(std::this_thread::get_id());

    // PID
#if defined(_WIN64) || defined(_WIN32)
    auto pid = crushto32(_getpid());
#else
    auto pid = crushto32(getpid());
#endif

#if defined(__has_builtin) && __has_builtin(__builtin_readcyclecounter)
    auto cpu = crushto32(__builtin_readcyclecounter());
#else
    uint32_t cpu = 0;
#endif

    return SeedSeq256(
        {compile_stamp, random_int, heap, stack, hitime, time_func, exit_func, self_func, thread_id, pid, cpu});
}

class AliasTable {
   public:
    AliasTable() = default;

    template <typename... Args>
    explicit AliasTable(Args &&...args) {
        create(std::forward<Args>(args)...);
    }

    // create the alias table
    void CreateInplace(std::vector<double> *v);

    // create the alias table
    template <typename... Args>
    void Create(Args &&...args) {
        std::vector<double> vv(std::forward<Args>(args)...);
        CreateInplace(&vv);
    }

    int Get(uint64_t u) const {
        auto yx = details::random_u32_pair(u >> shr_);
        return (yx.first < p_[yx.second]) ? yx.second : a_[yx.second];
    }

    const std::vector<uint32_t> &a() const { return a_; }
    const std::vector<uint32_t> &p() const { return p_; }

    int operator()(uint64_t u) const { return Get(u); }

   private:
    template <typename T>
    inline static std::pair<T, int> RoundUp(T x) {
        T y = static_cast<T>(2);
        int k = 1;
        for(; y < x; y *= 2, ++k) {
            /*noop*/;
        }
        return std::make_pair(y, k);
    }

    int shr_{0};
    std::vector<uint32_t> a_;
    std::vector<uint32_t> p_;
};

inline void AliasTable::CreateInplace(std::vector<double> *v) {
    assert(v != nullptr);
    assert(v->size() <= std::numeric_limits<uint32_t>::max());
    // round the size of vector up to the nearest power of two
    auto ru = RoundUp(v->size());
    size_t sz = ru.first;
    v->resize(sz, 0.0);
    a_.resize(sz, 0);
    p_.resize(sz, 0);
    // use the number of bits to calculate the right shift operand
    shr_ = 64 - ru.second;

    // find scale for input vector
    double d = std::accumulate(v->begin(), v->end(), 0.0) / sz;

    // find first large and small values
    //     g: current large value index
    //     m: current small value index
    //    mm: next possible small value index
    size_t g = 0, m = 0;
    for(; g < sz && (*v)[g] < d; ++g) {
        /*noop*/;
    }
    for(; m < sz && (*v)[m] >= d; ++m) {
        /*noop*/;
    }
    size_t mm = m + 1;

    // construct table
    while(g < sz && m < sz) {
        assert((*v)[m] < d);
        p_[m] = static_cast<uint32_t>(4294967296.0 / d * (*v)[m]);
        a_[m] = static_cast<uint32_t>(g);
        (*v)[g] = ((*v)[g] + (*v)[m]) - d;
        if((*v)[g] >= d || mm <= g) {
            for(m = mm; m < sz && (*v)[m] >= d; ++m) {
                /*noop*/;
            }
            mm = m + 1;
        } else {
            m = g;
        }
        for(; g < sz && (*v)[g] < d; ++g) {
            /*noop*/;
        }
    }
    // if we stopped early fill in the rest
    if(g < sz) {
        p_[g] = std::numeric_limits<uint32_t>::max();
        a_[g] = static_cast<uint32_t>(g);
        for(g = g + 1; g < sz; ++g) {
            if((*v)[g] < d) continue;
            p_[g] = std::numeric_limits<uint32_t>::max();
            a_[g] = static_cast<uint32_t>(g);
        }
    }
    // if we stopped early fill in the rest
    if(m < sz) {
        p_[m] = std::numeric_limits<uint32_t>::max();
        a_[m] = static_cast<uint32_t>(m);
        for(m = mm; m < sz; ++m) {
            if((*v)[m] > d) continue;
            p_[m] = std::numeric_limits<uint32_t>::max();
            a_[m] = static_cast<uint32_t>(m);
        }
    }
}

}  // namespace racutils::random

// RACUTILS_RANDOM
#endif
