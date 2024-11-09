/*
MIT License

Copyright (c) 2019-2020,2024 Reed A. Cartwright <racartwright@gmail.com>

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
#ifndef FRAGMITES_RANDOM_HPP
#define FRAGMITES_RANDOM_HPP

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

namespace fragmites::random {

template <size_t count>
class SeedSeq;
class Random;

#if defined(__has_builtin) && __has_builtin(__builtin_expect)
#define FRAGMITES_LIKELY(x) __builtin_expect(!!(x), 1)
#define FRAGMITES_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define FRAGMITES_LIKELY(x) !!(x)
#define FRAGMITES_UNLIKELY(x) !!(x)
#endif

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

// Use 128-bits to draw a random number between [0, range)
// Algorithm from https://github.com/apple/swift/pull/39143
// And https://github.com/KWillets/range_generator/blob/master/include/range_generator.hpp
// TODO: validate this
template <typename callback>
uint64_t random_u64_range(uint64_t range, callback &get) {
    uint64_t r1 = get();
    __uint128_t x = r1;
    x = x*range;
    uint64_t y = static_cast<uint64_t>(x >> 64);
    uint64_t f = static_cast<uint64_t>(x);
    // Check if fraction + range can carry.
    // (a+b < b) compiles to carry flag.
    // Optimize for small ranges.
    if(FRAGMITES_LIKELY(range + f >= f)) {
        return y;
    }
    uint64_t r2 = get();
    x = r2;
    x = x*range;
    uint64_t z = static_cast<uint64_t>(x >> 64);

    return y + ((z+f < f) ? 1 : 0);
}

// uniformly distributed between [0,range)
// Algorithm from https://github.com/apple/swift/pull/39143
// And https://github.com/KWillets/range_generator/blob/master/include/range_generator.hpp
// TODO: validate this
template <typename callback>
uint64_t random_u64_range_exact(uint64_t range, callback &get) {
    __uint128_t x = get();
    x = x*range;
    uint64_t y = static_cast<uint64_t>(x >> 64);
    uint64_t f = static_cast<uint64_t>(x);
    // Optimize for small ranges.
    if(FRAGMITES_LIKELY(range + f >= f)) {
        return y;
    }
    do {
        x = get();
        x = x*range;
        uint64_t z = static_cast<uint64_t>(x >> 64);
        z = z+f;
        if(z < f) {
            // we have carried
            return y + 1;
        } else if(z != -1) {
            // we will never carry
            break;
        }
        f = static_cast<uint64_t>(x);
    } while( range + f < f );

    return y;
}

inline std::pair<uint32_t, uint32_t> random_u32_pair(uint64_t u) { return {u, u >> 32}; }

inline double random_f53(uint64_t u) {
    auto n = static_cast<int64_t>(u >> 11);
    return n / 9007199254740992.0;
}

inline double random_f52(uint64_t u) {
    auto n = static_cast<int64_t>(u >> 11) | 1;
    return n / 9007199254740992.0;
}


inline double random_exp_inv(double f) { return -log(f); }

const int64_t * get_ek();
const double * get_ew();
const double * get_ef();

template <typename callback>
double random_exp_zig_internal(int64_t a, int b, callback &get) {
    constexpr const double r = 7.69711747013104972;
    const int64_t *ek = get_ek();
    const double *ew = get_ew();
    const double *ef = get_ef();
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
    const int64_t *ek = get_ek();
    const double *ew = get_ew();
    const double *ef = get_ef();
    int64_t a = random_i63(get());
    auto b = static_cast<int>(a & 255);
    if(FRAGMITES_LIKELY(a <= ek[b])) {
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

/************************************************************
 * Tables for exponential                                   *
 ************************************************************/

inline const int64_t * details::get_ek() {
    #define U(x) INT64_C(x)  // NOLINT
    static const int64_t ek[256] = {
        U(8162862958009850966), U(8317365004821266796), U(8608587457533758008), U(8747546414904111780),
        U(8830037661796713635), U(8885094666497980832), U(8924652222586790058), U(8954554714978013301),
        U(8978014716828853716), U(8996950217926807086), U(9012579824174066832), U(9025716384411536725),
        U(9036923732325504597), U(9046605762814123347), U(9055059702522883807), U(9062509347116970745),
        U(9069126555852812300), U(9075045585691907911), U(9080372908245545455), U(9085194091635816472),
        U(9089578725147156637), U(9093584008437555377), U(9097257410708403207), U(9100638670155140278),
        U(9103761317625194006), U(9106653851930257827), U(9109340656600290276), U(9111842722299704653),
        U(9114178221480881020), U(9116362969487005798), U(9118410797532468663), U(9120333856667860920),
        U(9122142867232936535), U(9123847324910810001), U(9125455671974332906), U(9126975440420644589),
        U(9128413372253481969), U(9129775521074630109), U(9131067338299545732), U(9132293746655141203),
        U(9133459203103853450), U(9134567752933576710), U(9135623076432663468), U(9136628529313676998),
        U(9137587177844972235), U(9138501829484038972), U(9139375059672990805), U(9140209235347706313),
        U(9141006535623069793), U(9141768970043748419), U(9142498394729435394), U(9143196526693605205),
        U(9143864956573184231), U(9144505159971792821), U(9145118507590137337), U(9145706274292643388),
        U(9146269647238760811), U(9146809733189905803), U(9147327565088134495), U(9147824107990051519),
        U(9148300264428576645), U(9148756879266028577), U(9149194744094020420), U(9149614601228742352),
        U(9150017147344460263), U(9150403036782800193), U(9150772884571045721), U(9151127269178795786),
        U(9151466735038992163), U(9151791794856380800), U(9152102931723882998), U(9152400601065214488),
        U(9152685232419954302), U(9152957231085667112), U(9153216979630091842), U(9153464839285070514),
        U(9153701151232655273), U(9153926237792866426), U(9154140403521535772), U(9154343936225869513),
        U(9154537107904630429), U(9154720175619186593), U(9154893382301016954), U(9155056957500802908),
        U(9155211118083722902), U(9155356068875154421), U(9155492003260584934), U(9155619103743209229),
        U(9155737542462376376), U(9155847481675736091), U(9155949074207750260), U(9156042463866939008),
        U(9156127785834021219), U(9156205167023021881), U(9156274726417075467), U(9156336575380691972),
        U(9156390817949935427), U(9156437551101969834), U(9156476865005213917), U(9156508843251305789),
        U(9156533563069945569), U(9156551095527572479), U(9156561505710820917), U(9156564852895547428),
        U(9156561190702166721), U(9156550567238037995), U(9156533025227461028), U(9156508602129892141),
        U(9156477330246889658), U(9156439236818230136), U(9156394344107664754), U(9156342669478644521),
        U(9156284225460376283), U(9156219019804516963), U(9156147055532747044), U(9156068330975468965),
        U(9155982839801816545), U(9155890571041145847), U(9155791509096144952), U(9155685633747648753),
        U(9155572920151253552), U(9155453338825780172), U(9155326855633574861), U(9155193431752695854),
        U(9155053023640886811), U(9154905582991326276), U(9154751056680040063), U(9154589386704867997),
        U(9154420510115830072), U(9154244358936739397), U(9154060860077836362), U(9153869935239234465),
        U(9153671500804879991), U(9153465467726793827), U(9153251741399175193), U(9153030221522087440),
        U(9152800801954269978), U(9152563370554659974), U(9152317809012124202), U(9152063992662892748),
        U(9151801790295100144), U(9151531063939835850), U(9151251668647985950), U(9150963452252197138),
        U(9150666255113097649), U(9150359909848963198), U(9150044241047887849), U(9149719064961449533),
        U(9149384189178772708), U(9149039412279838423), U(9148684523466762991), U(9148319302171653467),
        U(9147943517639573402), U(9147556928485021044), U(9147159282220160661), U(9146750314752930620),
        U(9146329749853016129), U(9145897298583425990), U(9145452658695364058), U(9144995513983704533),
        U(9144525533600323394), U(9144042371322220215), U(9143545664771051338), U(9143035034580539816),
        U(9142510083507759119), U(9141970395484038474), U(9141415534600808323), U(9140845044025273151),
        U(9140258444840364215), U(9139655234802854045), U(9139034887013004987), U(9138396848488404315),
        U(9137740538634056317), U(9137065347599854436), U(9136370634515884910), U(9135655725594897045),
        U(9134919912090232620), U(9134162448096416464), U(9133382548178095122), U(9132579384811715922),
        U(9131752085622469033), U(9130899730397333917), U(9130021347852788454), U(9129115912133549320),
        U(9128182339015852851), U(9127219481786016043), U(9126226126761344582), U(9125200988416891291),
        U(9124142704076985000), U(9123049828125643018), U(9121920825684261719), U(9120754065698622745),
        U(9119547813369876543), U(9118300221855756696), U(9117009323158726552), U(9115673018106577841),
        U(9114289065318365977), U(9112855069033762488), U(9111368465666926777), U(9109826508926222556),
        U(9108226253318129128), U(9106564535826924938), U(9104837955530412016), U(9103042850875206584),
        U(9101175274292077217), U(9099230963780780745), U(9097205311033859470), U(9095093325597322612),
        U(9092889594481259745), U(9090588236531886709), U(9088182850754808976), U(9085666457632919044),
        U(9083031432305413232), U(9080269428259872903), U(9077371289928083436), U(9074326952256459754),
        U(9071125324929106539), U(9067754158436081890), U(9064199888577030896), U(9060447455238410921),
        U(9056480090338858174), U(9052279068646041429), U(9047823413654964258), U(9043089548783006379),
        U(9038050881645731106), U(9032677305946930836), U(9026934601293794893), U(9020783705683674320),
        U(9014179828011887897), U(9007071358024495219), U(8999398517688769502), U(8991091679529586360),
        U(8982069251953252166), U(8972234995776471882), U(8961474585289418087), U(8949651153807399629),
        U(8936599456209602962), U(8922118120902181607), U(8905959220757275078), U(8887814016419182333),
        U(8867293129541507760), U(8843898435871603566), U(8816982352148886212), U(8785687406231795429),
        U(8748854008138864149), U(8704875095105001658), U(8651458309292236479), U(8585219286399828019),
        U(8500948185393581898), U(8390198088981398944), U(8238337428567649362), U(8017709005563621762),
        U(7669309967995944412), U(7042457982574018880), U(5617518561624195937), U(0),
    };
#undef U
    return ek;
}
inline const double * get_ew() {
    static const double ew[256] = {
        9.4294336554777197e-19, 8.3452314829922143e-19, 7.5254837402657195e-19, 7.0238720371966431e-19,
        6.6615166787391891e-19, 6.3774336460586336e-19, 6.1435342137476143e-19, 5.9445619298641686e-19,
        5.7713062906758818e-19, 5.6177797670930974e-19, 5.4798705612062106e-19, 5.3546328459557655e-19,
        5.2398837667109586e-19, 5.1339607441623787e-19, 5.0355681922637538e-19, 4.9436768282657819e-19,
        4.8574553087811921e-19, 4.7762224876877050e-19, 4.6994132547160527e-19, 4.6265535676391533e-19,
        4.5572418600698006e-19, 4.4911349657284497e-19, 4.4279373032869477e-19, 4.3673924553319651e-19,
        4.3092765322617875e-19, 4.2533928854433125e-19, 4.1995678531908898e-19, 4.1476473064691265e-19,
        4.0974938203841791e-19, 4.0489843401278370e-19, 4.0020082411162725e-19, 3.9564657060122790e-19,
        3.9122663584536545e-19, 3.8693281062370854e-19, 3.8275761565541302e-19, 3.7869421734452107e-19,
        3.7473635535050925e-19, 3.7087828004578861e-19, 3.6711469828283760e-19, 3.6344072617970784e-19,
        3.5985184786090057e-19, 3.5634387927388120e-19, 3.5291293634951768e-19, 3.4955540689494898e-19,
        3.4626792570554618e-19, 3.4304735246318168e-19, 3.3989075205443782e-19, 3.3679537699740477e-19,
        3.3375865171148808e-19, 3.3077815840288146e-19, 3.2785162437043409e-19, 3.2497691056363988e-19,
        3.2215200124729337e-19, 3.1937499464670326e-19, 3.1664409446381373e-19, 3.1395760216863265e-19,
        3.1131390998239396e-19, 3.0871149447920964e-19, 3.0614891074186178e-19, 3.0362478701506517e-19,
        3.0113781980618271e-19, 2.9868676938914667e-19, 2.9627045567236366e-19, 2.9388775439576359e-19,
        2.9153759362598108e-19, 2.8921895052201823e-19, 2.8693084834668253e-19, 2.8467235370168888e-19,
        2.8244257396660123e-19, 2.8024065492381037e-19, 2.7806577855353272e-19, 2.7591716098439982e-19,
        2.7379405058661953e-19, 2.7169572619594107e-19, 2.6962149545777516e-19, 2.6757069328181790e-19,
        2.6554268039841982e-19, 2.6353684200873925e-19, 2.6155258652143810e-19, 2.5958934436932086e-19,
        2.5764656689989681e-19, 2.5572372533436770e-19, 2.5382030979001440e-19, 2.5193582836138002e-19,
        2.5006980625603235e-19, 2.4822178498103640e-19, 2.4639132157658374e-19, 2.4457798789351148e-19,
        2.4278136991170432e-19, 2.4100106709661028e-19, 2.3923669179131555e-19, 2.3748786864182192e-19,
        2.3575423405334900e-19, 2.3403543567564688e-19, 2.3233113191545676e-19, 2.3064099147439204e-19,
        2.2896469291064046e-19, 2.2730192422300148e-19, 2.2565238245588025e-19, 2.2401577332395624e-19,
        2.2239181085533413e-19, 2.2078021705206780e-19, 2.1918072156702237e-19, 2.1759306139611143e-19,
        2.1601698058500942e-19, 2.1445222994949914e-19, 2.1289856680867036e-19, 2.1135575473023491e-19,
        2.0982356328727186e-19, 2.0830176782575975e-19, 2.0679014924229264e-19, 2.0528849377141536e-19,
        2.0379659278204692e-19, 2.0231424258249392e-19, 2.0084124423358580e-19, 1.9937740336949116e-19,
        1.9792253002580069e-19, 1.9647643847448588e-19, 1.9503894706536551e-19, 1.9360987807373240e-19,
        1.9218905755381232e-19, 1.9077631519774508e-19, 1.8937148419979494e-19, 1.8797440112551216e-19,
        1.8658490578558369e-19, 1.8520284111412277e-19, 1.8382805305116138e-19, 1.8246039042912043e-19,
        1.8109970486304394e-19, 1.7974585064439339e-19, 1.7839868463820828e-19, 1.7705806618344747e-19,
        1.7572385699633442e-19, 1.7439592107653644e-19, 1.7307412461601631e-19, 1.7175833591039960e-19,
        1.7044842527270844e-19, 1.6914426494931727e-19, 1.6784572903799142e-19, 1.6655269340787405e-19,
        1.6526503562129129e-19, 1.6398263485724894e-19, 1.6270537183649830e-19, 1.6143312874805053e-19,
        1.6016578917702329e-19, 1.5890323803370420e-19, 1.5764536148371869e-19, 1.5639204687919114e-19,
        1.5514318269078952e-19, 1.5389865844054462e-19, 1.5265836463533577e-19, 1.5142219270093532e-19,
        1.5019003491650365e-19, 1.4896178434942640e-19, 1.4773733479038476e-19, 1.4651658068854858e-19,
        1.4529941708677991e-19, 1.4408573955673361e-19, 1.4287544413373791e-19, 1.4166842725133674e-19,
        1.4046458567537018e-19, 1.3926381643746735e-19, 1.3806601676782084e-19, 1.3687108402710662e-19,
        1.3567891563740861e-19, 1.3448940901200014e-19, 1.3330246148382796e-19, 1.3211797023253671e-19,
        1.3093583220986327e-19, 1.2975594406322086e-19, 1.2857820205728253e-19, 1.2740250199336226e-19,
        1.2622873912637889e-19, 1.2505680807917479e-19, 1.2388660275394514e-19, 1.2271801624051748e-19,
        1.2155094072120244e-19, 1.2038526737191557e-19, 1.1922088625924867e-19, 1.1805768623314314e-19,
        1.1689555481479154e-19, 1.1573437807936227e-19, 1.1457404053310996e-19, 1.1341442498439632e-19,
        1.1225541240810640e-19, 1.1109688180289890e-19, 1.0993871004068042e-19, 1.0878077170763647e-19,
        1.0762293893609162e-19, 1.0646508122640143e-19, 1.0530706525800281e-19, 1.0414875468866358e-19,
        1.0299000994087663e-19, 1.0183068797423720e-19, 1.0067064204252170e-19, 9.9509721434052180e-20,
        9.8347771193778826e-20, 9.7184631825342523e-20, 9.6020138971186658e-20, 9.4854123068569567e-20,
        9.3686408979082029e-20, 9.2516815588993672e-20, 9.1345155377432912e-20, 9.0171233949040924e-20,
        8.8994849527323356e-20, 8.7815792404446870e-20, 8.6633844342678269e-20, 8.5448777922033056e-20,
        8.4260355827969878e-20, 8.3068330072122320e-20, 8.1872441138077557e-20, 8.0672417043066803e-20,
        7.9467972305095358e-20, 7.8258806803471271e-20, 7.7044604518845635e-20, 7.5825032136697759e-20,
        7.4599737495613545e-20, 7.3368347858630169e-20, 7.2130467982244075e-20, 7.0885677953268074e-20,
        6.9633530758404745e-20, 6.8373549544960262e-20, 6.7105224523279430e-20, 6.5828009451882065e-20,
        6.4541317634466860e-20, 6.3244517343321712e-20, 6.1936926565459744e-20, 6.0617806944953073e-20,
        5.9286356766085762e-20, 5.7941702785243609e-20, 5.6582890672390323e-20, 5.5208873762106148e-20,
        5.3818499734717752e-20, 5.2410494743341757e-20, 5.0983444363209715e-20, 4.9535770551788563e-20,
        4.8065703552030905e-20, 4.6571247317037466e-20, 4.5050136537921750e-20, 4.3499782649241914e-20,
        4.1917205160571539e-20, 4.0298943146249477e-20, 3.8640939434546418e-20, 3.6938386492867660e-20,
        3.5185517370453638e-20, 3.3375315822119411e-20, 3.1499104051136477e-20, 2.9545938772669743e-20,
        2.7501694865212413e-20, 2.5347614963973565e-20, 2.3057891385186881e-20, 2.0595362397114112e-20,
        1.7903172712145925e-20, 1.4886635862824366e-20, 1.1366613766300017e-20, 6.9228654726127063e-21,
    };
    return ew;
}
inline const double * get_ef() {
    static const double ef[256] = {
        0.00045413435384149660, 0.00096726928232717421, 0.0015362997803015719, 0.0021459677437189054, 0.0027887987935740748,
        0.0034602647778369027,  0.0041572951208337936,  0.0048776559835423906, 0.0056196422072054813, 0.0063819059373191774,
        0.0071633531836349821,  0.0079630774380170365,  0.0087803149858089718, 0.0096144136425022064, 0.010464810181029975,
        0.011331013597834593,   0.012212592426255376,   0.013109164931254986,  0.014020391403181932,  0.014945968011691143,
        0.015885621839973156,   0.016839106826039941,   0.017806200410911355,  0.018786700744696024,  0.019780424338009736,
        0.020787204072578110,   0.021806887504283574,   0.022839335406385230,  0.023884420511558160,  0.024942026419731773,
        0.026012046645134207,   0.027094383780955786,   0.028188948763978622,  0.029295660224637379,  0.030414443910466590,
        0.031545232172893588,   0.032687963508959514,   0.033842582150874309,  0.035009037697397390,  0.036187284781931395,
        0.037377282772959333,   0.038578995503074830,   0.039792391023374091,  0.041017441380414785,  0.042254122413316192,
        0.043502413568888142,   0.044762297732943240,   0.046033761076175128,  0.047316792913181506,  0.048611385573379448,
        0.049917534282706330,   0.051235237055126233,   0.052564494593071644,  0.053905310196046038,  0.055257689676696989,
        0.056621641283742821,   0.057997175631200604,   0.059384305633420210,  0.060783046445479584,  0.062193415408540946,
        0.063615431999807279,   0.065049117786753707,   0.066494496385339733,  0.067951593421936560,  0.069420436498728699,
        0.070901055162371773,   0.072393480875708682,   0.073897746992364691,  0.075413888734058354,  0.076941943170480462,
        0.078481949201606380,   0.080033947542319864,   0.081597980709237378,  0.083174093009632341,  0.084762330532368091,
        0.086362741140756871,   0.087975374467270176,   0.089600281910032817,  0.091237516631040114,  0.092887133556043500,
        0.094549189376055803,   0.096223742550432742,   0.097910853311492144,  0.099610583670637076,  0.10132299742595358,
        0.10304816017125766,    0.10478613930657012,    0.10653700405000160,   0.10830082545103374,   0.11007767640518533,
        0.11186763167005624,    0.11367076788274426,    0.11548716357863348,   0.11731689921155551,   0.11916005717532763,
        0.12101672182667478,    0.12288697950954508,    0.12477091858083091,   0.12666862943751062,   0.12858020454522812,
        0.13050573846833072,    0.13244532790138747,    0.13439907170221357,   0.13636707092642880,   0.13834942886358015,
        0.14034625107486237,    0.14235764543247212,    0.14438372216063469,   0.14642459387834486,   0.14848037564386671,
        0.15055118500103981,    0.15263714202744277,    0.15473836938446797,   0.15685499236936512,   0.15898713896931410,
        0.16113493991759192,    0.16329852875190171,    0.16547804187493589,   0.16767361861725008,   0.16988540130252755,
        0.17211353531531995,    0.17435816917135338,    0.17661945459049480,   0.17889754657247822,   0.18119260347549621,
        0.18350478709776738,    0.18583426276219703,    0.18818119940425421,   0.19054576966319531,   0.19292814997677124,
        0.19532852067956313,    0.19774706610509876,    0.20018397469191118,   0.20263943909370893,   0.20511365629383763,
        0.20760682772422195,    0.21011915938898817,    0.21265086199297820,   0.21520215107537860,   0.21777324714870044,
        0.22036437584335941,    0.22297576805812008,    0.22560766011668396,   0.22826029393071659,   0.23093391716962730,
        0.23362878343743321,    0.23634515245705950,    0.23908329026244904,   0.24184346939887710,   0.24462596913189200,
        0.24743107566532752,    0.25025908236886218,    0.25311029001562935,   0.25598500703041527,   0.25888354974901612,
        0.26180624268936287,    0.26475341883506209,    0.26772541993204468,   0.27072259679905991,   0.27374530965280286,
        0.27679392844851725,    0.27986883323697281,    0.28297041453878069,   0.28609907373707677,   0.28925522348967764,
        0.29243928816189246,    0.29565170428126109,    0.29889292101558163,   0.30216340067569336,   0.30546361924459009,
        0.30879406693456002,    0.31215524877417944,    0.31554768522712878,   0.31897191284495707,   0.32242848495608900,
        0.32591797239355608,    0.32944096426413622,    0.33299806876180887,   0.33658991402867749,   0.34021714906677997,
        0.34388044470450235,    0.34758049462163693,    0.35131801643748328,   0.35509375286678740,   0.35890847294874972,
        0.36276297335481772,    0.36665807978151410,    0.37059464843514595,   0.37457356761590210,   0.37859575940958073,
        0.38266218149600972,    0.38677382908413760,    0.39093173698479700,   0.39513698183329005,   0.39939068447523096,
        0.40369401253053017,    0.40804818315203228,    0.41245446599716107,   0.41691418643300282,   0.42142872899761652,
        0.42599954114303429,    0.43062813728845878,    0.43531610321563652,   0.44006510084235378,   0.44487687341454840,
        0.44975325116275489,    0.45469615747461539,    0.45970761564213758,   0.46478975625042607,   0.46994482528395987,
        0.47517519303737726,    0.48048336393045410,    0.48587198734188480,   0.49134386959403242,   0.49690198724154944,
        0.50254950184134761,    0.50828977641064277,    0.51412639381474845,   0.52006317736823349,   0.52610421398361962,
        0.53225388026304321,    0.53851687200286180,    0.54489823767243961,   0.55140341654064129,   0.55803828226258745,
        0.56480919291240017,    0.57172304866482571,    0.57878735860284491,   0.58601031847726792,   0.59340090169173332,
        0.60096896636523212,    0.60872538207962190,    0.61668218091520743,   0.62485273870366576,   0.63325199421436595,
        0.64189671642726598,    0.65080583341457099,    0.66000084107899970,   0.66950631673192473,   0.67935057226476536,
        0.68956649611707799,    0.70019265508278816,    0.71127476080507601,   0.72286765959357202,   0.73503809243142348,
        0.74786862198519510,    0.76146338884989628,    0.77595685204011566,   0.79152763697249573,   0.80842165152300849,
        0.82699329664305044,    0.84778550062398972,    0.87170433238120371,   0.90046992992574648,   0.93814368086217470,
        1.0000000000000000,
    };
    return ef;
}

}  // namespace fragmites::random

// FRAGMITES_RANDOM
#endif
