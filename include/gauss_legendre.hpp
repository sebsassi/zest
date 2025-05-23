/*
Copyright (c) 2024 Sebastian Sassi

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

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
#pragma once

#include <array>
#include <cmath>
#include <span>
#include <numbers>
#include <ranges>
#include <concepts>
#include <type_traits>
#include <utility>

namespace zest
{
namespace gl
{

/**
    @brief Packed layout of Gauss-Legendre nodes.

    note Gauss-Legendre nodes on the interval [-1,1] are distributed symmetrically about 0, such that for any node `x` the point `-x` is also a node with the same weight. Therefore the nodes and weights only need to be produced for nonnegative `x`. For the negative portion of the interval the nodes are `-x`, and the weights are given by the corresponding weights.
*/
struct PackedLayout
{
    [[nodiscard]] static constexpr std::size_t
    size(std::size_t num_total_nodes) noexcept
    {
        return (num_total_nodes + 1) >> 1;
    }

    [[nodiscard]] static constexpr std::size_t
    total_nodes(std::size_t size, std::size_t parity) noexcept
    {
        return 2*size - parity;
    }
};


/**
    @brief Unpacked layout of Gauss-Legendre nodes.

    note Gauss-Legendre nodes on the interval [-1,1] are distributed symmetrically about 0, such that for any node `x` the point `-x` is also a node with the same weight. Therefore the nodes and weights only need to be produced for nonnegative `x`. For the negative portion of the interval the nodes are `-x`, and the weights are given by the corresponding weights.
*/
struct UnpackedLayout
{
    [[nodiscard]] static constexpr std::size_t
    size(std::size_t num_total_nodes) noexcept
    {
        return num_total_nodes;
    }

    [[nodiscard]] static constexpr std::size_t total_nodes(
        std::size_t size, [[maybe_unused]] std::size_t parity) noexcept
    {
        return size;
    }
};

/**
    @brief Concept for restricting layout of Gauss-Legendre nodes.
*/
template <typename T>
concept gl_layout = std::same_as<T, PackedLayout> || std::same_as<T, UnpackedLayout>;

/**
    @brief Style of Gauss-Legendre nodes.
*/
enum class GLNodeStyle
{
    /** nodes as angles in the interval [0,pi] */
    angle,
    /** nodes as consines of the angles in the interval [-1,1] */
    cos
};

namespace detail
{

/**
    @brief Calculate `k`th zero of the Bessel function J_0.

    @tparam FloatType type of the return value

    @param k order of the zero

    @return value of the zero
*/
template <std::floating_point FloatType>
[[nodiscard]] constexpr FloatType bessel_zero(std::size_t k) noexcept
{
    constexpr std::size_t BESSEL_LUT_MAX = 20;
    constexpr std::array<FloatType, BESSEL_LUT_MAX> bessel_zeros = {
        2.4048255576957727686216318793265,
        5.5200781102863106495966041273312,
        8.6537279129110122169541987126450,
        11.791534439014281613743044912436,
        14.930917708487785947762593997389,
        18.071063967910922543147882975618,
        21.211636629879258959078393350553,
        24.352471530749302737057944763179,
        27.493479132040254795877288234626,
        30.634606468431975117549578926854,
        33.775820213573568684238546346719,
        36.917098353664043979769493063273,
        40.058425764628239294799307373994,
        43.199791713176730357524072728744,
        46.341188371661814018685788879113,
        49.482609897397817173602761533179,
        52.624051841114996029251285380392,
        55.765510755019979311683492773462,
        58.906983926080942132834406634616,
        62.048469190227169882852500264651
    };
    constexpr std::array<FloatType, 5> c = {
        1.0, -124.0/3.0, 120928.0/15.0, -401743168.0/105.0, 1071187749376.0/315.0
    };

    if (k < BESSEL_LUT_MAX) return bessel_zeros[k - 1];

    const FloatType ak = std::numbers::pi_v<FloatType>*(FloatType(k) - 0.25);
    const FloatType x = 0.125/ak;
    const FloatType x2 = x*x;
    return ak + x*(c[0] + x2*(c[1] + x2*(c[2] + x2*(c[3] + x2*c[4]))));
}

/**
    @brief Calculates the square of the Bessel function J_1 evaluated at the `k`th zero of the Bessel function J_0.

    @tparam FloatType type of the return value

    @param k order of the zero

    @return value of square of J_1 at the zero
*/
template <std::floating_point FloatType>
[[nodiscard]] constexpr FloatType bessel_J2_k(std::size_t k) noexcept
{
    constexpr std::size_t BESSEL_LUT_MAX = 20;
    constexpr std::array<FloatType, BESSEL_LUT_MAX> bessel_J2_k_vals = {
        0.2695141239419169261390219929104,
        0.1157801385822036958078128361820,
        0.07368635113640821514064768119854,
        0.05403757319811628204177491827591,
        0.04266142901724309126551060634965,
        0.03524210349099610135874730336482,
        0.03002107010305467267508881576881,
        0.02614739149530808859045846753992,
        0.02315912182469139226526763821780,
        0.02078382912226785760398080572963,
        0.01885045066931766781610568002133,
        0.01724615756966500829952400535420,
        0.01589351810592359780270655942874,
        0.01473762609647218958957429825913,
        0.01373846514538711791828804841348,
        0.01286618173761513287914066372288,
        0.01209805154862679754710754384969,
        0.01141647122449160851686272229866,
        0.01080759279118020401155472868305,
        0.01026037292628076281104239927887
    };
    constexpr std::array<FloatType, 5> c = {
        -7.0/24.0, 151.0/80.0, -172913.0/8064.0, 461797.0/1152.0, -171497088497.0/15206400.0
    };

    if (k < BESSEL_LUT_MAX) return bessel_J2_k_vals[k - 1];

    const FloatType ak = std::numbers::pi_v<FloatType>*(FloatType(k) - 0.25);
    const FloatType x = 1.0/ak;
    const FloatType x2 = x*x;
    return (1.0/std::numbers::pi)*x*(2.0 + x2*x2*(c[0] + x2*(c[1] + x2*(c[2] + x2*(c[3] + x2*c[4])))));
}

template <std::floating_point FloatType, GLNodeStyle node_style_param>
[[nodiscard]] constexpr FloatType gl_node_bogaert(
    FloatType vn_sq, FloatType an_k, FloatType inv_sinc_an_k, FloatType vis_sq, FloatType x) noexcept
{
    // Polynomial coefficients copied from FastGL
    constexpr std::array<FloatType, 7> c_f1 = {
        -0.416666666666662959639712457549e-1,
        +0.416666666665193394525296923981e-2,
        -0.148809523713909147898955880165e-3,
        +0.275573168962061235623801563453e-5,
        -3.13148654635992041468855740012e-8,
        +2.40724685864330121825976175184e-10,
        -1.29052996274280508473467968379e-12
    };
    constexpr std::array<FloatType, 7> c_f2 = {
        +0.815972221772932265640401128517e-2,
        -0.209022248387852902722635654229e-2,
        +0.282116886057560434805998583817e-3,
        -0.253300326008232025914059965302e-4,
        +0.161969259453836261731700382098e-5,
        -7.53036771373769326811030753538e-8,
        +2.20639421781871003734786884322e-9
    };
    constexpr std::array<FloatType, 7> c_f3 = {
        -0.416012165620204364833694266818e-2,
        +0.128654198542845137196151147483e-2,
        -0.251395293283965914823026348764e-3,
        +0.418498100329504574443885193835e-4,
        -0.567797841356833081642185432056e-5,
        +5.55845330223796209655886325712e-7,
        -2.97058225375526229899781956673e-8
    };

    const FloatType f1_cheb = c_f1[0] + x*(c_f1[1] + x*(c_f1[2] + x*(c_f1[3] + x*(c_f1[4] + x*(c_f1[5] + x*c_f1[6])))));
    const FloatType f2_cheb = c_f2[0] + x*(c_f2[1] + x*(c_f2[2] + x*(c_f2[3] + x*(c_f2[4] + x*(c_f2[5] + x*c_f2[6])))));
    const FloatType f3_cheb = c_f3[0] + x*(c_f3[1] + x*(c_f3[2] + x*(c_f3[3] + x*(c_f3[4] + x*(c_f3[5] + x*c_f3[6])))));
    const FloatType f_sum = f1_cheb + vis_sq*(f2_cheb + vis_sq*f3_cheb);

    if constexpr (node_style_param == GLNodeStyle::cos)
        return std::cos(an_k*(1.0 + vn_sq*inv_sinc_an_k*f_sum));
    else
        return an_k*(1.0 + vn_sq*inv_sinc_an_k*f_sum);
}

template <std::floating_point FloatType>
[[nodiscard]] constexpr FloatType gl_weight_bogaert(
    FloatType vn_sq, FloatType inv_sinc_an_k, FloatType vis_sq, FloatType x, std::size_t k) noexcept
{
    // Polynomial coefficients copied from FastGL
    constexpr std::array<FloatType, 10> c_w1 = {
        +0.833333333333333302184063103900e-1,
        -0.305555555555553028279487898503e-1,
        +0.436507936507598105249726413120e-2,
        -0.326278659594412170300449074873e-3,
        +0.149644593625028648361395938176e-4,
        -4.63968647553221331251529631098e-7,
        +1.03756066927916795821098009353e-8,
        -1.75257700735423807659851042318e-10,
        +2.30365726860377376873232578871e-12,
        -2.20902861044616638398573427475e-14
    };
    constexpr std::array<FloatType, 9> c_w2 = {
        -0.111111111111214923138249347172e-1,
        +0.268959435694729660779984493795e-2,
        -0.407297185611335764191683161117e-3,
        +0.465969530694968391417927388162e-4,
        -0.381817918680045468483009307090e-5,
        +2.11483880685947151466370130277e-7,
        -7.12912857233642220650643150625e-9,
        +7.67643545069893130779501844323e-11,
        +3.63117412152654783455929483029e-12
    };
    constexpr std::array<FloatType, 9> c_w3 = {
        +0.656966489926484797412985260842e-2,
        -0.947969308958577323145923317955e-4,
        -0.105646050254076140548678457002e-3,
        -0.422888059282921161626339411388e-4,
        +0.200559326396458326778521795392e-4,
        -0.397933316519135275712977531366e-5,
        +5.08898347288671653137451093208e-7,
        -4.38647122520206649251063212545e-8,
        +2.01826791256703301806643264922e-9
    };

    const FloatType J2_k = bessel_J2_k<FloatType>(k);
    const FloatType beta = J2_k*inv_sinc_an_k;

    const FloatType w1_cheb = c_w1[0] + x*(c_w1[1] + x*(c_w1[2] + x*(c_w1[3] + x*(c_w1[4] + x*(c_w1[5] + x*(c_w1[6] + x*(c_w1[7] + x*(c_w1[8] + x*c_w1[9]))))))));
    const FloatType w2_cheb = c_w2[0] + x*(c_w2[1] + x*(c_w2[2] + x*(c_w2[3] + x*(c_w2[4] + x*(c_w2[5] + x*(c_w2[6] + x*(c_w2[7] + x*c_w2[8])))))));
    const FloatType w3_cheb = c_w3[0] + x*(c_w3[1] + x*(c_w3[2] + x*(c_w3[3] + x*(c_w3[4] + x*(c_w3[5] + x*(c_w3[6] + x*(c_w3[7] + x*c_w3[8])))))));
    const FloatType w_sum = w1_cheb + vis_sq*(w2_cheb + vis_sq*w3_cheb);

    return 2.0*vn_sq/(beta*(1.0 + vis_sq*w_sum));
}

/*
Gauss-Legendre nodes and weights: 
    I. Bogaert, Iteration-free computation of Gauss-Legendre quadrature nodes and weights, SIAM J. Sci. Comput., 36 (2014), pp. C1008-C1026)

The implementation here is based on the FastGL reference implementation by Bogaert.

Accurate to double machine epsilon for `num_points > 70`
*/
template <std::floating_point FloatType, GLNodeStyle node_style_param>
[[nodiscard]] constexpr std::pair<FloatType, FloatType>
gl_node_weight_bogaert(FloatType vn, FloatType vn_sq, std::size_t k) noexcept
{
    const FloatType j0_k = bessel_zero<FloatType>(k);

    const FloatType an_k = vn*j0_k;
    const FloatType sin_an_k = std::sin(an_k);
    const FloatType inv_sinc_an_k = an_k/sin_an_k;
    const FloatType vis_sq = vn_sq*inv_sinc_an_k*inv_sinc_an_k;

    const FloatType x = an_k*an_k;

    return {
        gl_node_bogaert<FloatType, node_style_param>(vn_sq, an_k, inv_sinc_an_k, vis_sq, x),
        gl_weight_bogaert<FloatType>(vn_sq, inv_sinc_an_k, vis_sq, x, k)
    };
}

template <std::floating_point FloatType, GLNodeStyle node_style_param>
[[nodiscard]] constexpr FloatType gl_node_bogaert(
    FloatType vn, FloatType vn_sq, std::size_t k) noexcept
{
    const FloatType j0_k = bessel_zero<FloatType>(k);

    const FloatType an_k = vn*j0_k;
    const FloatType sin_an_k = std::sin(an_k);
    const FloatType inv_sinc_an_k = an_k/sin_an_k;
    const FloatType vis_sq = vn_sq*inv_sinc_an_k*inv_sinc_an_k;

    const FloatType x = an_k*an_k;

    return gl_node_bogaert<FloatType, node_style_param>(
            vn_sq, an_k, inv_sinc_an_k, vis_sq, x);
}

template <std::floating_point FloatType>
[[nodiscard]] constexpr FloatType gl_weight_bogaert(
    FloatType vn, FloatType vn_sq, std::size_t k) noexcept
{
    const FloatType j0_k = bessel_zero<FloatType>(k);

    const FloatType an_k = vn*j0_k;
    const FloatType sin_an_k = std::sin(an_k);
    const FloatType inv_sinc_an_k = an_k/sin_an_k;
    const FloatType vis_sq = vn_sq*inv_sinc_an_k*inv_sinc_an_k;

    const FloatType x = an_k*an_k;

    return gl_weight_bogaert(vn_sq, inv_sinc_an_k, vis_sq, x, k);
}

template <gl_layout Layout, GLNodeStyle node_style_param, std::ranges::random_access_range R>
    requires std::floating_point<
        typename std::remove_reference_t<R>::value_type>
constexpr void gl_nodes_bogaert(R&& nodes, std::size_t parity) noexcept
{
    using FloatType = std::remove_reference_t<R>::value_type;
    if constexpr (std::same_as<Layout, PackedLayout>)
    {
        const std::size_t num_unique_nodes = std::ranges::size(nodes);
        const std::size_t num_nodes = 2*num_unique_nodes - parity;
        const FloatType vn = 1.0/(FloatType(num_nodes) + 0.5);
        const FloatType vn_sq = vn*vn;
        for (std::size_t k = 1; k <= num_unique_nodes; ++k)
        {
            const auto node = gl_node_bogaert<FloatType, node_style_param>(vn, vn_sq, k);
            nodes[num_unique_nodes - k] = node;
        }
    }
    else if constexpr (std::same_as<Layout, UnpackedLayout>)
    {
        const std::size_t num_nodes = std::ranges::size(nodes);
        const std::size_t m = num_nodes >> 1;
        const std::size_t parity = num_nodes & 1;
        const std::size_t num_unique_nodes = m + parity;
        const FloatType vn = 1.0/(FloatType(num_nodes) + 0.5);
        const FloatType vn_sq = vn*vn;
        if (parity)
        {
            const auto node = gl_node_bogaert<FloatType, node_style_param>(
                    vn, vn_sq, num_unique_nodes);
            nodes[m] = node;
            for (std::size_t k = 1; k < num_unique_nodes; ++k)
            {
                const auto node = gl_node_bogaert<FloatType, node_style_param>(
                        vn, vn_sq, k);
                if constexpr (node_style_param == GLNodeStyle::cos)
                {
                    nodes[k - 1] = -node;
                    nodes[num_nodes - k] = node;
                }
                else
                {
                    nodes[k - 1] = std::numbers::pi_v<FloatType> - node;
                    nodes[num_nodes - k] = node;
                }
            }
        }
        else
        {
            for (std::size_t k = 1; k <= num_unique_nodes; ++k)
            {
                const auto node = gl_node_bogaert<FloatType, node_style_param>(
                        vn, vn_sq, k);
                if constexpr (node_style_param == GLNodeStyle::cos)
                {
                    nodes[k - 1] = -node;
                    nodes[num_nodes - k] = node;
                }
                else
                {
                    nodes[k - 1] = std::numbers::pi_v<FloatType> - node;
                    nodes[num_nodes - k] = node;
                }
            }
        }
    }
}

template <gl_layout Layout, std::ranges::random_access_range R>
    requires std::floating_point<
        typename std::remove_reference_t<R>::value_type>
constexpr void gl_weights_bogaert(R&& weights, std::size_t parity) noexcept
{
    using FloatType = std::remove_reference_t<R>::value_type;
    if constexpr (std::same_as<Layout, PackedLayout>)
    {
        const std::size_t num_unique_nodes = std::ranges::size(weights);
        const std::size_t num_nodes = 2*num_unique_nodes - parity;
        const FloatType vn = 1.0/(FloatType(num_nodes) + 0.5);
        const FloatType vn_sq = vn*vn;
        for (std::size_t k = 1; k <= num_unique_nodes; ++k)
        {
            const auto weight = gl_weight_bogaert(vn, vn_sq, k);
            weights[num_unique_nodes - k] = weight;
        }
    }
    else if constexpr (std::same_as<Layout, UnpackedLayout>)
    {
        const std::size_t num_nodes = std::ranges::size(weights);
        const std::size_t m = num_nodes >> 1;
        const std::size_t parity = num_nodes & 1;
        const std::size_t num_unique_nodes = m + parity;
        const FloatType vn = 1.0/(FloatType(num_nodes) + 0.5);
        const FloatType vn_sq = vn*vn;
        if (parity)
        {
            const auto weight = gl_weight_bogaert(
                    vn, vn_sq, num_unique_nodes);
            weights[m] = weight;
            for (std::size_t k = 1; k < num_unique_nodes; ++k)
            {
                const auto weight = gl_node_weight_bogaert(
                        vn, vn_sq, k);
                weights[k - 1] = weight;
                weights[num_nodes - k] = weight;
            }
        }
        else
        {
            for (std::size_t k = 1; k <= num_unique_nodes; ++k)
            {
                const auto weight = gl_node_bogaert(
                        vn, vn_sq, k);
                weights[k - 1] = weight;
                weights[num_nodes - k] = weight;
            }
        }
    }
}

template <gl_layout Layout, GLNodeStyle node_style_param, std::ranges::random_access_range R>
    requires std::floating_point<
        typename std::remove_reference_t<R>::value_type>
constexpr void gl_nodes_and_weights_bogaert(
    R&& nodes, R&& weights, std::size_t parity) noexcept
{
    using FloatType = std::remove_reference_t<R>::value_type;
    if constexpr (std::same_as<Layout, PackedLayout>)
    {
        const std::size_t num_unique_nodes = std::ranges::size(weights);
        const std::size_t num_nodes = 2*num_unique_nodes - parity;
        const FloatType vn = 1.0/(FloatType(num_nodes) + 0.5);
        const FloatType vn_sq = vn*vn;
        for (std::size_t k = 1; k <= num_unique_nodes; ++k)
        {
            const auto& [node, weight]
                    = gl_node_weight_bogaert<FloatType, node_style_param>(vn, vn_sq, k);
            nodes[num_unique_nodes - k] = node;
            weights[num_unique_nodes - k] = weight;
        }
    }
    else if constexpr (std::same_as<Layout, UnpackedLayout>)
    {
        const std::size_t num_nodes = std::ranges::size(weights);
        const std::size_t m = num_nodes >> 1;
        const std::size_t parity = num_nodes & 1;
        const std::size_t num_unique_nodes = m + parity;
        const FloatType vn = 1.0/(FloatType(num_nodes) + 0.5);
        const FloatType vn_sq = vn*vn;
        if (parity)
        {
            const auto& [node, weight]
                    = gl_node_weight_bogaert<FloatType, node_style_param>(
                            vn, vn_sq, num_unique_nodes);
            nodes[m] = node;
            weights[m] = weight;
            for (std::size_t k = 1; k < num_unique_nodes; ++k)
            {
                const auto& [node, weight]
                        = gl_node_weight_bogaert<FloatType, node_style_param>(vn, vn_sq, k);
                nodes[k - 1] = -node;
                nodes[num_nodes - k] = node;
                weights[k - 1] = weight;
                weights[num_nodes - k] = weight;
            }
        }
        else
        {
            for (std::size_t k = 1; k <= num_unique_nodes; ++k)
            {
                const auto& [node, weight]
                        = gl_node_weight_bogaert<FloatType, node_style_param>(vn, vn_sq, k);
                nodes[k - 1] = -node;
                nodes[num_nodes - k] = node;
                weights[k - 1] = weight;
                weights[num_nodes - k] = weight;
            }
        }
    }
}

template <gl_layout Layout, GLNodeStyle node_style_param, std::ranges::random_access_range R>
    requires std::floating_point<
        typename std::remove_reference_t<R>::value_type>
constexpr void gl_nodes_table(R&& nodes, std::size_t parity) noexcept
{
    using FloatType = std::remove_reference_t<R>::value_type;
    // The theta values from FastGL for orders <= 70.
    constexpr const FloatType nodes_2[1] = {
        0.9553166181245092781638573e0};
	constexpr const FloatType nodes_4[2] = {
        0.1223899586470372591854100e1,0.5332956802491269896325121e0};
	constexpr const FloatType nodes_6[3] = {
        0.1329852612388110166006182e1,0.8483666264874876548310910e0,
        0.3696066519448289481138796e0};
	constexpr const FloatType nodes_8[4] = {
        0.1386317078892131346282665e1,0.1017455539490153431016397e1,
        0.6490365804607796100719162e0,0.2827570635937967783987981e0};
	constexpr const FloatType nodes_10[5] = {
        0.1421366498439524924081833e1,0.1122539327631709474018620e1,
        0.8238386589997556048023640e0,0.5255196555285001171749362e0,
        0.2289442988470260178701589e0};
	constexpr const FloatType nodes_12[6] = {
        0.1445233238471440081118261e1,0.1194120375947706635968399e1,
        0.9430552870605735796668951e0,0.6921076988818410126677201e0,
        0.4414870814893317611922530e0,0.1923346793046672033050762e0};
	constexpr const FloatType nodes_14[7] = {
        0.1462529992921481833498746e1,0.1246003586776677662375070e1,
        0.1029498592525136749641068e1,0.8130407055389454598609888e0,
        0.5966877608172733931509619e0,0.3806189306666775272453522e0,
        0.1658171411523664030454318e0};
	constexpr const FloatType nodes_16[8] = {
        0.1475640280808194256470687e1,0.1285331444322965257106517e1,
        0.1095033401803444343034890e1,0.9047575323895165085030778e0,
        0.7145252532340252146626998e0,0.5243866409035941583262629e0,
        0.3344986386876292124968005e0,0.1457246820036738335698855e0};
	constexpr const FloatType nodes_18[9] = {
        0.1485919440392653014379727e1,0.1316167494718022699851110e1,
        0.1146421481056642228295923e1,0.9766871104439832694094465e0,
        0.8069738930788195349918620e0,0.6373005058706191519531139e0,
        0.4677113145328286263205134e0,0.2983460782092324727528346e0,
        0.1299747364196768405406564e0};
	constexpr const FloatType nodes_20[10] = {
        0.1494194914310399553510039e1,0.1340993178589955138305166e1,
        0.1187794926634098887711586e1,0.1034603297590104231043189e1,
        0.8814230742890135843662021e0,0.7282625848696072912405713e0,
        0.5751385026314284688366450e0,0.4220907301111166004529037e0,
        0.2692452880289302424376614e0,0.1172969277059561308491253e0};
	constexpr const FloatType nodes_22[11] = {
        0.1501000399130816063282492e1,0.1361409225664372117193308e1,
        0.1221820208990359866348145e1,0.1082235198111836788818162e1,
        0.9426568273796608630446470e0,0.8030892957063359443650460e0,
        0.6635400754448062852164288e0,0.5240242709487281141128643e0,
        0.3845781703583910933413978e0,0.2453165389983612942439953e0,
        0.1068723357985259945018899e0};
	constexpr const FloatType nodes_24[12] = {
        0.1506695545558101030878728e1,0.1378494427506219143960887e1,
        0.1250294703417272987066314e1,0.1122097523267250692925104e1,
        0.9939044422989454674968570e0,0.8657177770401081355080608e0,
        0.7375413075437535618804594e0,0.6093818382449565759195927e0,
        0.4812531951313686873528891e0,0.3531886675690780704072227e0,
        0.2252936226353075734690198e0,0.9814932949793685067733311e-1};
	constexpr const FloatType nodes_26[13] = {
        0.1511531546703289231944719e1,0.1393002286179807923400254e1,
        0.1274473959424494104852958e1,0.1155947313793812040125722e1,
        0.1037423319077439147088755e1,0.9189033445598992550553862e0,
        0.8003894803353296871788647e0,0.6818851814129298518332401e0,
        0.5633967073169293284500428e0,0.4449368152119130526034289e0,
        0.3265362611165358134766736e0,0.2082924425598466358987549e0,
        0.9074274842993199730441784e-1};
	constexpr const FloatType nodes_28[14] = {
        0.1515689149557281132993364e1,0.1405475003062348722192382e1,
        0.1295261501292316172835393e1,0.1185049147889021579229406e1,
        0.1074838574917869281769567e1,0.9646306371285440922680794e0,
        0.8544265718392254369377945e0,0.7442282945111358297916378e0,
        0.6340389954584301734412433e0,0.5238644768825679339859620e0,
        0.4137165857369637683488098e0,0.3036239070914333637971179e0,
        0.1936769929947376175341314e0,0.8437551461511597225722252e-1};
	constexpr const FloatType nodes_30[15] = {
        0.1519301729274526620713294e1,0.1416312682230741743401738e1,
        0.1313324092045794720169874e1,0.1210336308624476413072722e1,
        0.1107349759228459143499061e1,0.1004365001539081003659288e1,
        0.9013828087667156388167226e0,0.7984043170121235411718744e0,
        0.6954313000299367256853883e0,0.5924667257887385542924194e0,
        0.4895160050896970092628705e0,0.3865901987860504829542802e0,
        0.2837160095793466884313556e0,0.1809780449917272162574031e0,
        0.7884320726554945051322849e-1};
	constexpr const FloatType nodes_32[16] = {
        0.1522469852641529230282387e1,0.1425817011963825344615095e1,
        0.1329164502391080681347666e1,0.1232512573416362994802398e1,
        0.1135861522840293704616614e1,0.1039211728068951568003361e1,
        0.9425636940046777101926515e0,0.8459181315837993237739032e0,
        0.7492760951181414487254243e0,0.6526392394594561548023681e0,
        0.5560103418005302722406995e0,0.4593944730762095704649700e0,
        0.3628020075350028174968692e0,0.2662579994723859636910796e0,
        0.1698418454282150179319973e0,0.7399171309970959768773072e-1};
	constexpr const FloatType nodes_34[17] = {
        0.1525270780617194430047563e1,0.1434219768045409606267345e1,
        0.1343169000217435981125683e1,0.1252118659062444379491066e1,
        0.1161068957629157748792749e1,0.1070020159291475075961444e1,
        0.9789726059789103169325141e0,0.8879267623988119819560021e0,
        0.7968832893748414870413015e0,0.7058431727509840105946884e0,
        0.6148079652926100198490992e0,0.5237802779694730663856110e0,
        0.4327648832448234459097574e0,0.3417715500266717765568488e0,
        0.2508238767288223767569849e0,0.1599966542668327644694431e0,
        0.6970264809814094464033170e-1};
	constexpr const FloatType nodes_36[18] = {
        0.1527764849261740485876940e1,0.1441701954349064743573367e1,
        0.1355639243522655042028688e1,0.1269576852063768424508476e1,
        0.1183514935851550608323947e1,0.1097453683555812711123880e1,
        0.1011393333949027021740881e1,0.9253342019812867059380523e0,
        0.8392767201322475821509486e0,0.7532215073977623159515351e0,
        0.6671694908788198522546767e0,0.5811221342350705406265672e0,
        0.4950819018993074588093747e0,0.4090533017972007314666814e0,
        0.3230455648729987995657071e0,0.2370809940997936908335290e0,
        0.1512302802537625099602687e0,0.6588357082399222649528476e-1};
	constexpr const FloatType nodes_38[19] = {
        0.1529999863223206659623262e1,0.1448406982124841835685420e1,
        0.1366814241651488684482888e1,0.1285221744143731581870833e1,
        0.1203629605904952775544878e1,0.1122037965173751996510051e1,
        0.1040446993107623345153211e1,0.9588569097730895525404200e0,
        0.8772680085516152329147030e0,0.7956806951062012653043722e0,
        0.7140955526031660805347356e0,0.6325134568448222221560326e0,
        0.5509357927460004487348532e0,0.4693648943475422765864580e0,
        0.3878050333015201414955289e0,0.3062649591511896679168503e0,
        0.2247658146033686460963295e0,0.1433746167818849555570557e0,
        0.6246124541276674097388211e-1};
	constexpr const FloatType nodes_40[20] = {
        0.1532014188279762793560699e1,0.1454449946977268522285131e1,
        0.1376885814601482670609845e1,0.1299321869764876494939757e1,
        0.1221758200747475205847413e1,0.1144194910846247537582396e1,
        0.1066632125552939823863593e1,0.9890700026972186303565530e0,
        0.9115087474225932692070479e0,0.8339486352158799520695092e0,
        0.7563900488174808348719219e0,0.6788335401193977027577509e0,
        0.6012799395312684623216685e0,0.5237305617022755897200291e0,
        0.4461876237541810478131970e0,0.3686551849119556335824055e0,
        0.2911415613085158758589405e0,0.2136668503694680525340165e0,
        0.1362947587312224822844743e0,0.5937690028966411906487257e-1};
	constexpr const FloatType nodes_42[21] = {
        0.1533838971193864306068338e1,0.1459924288056445029654271e1,
        0.1386009690354996919044862e1,0.1312095239305276612560739e1,
        0.1238181002944535867235042e1,0.1164267059803796726229370e1,
        0.1090353503721897748980095e1,0.1016440450472067349837507e1,
        0.9425280472651469176638349e0,0.8686164868955467866176243e0,
        0.7947060295895204342519786e0,0.7207970381018823842440224e0,
        0.6468900366403721167107352e0,0.5729858150363658839291287e0,
        0.4990856247464946058899833e0,0.4251915773724379089467945e0,
        0.3513075400485981451355368e0,0.2774414365914335857735201e0,
        0.2036124177925793565507033e0,0.1298811916061515892914930e0,
        0.5658282534660210272754152e-1};
	constexpr const FloatType nodes_44[22] = {
        0.1535499761264077326499892e1,0.1464906652494521470377318e1,
        0.1394313611500109323616335e1,0.1323720686538524176057236e1,
        0.1253127930763390908996314e1,0.1182535404796980113294400e1,
        0.1111943180033868679273393e1,0.1041351343083674290731439e1,
        0.9707600019805773720746280e0,0.9001692951667510715040632e0,
        0.8295794049297955988640329e0,0.7589905782114329186155957e0,
        0.6884031600807736268672129e0,0.6178176499732537480601935e0,
        0.5472348011493452159473826e0,0.4766558078624760377875119e0,
        0.4060826859477620301047824e0,0.3355191279517093844978473e0,
        0.2649727008485465487101933e0,0.1944616940738156405895778e0,
        0.1240440866043499301839465e0,0.5403988657613871827831605e-1};
	constexpr const FloatType nodes_46[23] = {
        0.1537017713608809830855653e1,0.1469460505124226636602925e1,
        0.1401903350962364703169699e1,0.1334346289590505369693957e1,
        0.1266789363044399933941254e1,0.1199232618763735058848455e1,
        0.1131676111906105521856066e1,0.1064119908394702657537061e1,
        0.9965640890815034701957497e0,0.9290087556203499065939494e0,
        0.8614540390091103102510609e0,0.7939001124053586164046432e0,
        0.7263472110048245091518914e0,0.6587956640463586742461796e0,
        0.5912459486086227271608064e0,0.5236987847717837556177452e0,
        0.4561553147193391989386660e0,0.3886174669444433167860783e0,
        0.3210887745896478259115420e0,0.2535764786314617292100029e0,
        0.1860980813776342452540915e0,0.1187090676924131329841811e0,
        0.5171568198966901682810573e-1};
	constexpr const FloatType nodes_48[24] = {
        0.1538410494858190444190279e1,0.1473638845472165977392911e1,
        0.1408867240039222913928858e1,0.1344095709533508756473909e1,
        0.1279324287566779722061664e1,0.1214553011719528935181709e1,
        0.1149781925191718586091000e1,0.1085011078936665906275419e1,
        0.1020240534516704208782618e1,0.9554703680422404498752066e0,
        0.8907006757608306209160649e0,0.8259315822134856671969566e0,
        0.7611632524946588128425351e0,0.6963959112887657683892237e0,
        0.6316298735371143844913976e0,0.5668655960010826255149266e0,
        0.5021037684870694065589284e0,0.4373454855522296089897130e0,
        0.3725925956833896735786860e0,0.3078484858841616878136371e0,
        0.2431200981264999375962973e0,0.1784242126043536701754986e0,
        0.1138140258514833068653307e0,0.4958315373802413441075340e-1};
	constexpr const FloatType nodes_50[25] = {
        0.1539692973716708504412697e1,0.1477486279394502338589519e1,
        0.1415279620944410339318226e1,0.1353073023537942666830874e1,
        0.1290866514321280958405103e1,0.1228660123395079609266898e1,
        0.1166453885011658611362850e1,0.1104247839096738022319035e1,
        0.1042042033248543055386770e1,0.9798365254403234947595400e0,
        0.9176313877712591840677176e0,0.8554267118081827231209625e0,
        0.7932226163976800550406599e0,0.7310192594231560707888939e0,
        0.6688168560730805146438886e0,0.6066157082814543103941755e0,
        0.5444162542389049922529553e0,0.4822191559963931133878621e0,
        0.4200254643636986308379697e0,0.3578369542536859435571624e0,
        0.2956568781922605524959448e0,0.2334919029083292837123583e0,
        0.1713581437497397360313735e0,0.1093066902335822942650053e0,
        0.4761952998197036029817629e-1};
	constexpr const FloatType nodes_52[26] = {
        0.1540877753740080417345045e1,0.1481040617373741365390254e1,
        0.1421203510518656600018143e1,0.1361366453804322852131292e1,
        0.1301529469356044341206877e1,0.1241692581525935716830402e1,
        0.1181855817774264617619371e1,0.1122019209772750368801179e1,
        0.1062182794829879659341536e1,0.1002346617783007482854908e1,
        0.9425107335729934538419206e0,0.8826752108319277463183701e0,
        0.8228401370047382776784725e0,0.7630056258499810562932058e0,
        0.7031718287376427885875898e0,0.6433389522119553277924537e0,
        0.5835072863023426715977658e0,0.5236772521416453354847559e0,
        0.4638494862268433259444639e0,0.4040249990308909882616381e0,
        0.3442054975680110060507306e0,0.2843941101955779333389742e0,
        0.2245972494281051799602510e0,0.1648304164747050021714385e0,
        0.1051427544146599992432949e0,0.4580550859172367960799915e-1};
	constexpr const FloatType nodes_54[27] = {
        0.1541975588842621898865181e1,0.1484334121018556567335167e1,
        0.1426692677652358867201800e1,0.1369051275783071487471360e1,
        0.1311409933595114953831618e1,0.1253768670970438091691833e1,
        0.1196127510146226323327062e1,0.1138486476526912406867032e1,
        0.1080845599717322003702293e1,0.1023204914871722785830020e1,
        0.9655644644970043364617272e0,0.9079243009168822510582606e0,
        0.8502844897148263889326479e0,0.7926451146568312828354346e0,
        0.7350062849078710810840430e0,0.6773681459074923011631400e0,
        0.6197308962817025162722438e0,0.5620948151095422609589585e0,
        0.5044603077892199488064657e0,0.4468279872027509013135997e0,
        0.3891988265038338944044115e0,0.3315744698431505326770711e0,
        0.2739579305700525818998611e0,0.2163553856859193758294342e0,
        0.1587817673749480300092784e0,0.1012844151694839452028589e0,
        0.4412462056235422293371300e-1};
	constexpr const FloatType nodes_56[28] = {
        0.1542995710582548837472073e1,0.1487394484904746766220933e1,
        0.1431793279635669382208875e1,0.1376192108950239363921811e1,
        0.1320590987909222553912422e1,0.1264989932881031519687125e1,
        0.1209388962038683919740547e1,0.1153788095965648154683658e1,
        0.1098187358416032947576489e1,0.1042586777292402877200408e1,
        0.9869863859317282394719449e0,0.9313862248321055503829503e0,
        0.8757863440192765677772914e0,0.8201868063589761051746975e0,
        0.7645876922981545448147078e0,0.7089891068198449136125464e0,
        0.6533911899285832425290628e0,0.5977941329592257586198087e0,
        0.5421982048745539015834188e0,0.4866037965045890355211229e0,
        0.4310114988353693539492225e0,0.3754222503860499120445385e0,
        0.3198376369331602148544626e0,0.2642605649958747239907310e0,
        0.2086969927688100977274751e0,0.1531613237261629042774314e0,
        0.9769922156300582041279299e-1,0.4256272861907242306694832e-1};
	constexpr const FloatType nodes_58[29] = {
        0.1543946088331101630230404e1,0.1490245617072432741470241e1,
        0.1436545162952171175361532e1,0.1382844737841275627385236e1,
        0.1329144354302189376680665e1,0.1275444025914442882448630e1,
        0.1221743767654456436125309e1,0.1168043596353244531685999e1,
        0.1114343531263457295536939e1,0.1060643594778787047442989e1,
        0.1006943813366184678568021e1,0.9532442187977767941200107e0,
        0.8995448498101763729640445e0,0.8458457543830885615091264e0,
        0.7921469929325243736682034e0,0.7384486428849507503612470e0,
        0.6847508053901545384892447e0,0.6310536154445759741044291e0,
        0.5773572576394624029563656e0,0.5236619915567428835581025e0,
        0.4699681944935857341529219e0,0.4162764370726533962791279e0,
        0.3625876255789859906927245e0,0.3089032914359211154562848e0,
        0.2552262416643531728802047e0,0.2015622306384971766058615e0,
        0.1479251692966707827334002e0,0.9435916010280739398532997e-1,
        0.4110762866287674188292735e-1};
	constexpr const FloatType nodes_60[30] = {
        0.1544833637851665335244669e1,0.1492908264756388370493025e1,
        0.1440982906138650837480037e1,0.1389057572001580364167786e1,
        0.1337132272892735072773304e1,0.1285207020157876647295968e1,
        0.1233281826234298389291217e1,0.1181356705000596722238457e1,
        0.1129431672204958843918638e1,0.1077506746001711267258715e1,
        0.1025581947637229234301640e1,0.9736573023432582093437126e0,
        0.9217328405213548692702866e0,0.8698085993416727107979968e0,
        0.8178846249414537373941032e0,0.7659609755086193214466010e0,
        0.7140377257012462393241274e0,0.6621149731355525426273686e0,
        0.6101928481720243483360470e0,0.5582715291407654489802101e0,
        0.5063512668959282414914789e0,0.4544324261262307197237056e0,
        0.4025155584642650335664553e0,0.3506015401168133792671488e0,
        0.2986918517703509333332016e0,0.2467892075469457255751440e0,
        0.1948991714956708008247732e0,0.1430351946011564171352354e0,
        0.9123992133264713232350199e-1,0.3974873026126591246235829e-1};
	constexpr const FloatType nodes_62[31] = {
        0.1545664389841685834178882e1,0.1495400520006868605194165e1,
        0.1445136662469633349524466e1,0.1394872825707861196682996e1,
        0.1344609018631531661347402e1,0.1294345250782284139500904e1,
        0.1244081532562166402923175e1,0.1193817875503760392032898e1,
        0.1143554292597402872188167e1,0.1093290798696377946301336e1,
        0.1043027411028491785799717e1,0.9927641498535133311947588e0,
        0.9425010393224361375194941e0,0.8922381086194002226900769e0,
        0.8419753935054036625982058e0,0.7917129384431112475049142e0,
        0.7414507995789214800057706e0,0.6911890490185720721582180e0,
        0.6409277811053987947460976e0,0.5906671218914768219060599e0,
        0.5404072438741681591850965e0,0.4901483897634232956856935e0,
        0.4398909124691513811974471e0,0.3896353458699818240468259e0,
        0.3393825380385224469051922e0,0.2891339221891949677776928e0,
        0.2388921255071779766209942e0,0.1886625339124777570188312e0,
        0.1384581678870181657476050e0,0.8832030722827102577102185e-1,
        0.3847679847963676404657822e-1};
	constexpr const FloatType nodes_64[32] = {
        0.1546443627125265521960044e1,0.1497738231263909315513507e1,
        0.1449032845902631477147772e1,0.1400327478265391242178337e1,
        0.1351622135921668846451224e1,0.1302916826944702448727527e1,
        0.1254211560091483702838765e1,0.1205506345013417018443405e1,
        0.1156801192508980685500292e1,0.1108096114833249453312212e1,
        0.1059391126084216587933501e1,0.1010686242693213908544820e1,
        0.9619814840575052973573711e0,0.9132768733691264344256970e0,
        0.8645724387181842642305406e0,0.8158682145859558652971026e0,
        0.7671642439014559105969752e0,0.7184605809290069459742089e0,
        0.6697572954095121564500879e0,0.6210544786425143220264938e0,
        0.5723522526623283741373995e0,0.5236507845164779831804685e0,
        0.4749503092950064087413842e0,0.4262511688770346357064771e0,
        0.3775538805043668894422883e0,0.3288592658750793954850446e0,
        0.2801687136893753887834348e0,0.2314847695998852605184853e0,
        0.1828126524563463299986617e0,0.1341649789468091132459783e0,
        0.8558174883654483804697753e-1,0.3728374374031613183399036e-1};
	constexpr const FloatType nodes_66[33] = {
        0.1547175997094614757138430e1,0.1499935340679181525271649e1,
        0.1452694693272706215568985e1,0.1405454061061768876728643e1,
        0.1358213450511184239883293e1,0.1310972868490444296079765e1,
        0.1263732322416537730871712e1,0.1216491820419724046503073e1,
        0.1169251371540540180758674e1,0.1122010985968754004469355e1,
        0.1074770675338453464761893e1,0.1027530453098431393666936e1,
        0.9802903349842005856557204e0,0.9330503396284544173873149e0,
        0.8858104893623263267477775e0,0.8385708112832335506864354e0,
        0.7913313387011139500976360e0,0.7440921131314510897906335e0,
        0.6968531870945337206139839e0,0.6496146281309018959581539e0,
        0.6023765246993705639765525e0,0.5551389950762090311242875e0,
        0.5079022012032895030848024e0,0.4606663710240282967569630e0,
        0.4134318360639670775957014e0,0.3661990979414348851212686e0,
        0.3189689535781378596191439e0,0.2717427498485401725509746e0,
        0.2245229557871702595200694e0,0.1773146332323969343091350e0,
        0.1301300193754780766338959e0,0.8300791095077070533235660e-1,
        0.3616244959900389221395842e-1};
	constexpr const FloatType nodes_68[34] = {
        0.1547865604457777747119921e1,0.1502004162357357213441384e1,
        0.1456142728021903760325049e1,0.1410281306774684706589738e1,
        0.1364419904164498130803254e1,0.1318558526067441138200403e1,
        0.1272697178801115154796514e1,0.1226835869256177571730448e1,
        0.1180974605051351016009903e1,0.1135113394719709026888693e1,
        0.1089252247936466574864114e1,0.1043391175801911243726755e1,
        0.9975301911979639874925565e0,0.9516693092438447484954432e0,
        0.9058085478865097428655118e0,0.8599479286766250282572181e0,
        0.8140874778035996603018790e0,0.7682272274981820559251592e0,
        0.7223672179660643783333797e0,0.6765075001043380283085699e0,
        0.6306481393987597674748178e0,0.5847892216487432573582268e0,
        0.5389308616059791284685642e0,0.4930732164176132508179420e0,
        0.4472165073094733435432890e0,0.4013610560689043520551232e0,
        0.3555073496130768130758891e0,0.3096561615434305328219637e0,
        0.2638087993597793691714182e0,0.2179676599607749036552390e0,
        0.1721376573496165890967450e0,0.1263306713881449555499955e0,
        0.8058436603519718986295825e-1,0.3510663068970053260227480e-1};
	constexpr const FloatType nodes_70[35] = {
        0.1548516088202564202943238e1,0.1503955613246577879586994e1,
        0.1459395145012190281751360e1,0.1414834688100222735099866e1,
        0.1370274247295441414922756e1,0.1325713827649021532002630e1,
        0.1281153434570536124285912e1,0.1236593073933169034954499e1,
        0.1192032752196710979323473e1,0.1147472476554108430135576e1,
        0.1102912255109027578275434e1,0.1058352097094263144928973e1,
        0.1013792013144153206047048e1,0.9692320156388929821870602e0,
        0.9246721191454417746654622e0,0.8801123409896300773149632e0,
        0.8355527020087518049947413e0,0.7909932275560464363973909e0,
        0.7464339488624693592395086e0,0.7018749049145358048463504e0,
        0.6573161450929179933243905e0,0.6127577329584494909986789e0,
        0.5681997518140860838771656e0,0.5236423130979094957496400e0,
        0.4790855694444512920982626e0,0.4345297357523596151738496e0,
        0.3899751246318782591316393e0,0.3454222091410984787772492e0,
        0.3008717408917773811461237e0,0.2563249902500918978614004e0,
        0.2117842860782107775954396e0,0.1672544029381415755198150e0,
        0.1227468836419337342946123e0,0.7829832364814667171382217e-1,
        0.3411071484766340151578357e-1};
	
    constexpr const FloatType nodes_1[1] {
        0.1570796326794896619231321e1
    };
	constexpr const FloatType nodes_3[2] = {
        0.1570796326794896619231321e1,0.6847192030022829138880982e0};
	constexpr const FloatType nodes_5[3] = {
        0.1570796326794896619231321e1,0.1002176803643121641749915e1,
        0.4366349492255221620374655e0};
	constexpr const FloatType nodes_7[4] = {
        0.1570796326794896619231321e1,0.1152892953722227341986065e1,
        0.7354466143229520469385622e0,0.3204050902900619825355950e0};
	constexpr const FloatType nodes_9[5] = {
        0.1570796326794896619231321e1,0.1240573923404363422789550e1,
        0.9104740292261473250358755e0,0.5807869795060065580284919e0,
        0.2530224166119306882187233e0};
	constexpr const FloatType nodes_11[6] = {
        0.1570796326794896619231321e1,0.1297877729331450368298142e1,
        0.1025003226369574843297844e1,0.7522519395990821317003373e0,
        0.4798534223256743217333579e0,0.2090492874137409414071522e0};
	constexpr const FloatType nodes_13[7] = {
        0.1570796326794896619231321e1,0.1338247676100454369194835e1,
        0.1105718066248490075175419e1,0.8732366099401630367220948e0,
        0.6408663264733867770811230e0,0.4088002373420211722955679e0,
        0.1780944581262765470585931e0};
	constexpr const FloatType nodes_15[8] = {
        0.1570796326794896619231321e1,0.1368219536992351783359098e1,
        0.1165652065603030148723847e1,0.9631067821301481995711685e0,
        0.7606069572889918619145483e0,0.5582062109125313357140248e0,
        0.3560718303314725022788878e0,0.1551231069747375098418591e0};
	constexpr const FloatType nodes_17[9] = {
        0.1570796326794896619231321e1,0.1391350647015287461874435e1,
        0.1211909966211469688151240e1,0.1032480728417239563449772e1,
        0.8530732514258505686069670e0,0.6737074594242522259878462e0,
        0.4944303818194983217354808e0,0.3153898594929282395996014e0,
        0.1373998952992547671039022e0};
	constexpr const FloatType nodes_19[10] = {
        0.1570796326794896619231321e1,0.1409742336767428999667236e1,
        0.1248691224331339221187704e1,0.1087646521650454938943641e1,
        0.9266134127998189551499083e0,0.7656007620508340547558669e0,
        0.6046261769405451549818494e0,0.4437316659960951760051408e0,
        0.2830497588453068048261493e0,0.1233108673082312764916251e0};
	constexpr const FloatType nodes_21[11] = {
        0.1570796326794896619231321e1,0.1424715475176742734932665e1,
        0.1278636375242898727771561e1,0.1132561101012537613667002e1,
        0.9864925055883793730483278e0,0.8404350520135058972624775e0,
        0.6943966110110701016065380e0,0.5483930281810389839680525e0,
        0.4024623099018152227701990e0,0.2567245837448891192759858e0,
        0.1118422651428890834760883e0};
	constexpr const FloatType nodes_23[12] = {
        0.1570796326794896619231321e1,0.1437141935303526306632113e1,
        0.1303488659735581140681362e1,0.1169837785762829821262819e1,
        0.1036190996404462300207004e1,0.9025507517347875930425807e0,
        0.7689210263823624893974324e0,0.6353089402976822861185532e0,
        0.5017289283414202278167583e0,0.3682157131008289798868520e0,
        0.2348791589702580223688923e0,0.1023252788872632487579640e0};
	constexpr const FloatType nodes_25[13] = {
        0.1570796326794896619231321e1,0.1447620393135667144403507e1,
        0.1324445197736386798102445e1,0.1201271573324181312770120e1,
        0.1078100568411879956441542e1,0.9549336362382321811515336e0,
        0.8317729718814276781352878e0,0.7086221837538611370849622e0,
        0.5854877911108011727748238e0,0.4623830630132757357909198e0,
        0.3393399712563371486343129e0,0.2164597408964339264361902e0,
        0.9430083986305519349231898e-1};
	constexpr const FloatType nodes_27[14] = {
        0.1570796326794896619231321e1,0.1456575541704195839944967e1,
        0.1342355260834552126304154e1,0.1228136043468909663499174e1,
        0.1113918572282611841378549e1,0.9997037539874953933323299e0,
        0.8854928869950799998575862e0,0.7712879690777516856072467e0,
        0.6570923167092416238233585e0,0.5429119513798658239789812e0,
        0.4287591577660783587509129e0,0.3146635662674373982102762e0,
        0.2007190266590380629766487e0,0.8744338280630300217927750e-1};
	constexpr const FloatType nodes_29[15] = {
        0.1570796326794896619231321e1,0.1464317002991565219979113e1,
        0.1357838033080061766980173e1,0.1251359804334884770836945e1,
        0.1144882777708662655968171e1,0.1038407544520296695714932e1,
        0.9319349156915986836657782e0,0.8254660749671546663859351e0,
        0.7190028636037068047812305e0,0.6125483562383020473196681e0,
        0.5061081521562999836102547e0,0.3996936914666951732317457e0,
        0.2933325857619472952507468e0,0.1871123137498061864373407e0,
        0.8151560650977882057817999e-1};
	constexpr const FloatType nodes_31[16] = {
        0.1570796326794896619231321e1,0.1471075823713997440657641e1,
        0.1371355574944658989649887e1,0.1271635855736122280723838e1,
        0.1171916986981363820797100e1,0.1072199368669106404814915e1,
        0.9724835301003496870596165e0,0.8727702114891848603047954e0,
        0.7730605060747958359120755e0,0.6733561257504194406005404e0,
        0.5736599396529727772420934e0,0.4739771829190733570809765e0,
        0.3743185619229329461021810e0,0.2747099287638327553949437e0,
        0.1752332025619508475799133e0,0.7634046205384429302353073e-1};
	constexpr const FloatType nodes_33[17] = {
        0.1570796326794896619231321e1,0.1477027911291552393547878e1,
        0.1383259682348271685979143e1,0.1289491840051302622319481e1,
        0.1195724613675799550484673e1,0.1101958282220461402990667e1,
        0.1008193204014774090964219e1,0.9144298626454031699590564e0,
        0.8206689427646120483710056e0,0.7269114630504563073034288e0,
        0.6331590254855162126233733e0,0.5394143214244183829842424e0,
        0.4456822679082866369288652e0,0.3519729273095236644049666e0,
        0.2583106041071417718760275e0,0.1647723231643112502628240e0,
        0.7178317184275122449502857e-1};
	constexpr const FloatType nodes_35[18] = {
        0.1570796326794896619231321e1,0.1482309554825692463999299e1,
        0.1393822922226542123661077e1,0.1305336577335833571381699e1,
        0.1216850687682353365944624e1,0.1128365453024608460982204e1,
        0.1039881123511957522668140e1,0.9513980267579228357946521e0,
        0.8629166105524045911461307e0,0.7744375139383604902604254e0,
        0.6859616923374368587817328e0,0.5974906525247623278123711e0,
        0.5090269299866796725116786e0,0.4205751610647263669405267e0,
        0.3321448379994943116084719e0,0.2437588931448048912587688e0,
        0.1554900095178924564386865e0,0.6773932498157585698088354e-1};
	constexpr const FloatType nodes_37[19] = {
        0.1570796326794896619231321e1,0.1487027983239550912222135e1,
        0.1403259745496922270264564e1,0.1319491725464661433609663e1,
        0.1235724047968681189212364e1,0.1151956859289811446164825e1,
        0.1068190338689553494802072e1,0.9844247150109837231349622e0,
        0.9006602918737365182850484e0,0.8168974877846821404275069e0,
        0.7331369031796229223580227e0,0.6493794386888650054486281e0,
        0.5656265174356596757139537e0,0.4818805368222631487731579e0,
        0.3981458834052590173509113e0,0.3144315409387123154212535e0,
        0.2307592167302372059759857e0,0.1471977156945989772472748e0,
        0.6412678117309944052403703e-1};
	constexpr const FloatType nodes_39[20] = {
        0.1570796326794896619231321e1,0.1491268718102344688271411e1,
        0.1411741190914640487505771e1,0.1332213830951015404441941e1,
        0.1252686732830809999680267e1,0.1173160005794509313174730e1,
        0.1093633781237958896879965e1,0.1014108223243148393065201e1,
        0.9345835440325075907377330e0,0.8550600276575269107773349e0,
        0.7755380679025248517258532e0,0.6960182317959841585145109e0,
        0.6165013717819833504477346e0,0.5369888366794912945318079e0,
        0.4574829005269902932408889e0,0.3779877260196973978940863e0,
        0.2985118404618624984946326e0,0.2190758506462427957069113e0,
        0.1397450765119767349146353e0,0.6088003363863534825005464e-1};
	constexpr const FloatType nodes_41[21] = {
        0.1570796326794896619231321e1,0.1495100801651051409999732e1,
        0.1419405340110198552778393e1,0.1343710008748627892724810e1,
        0.1268014880389353000310414e1,0.1192320038028903827079750e1,
        0.1116625579891689469044026e1,0.1040931626310454794079799e1,
        0.9652383295306942866661884e0,0.8895458882533946571137358e0,
        0.8138545700535261740447950e0,0.7381647473570304814395029e0,
        0.6624769578126105498149624e0,0.5867920109947446493391737e0,
        0.5111111891461744489290992e0,0.4354366553151050147918632e0,
        0.3597723703299625354660452e0,0.2841264494060559943920389e0,
        0.2085185052177154996230005e0,0.1330107089065635461375419e0,
        0.5794620170990797798650123e-1};
	constexpr const FloatType nodes_43[22] = {
        0.1570796326794896619231321e1,0.1498580583401444174317386e1,
        0.1426364890228584522673414e1,0.1354149299629923281192036e1,
        0.1281933868420423988034246e1,0.1209718660626713399048551e1,
        0.1137503750956414845248481e1,0.1065289229411733880607916e1,
        0.9930752076949068878557126e0,0.9208618284397049456535757e0,
        0.8486492789905562098591586e0,0.7764378127156926158031943e0,
        0.7042277832708635930867344e0,0.6320197021480767602848178e0,
        0.5598143404345395912377042e0,0.4876129202946139420188428e0,
        0.4154175043169533365541148e0,0.3432318703096418027524597e0,
        0.2710637595435203246492797e0,0.1989318822110657561806962e0,
        0.1268955503926593166308254e0,0.5528212871240371048241379e-1};
	constexpr const FloatType nodes_45[23] = {
        0.1570796326794896619231321e1,0.1501754508594837337089856e1,
        0.1432712730475143340404518e1,0.1363671034069754274950592e1,
        0.1294629464249430679064317e1,0.1225588071083248538559259e1,
        0.1156546912269029268686830e1,0.1087506056298747798071893e1,
        0.1018465586752840651469411e1,0.9494256083335850798964741e0,
        0.8803862556198167553278643e0,0.8113477061841624760598814e0,
        0.7423102009244498727845341e0,0.6732740767851639064676858e0,
        0.6042398217472142478598295e0,0.5352081720899522889584566e0,
        0.4661802954366277026594659e0,0.3971581629712621730826920e0,
        0.3281453857685808451825081e0,0.2591493642052661979197670e0,
        0.1901879854885491785792565e0,0.1213179541186130699071317e0,
        0.5285224511635143601147552e-1};
	constexpr const FloatType nodes_47[24] = {
        0.1570796326794896619231321e1,0.1504661202517196460191540e1,
        0.1438526110541037227495230e1,0.1372391084315255737540026e1,
        0.1306256159670931796771616e1,0.1240121376243315949825014e1,
        0.1173986779205849344923421e1,0.1107852421486856229076325e1,
        0.1041718366715156747157745e1,0.9755846932657442605621389e0,
        0.9094514999854931965227238e0,0.8433189145364798253029042e0,
        0.7771871059265138564989363e0,0.7110563039566125173946002e0,
        0.6449268305419475123120585e0,0.5787991523675322133651034e0,
        0.5126739740395088296453592e0,0.4465524134105889084933393e0,
        0.3804363581140941600870992e0,0.3143292666717729726674543e0,
        0.2482382273986418438740754e0,0.1821803739336923550363257e0,
        0.1162100228791666307841708e0,0.5062697144246344520692308e-1};
	constexpr const FloatType nodes_49[25] = {
        0.1570796326794896619231321e1,0.1507333049739684406957329e1,
        0.1443869798951040686809862e1,0.1380406601553595646811530e1,
        0.1316943486448336467960940e1,0.1253480485358734060913055e1,
        0.1190017634088428795118215e1,0.1126554974102287077081806e1,
        0.1063092554588577221978254e1,0.9996304352342330000643921e0,
        0.9361686900661624628632729e0,0.8727074129127595264965883e0,
        0.8092467253835331800652228e0,0.7457867888716805068068402e0,
        0.6823278231980088937854296e0,0.6188701366516795329577182e0,
        0.5554141765061554178407906e0,0.4919606183965743300387332e0,
        0.4285105345527885639657014e0,0.3650657359209552112046854e0,
        0.3016295408979540017854803e0,0.2382087510453128743250072e0,
        0.1748198074104535338147956e0,0.1115148317291502081079519e0,
        0.4858150828905663931389750e-1};
	constexpr const FloatType nodes_51[26] = {
        0.1570796326794896619231321e1,0.1509797405521643600800862e1,
        0.1448798505784201776188819e1,0.1387799649767640868379247e1,
        0.1326800860997572277878513e1,0.1265802165120213614545418e1,
        0.1204803590828283748583827e1,0.1143805171007496028164312e1,
        0.1082806944206958485218487e1,0.1021808956582037259849130e1,
        0.9608112645303606832220554e0,0.8998139383584991342974664e0,
        0.8388170675106567024157190e0,0.7778207682214244793380700e0,
        0.7168251950382156442798800e0,0.6558305587295081487906238e0,
        0.5948371551492265376377962e0,0.5338454137827292925154468e0,
        0.4728559836463229599006206e0,0.4118698949811841042358258e0,
        0.3508888880839026413717319e0,0.2899161521835467942607342e0,
        0.2289582244272697168835150e0,0.1680309071251709912058722e0,
        0.1071842976730454709494914e0,0.4669490825917857848258897e-1};
	constexpr const FloatType nodes_53[27] = {
        0.1570796326794896619231321e1,0.1512077535592702651885542e1,
        0.1453358762182399391553360e1,0.1394640024852448295479492e1,
        0.1335921342914185177270306e1,0.1277202737290683500323248e1,
        0.1218484231207691826029908e1,0.1159765851037557179133987e1,
        0.1101047627365156083369632e1,0.1042329596373083545617043e1,
        0.9836118016874520301049009e0,0.9248942968954766185908511e0,
        0.8661771490588063053774554e0,0.8074604437333368789787031e0,
        0.7487442923247565105494255e0,0.6900288431709550365296138e0,
        0.6313142987730108226833704e0,0.5726009435739572428629866e0,
        0.5138891906843943809444838e0,0.4551796645660731149033106e0,
        0.3964733566771858874923011e0,0.3377719420068963817561906e0,
        0.2790784903284342592940125e0,0.2203992941938221111139898e0,
        0.1617495649772923108686624e0,0.1031775271253784724197264e0,
        0.4494935602951385601335598e-1};
	constexpr const FloatType nodes_55[28] = {
        0.1570796326794896619231321e1,0.1514193352804819997509006e1,
        0.1457590393617468793209691e1,0.1400987464419153080392546e1,
        0.1344384581184662080889348e1,0.1287781761126833878758488e1,
        0.1231179023218584237510462e1,0.1174576388822640925688125e1,
        0.1117973882475943676285829e1,0.1061371532893653466992815e1,
        0.1004769374285310770780417e1,0.9481674481184788854172919e0,
        0.8915658055327279211293483e0,0.8349645107156934027761499e0,
        0.7783636457331086848148917e0,0.7217633176118399859733190e0,
        0.6651636690166557413471029e0,0.6085648948549621671933311e0,
        0.5519672690500084950513985e0,0.4953711895788266953367288e0,
        0.4387772581729219934583483e0,0.3821864303519236078766179e0,
        0.3256003205491779498477363e0,0.2690218877324958059454348e0,
        0.2124571975249336244841297e0,0.1559209129891515317090843e0,
        0.9945952063842375053227931e-1,0.4332960406341033436157524e-1};
	constexpr const FloatType nodes_57[29] = {
        0.1570796326794896619231321e1,0.1516162000094549207021851e1,
        0.1461527685790782385188426e1,0.1406893396579229558427657e1,
        0.1352259145769086826235918e1,0.1297624947629923059740243e1,
        0.1242990817790597917328601e1,0.1188356773715062198539162e1,
        0.1133722835287525783953663e1,0.1079089025551156002698850e1,
        0.1024455371662101389801169e1,0.9698219061474760364582928e0,
        0.9151886685974009713577537e0,0.8605557079864861100238346e0,
        0.8059230859253162466918892e0,0.7512908813164713594661588e0,
        0.6966591971861112012613682e0,0.6420281709850565965229799e0,
        0.5873979906122764301937499e0,0.5327689202536826556885353e0,
        0.4781413438508069051295597e0,0.4235158420269503798571552e0,
        0.3688933369002844229314675e0,0.3142753865947702189467806e0,
        0.2596648470121556361200229e0,0.2050675726616484232653526e0,
        0.1504977164639767777858359e0,0.9600014792058154736462106e-1,
        0.4182252607645932321862773e-1};
	constexpr const FloatType nodes_59[30] = {
        0.1570796326794896619231321e1,0.1517998315905975681819213e1,
        0.1465200315462026532129551e1,0.1412402336143180968579639e1,
        0.1359604389111228213837104e1,0.1306806486279734731351497e1,
        0.1254008640622089183072742e1,0.1201210866535131048800458e1,
        0.1148413180281179970113571e1,0.1095615600538999408768381e1,
        0.1042818149105710558651372e1,0.9900208518088600875617620e0,
        0.9372237397138955502862203e0,0.8844268507524555199840381e0,
        0.8316302319600398731649744e0,0.7788339426133210890795576e0,
        0.7260380587255163256281298e0,0.6732426796448045921910045e0,
        0.6204479380061240544867289e0,0.5676540152134466427854705e0,
        0.5148611664077887834613451e0,0.4620697624728053757183766e0,
        0.4092803643735033357684553e0,0.3564938631002461237979451e0,
        0.3037117642790043703921396e0,0.2509368276982060978106092e0,
        0.1981747109679032915697317e0,0.1454390911823840643137232e0,
        0.9277332955453467429763451e-1,0.4041676055113025684436480e-1};
	constexpr const FloatType nodes_61[31] = {
        0.1570796326794896619231321e1,0.1519715208823086817411929e1,
        0.1468634099702062550682430e1,0.1417553008469014674939490e1,
        0.1366471944542347269659860e1,0.1315390917933946912760115e1,
        0.1264309939489363760018555e1,0.1213229021168654147755139e1,
        0.1162148176384137345494752e1,0.1111067420416500738111992e1,
        0.1059986770938296676746064e1,0.1008906248685091746434581e1,
        0.9578258783312407255956784e0,0.9067456896525242756445150e0,
        0.8556657190967860708153477e0,0.8045860119448479090873824e0,
        0.7535066253423996943740445e0,0.7024276326462752642452137e0,
        0.6513491298057893513225544e0,0.6002712449887427739045163e0,
        0.5491941535583390603837715e0,0.4981181022276018128369963e0,
        0.4470434496975185070560821e0,0.3959707385770101868486847e0,
        0.3449008307748737032772825e0,0.2938351828535981363494671e0,
        0.2427764647581323719392653e0,0.1917301500230701193408602e0,
        0.1407094708800750523796875e0,0.8975637836633630394302762e-1,
        0.3910242380354419363081899e-1};
	constexpr const FloatType nodes_63[32] = {
        0.1570796326794896619231321e1,0.1521323961422700444944464e1,
        0.1471851603590422118622546e1,0.1422379260986849454727777e1,
        0.1372906941604798453293218e1,0.1323434653909307929892118e1,
        0.1273962407026590487708892e1,0.1224490210963055761921526e1,
        0.1175018076866133593082748e1,0.1125546017342156230227131e1,
        0.1076074046851682267877939e1,0.1026602182210094558879809e1,
        0.9771304432322302018639612e0,0.9276588535760335871906045e0,
        0.8781874418647315968408864e0,0.8287162432047307488040550e0,
        0.7792453012756761070555010e0,0.7297746712644485550469075e0,
        0.6803044240724808212528033e0,0.6308346524943159683026367e0,
        0.5813654805388740483542438e0,0.5318970779332963132260134e0,
        0.4824296835154055410257004e0,0.4329636445908698102350729e0,
        0.3834994865870752458854056e0,0.3340380441799942088370002e0,
        0.2845807279748544733570760e0,0.2351301237470960623526672e0,
        0.1856915325646991222655151e0,0.1362777698319134965765757e0,
        0.8692946525012054120187353e-1,0.3787087726949234365520114e-1};
	constexpr const FloatType nodes_65[33] = {
        0.1570796326794896619231321e1,0.1522834478472358672931947e1,
        0.1474872636605138418026177e1,0.1426910807768284322082436e1,
        0.1378948998781055367310047e1,0.1330987216841224680164684e1,
        0.1283025469674968454386883e1,0.1235063765709222885799986e1,
        0.1187102114275073728898860e1,0.1139140525853183114841234e1,
        0.1091179012375759666645271e1,0.1043217587604604879578741e1,
        0.9952562676120370548458597e0,0.9472950714021223337048082e0,
        0.8993340217254078241758816e0,0.8513731461641338285808219e0,
        0.8034124786014693431693904e0,0.7554520612457602368887930e0,
        0.7074919474732165281510693e0,0.6595322059052657580628641e0,
        0.6115729263971504325174172e0,0.5636142290734363894767612e0,
        0.5156562783879918167717991e0,0.4676993058012953469089537e0,
        0.4197436479350834076514896e0,0.3717898140987174444032373e0,
        0.3238386134116156886828960e0,0.2758914133405791810724762e0,
        0.2279507206431424610498769e0,0.1800216744637006612298520e0,
        0.1321166988439841543825694e0,0.8427518284958235696897899e-1,
        0.3671453742186897322954009e-1};
	constexpr const FloatType nodes_67[34] = {
        0.1570796326794896619231321e1,0.1524255491013576804195881e1,
        0.1477714660784952783237945e1,0.1431173841758652772349485e1,
        0.1384633039781787069436630e1,0.1338092261006965672253841e1,
        0.1291551512012124788593875e1,0.1245010799937299944123195e1,
        0.1198470132644670409416924e1,0.1151929518909907204916554e1,
        0.1105388968655282680015213e1,0.1058848493238442193822372e1,
        0.1012308105815651361079674e1,0.9657678218054126734684090e0,
        0.9192276594886802366068293e0,0.8726876407972167893294764e0,
        0.8261477923647281669131478e0,0.7796081469509049827753598e0,
        0.7330687454042532567721262e0,0.6865296394193009886613469e0,
        0.6399908954920466591029822e0,0.5934526007301573325059582e0,
        0.5469148716199143611697357e0,0.5003778676688561814362271e0,
        0.4538418134105091550464446e0,0.4073070354279485829740435e0,
        0.3607740278788822846227453e0,0.3142435758510728338330843e0,
        0.2677170062389944640113953e0,0.2211967514739567668334169e0,
        0.1746877983807874325844051e0,0.1282022028383479964348629e0,
        0.8177818680168764430245080e-1,0.3562671947817428176226631e-1};
	constexpr const FloatType nodes_69[35] = {
        0.1570796326794896619231321e1,0.1525594725214770881206476e1,
        0.1480393128432045740356817e1,0.1435191541323085582529217e1,
        0.1389989968924959812091252e1,0.1344788416522907866060817e1,
        0.1299586889746827997174554e1,0.1254385394680661996389736e1,
        0.1209183937989395175829969e1,0.1163982527069600127515982e1,
        0.1118781170231154762473596e1,0.1073579876920155012130433e1,
        0.1028378657996412636748477e1,0.9831775260837211038023103e0,
        0.9379764960179657076015136e0,0.8927755854282048597997986e0,
        0.8475748155007347757967789e0,0.8023742119985848209905761e0,
        0.7571738066433708695662393e0,0.7119736390205872251796930e0,
        0.6667737592565460745639184e0,0.6215742318591892056934095e0,
        0.5763751413603713322640298e0,0.5311766008298875656047892e0,
        0.4859787651249621588538330e0,0.4407818522612533891543536e0,
        0.3955861793705505114602136e0,0.3503922263398633798966312e0,
        0.3052007556167344348049303e0,0.2600130558662051177480644e0,
        0.2148314894784555841956251e0,0.1696608997322034095150907e0,
        0.1245129955389270002683579e0,0.7942489891978153749097006e-1,
        0.3460150809198016850782325e-1};


    const std::array<const FloatType*, 71> node_arrs = {
        nullptr, nodes_1, nodes_2, nodes_3, nodes_4, nodes_5, nodes_6, nodes_7,
        nodes_8, nodes_9, nodes_10, nodes_11, nodes_12, nodes_13, nodes_14,
        nodes_15, nodes_16, nodes_17, nodes_18, nodes_19, nodes_20, nodes_21,
        nodes_22, nodes_23, nodes_24, nodes_25, nodes_26, nodes_27, nodes_28,
        nodes_29, nodes_30, nodes_31, nodes_32, nodes_33, nodes_34, nodes_35,
        nodes_36, nodes_37, nodes_38, nodes_39, nodes_40, nodes_41, nodes_42,
        nodes_43, nodes_44, nodes_45, nodes_46, nodes_47, nodes_48, nodes_49,
        nodes_50, nodes_51, nodes_52, nodes_53, nodes_54, nodes_55, nodes_56,
        nodes_57, nodes_58, nodes_59, nodes_60, nodes_61, nodes_62, nodes_63,
        nodes_64, nodes_65, nodes_66, nodes_67, nodes_68, nodes_69, nodes_70
    };

    if (std::ranges::size(nodes) == 0) return;

    if constexpr (std::same_as<Layout, PackedLayout>)
    {
        const std::size_t num_unique_nodes = std::ranges::size(nodes);
        const std::size_t num_nodes = 2*num_unique_nodes - parity;
        const FloatType* node_arr = node_arrs[num_nodes];
        for (std::size_t i = 0; i < num_unique_nodes; ++i)
        {
            if constexpr (node_style_param == GLNodeStyle::cos)
                nodes[i] = std::cos(node_arr[i]);
            else
                nodes[i] = node_arr[i];

        }
    }
    else if constexpr (std::same_as<Layout, UnpackedLayout>)
    {
        const std::size_t num_nodes = std::ranges::size(nodes);
        const std::size_t m = num_nodes >> 1;
        const std::size_t parity = num_nodes & 1;
        const std::size_t num_unique_nodes = m + parity;
        const FloatType* node_arr = node_arrs[num_nodes];
        if (parity)
        {
            if constexpr (node_style_param == GLNodeStyle::cos)
                nodes[m] = std::cos(node_arr[0]);
            else
                nodes[m] = node_arr[0];

            for (std::size_t i = 1; i < num_unique_nodes; ++i)
            {
                if constexpr (node_style_param == GLNodeStyle::cos)
                {
                    const FloatType node = std::cos(node_arr[i]);
                    nodes[m - i] = -node;
                    nodes[m + i] = node;
                }
                else
                {
                    nodes[m - i] = std::numbers::pi_v<FloatType> - node_arr[i];
                    nodes[m + i] = node_arr[i];
                }
            }
        }
        else
        {
            for (std::size_t i = 0; i < num_unique_nodes; ++i)
            {
                if constexpr (node_style_param == GLNodeStyle::cos)
                {
                    const FloatType node = std::cos(node_arr[i]);
                    nodes[m - i - 1] = -node;
                    nodes[m + i] = node;
                }
                else
                {
                    nodes[m - i - 1] = std::numbers::pi_v<FloatType> - node_arr[i];
                    nodes[m + i] = node_arr[i];
                }
            }
        }
    }
}

template <gl_layout Layout, std::ranges::random_access_range R>
    requires std::floating_point<
        typename std::remove_reference_t<R>::value_type>
constexpr void gl_weights_table(R&& weights, std::size_t parity) noexcept
{
    using FloatType = std::remove_reference_t<R>::value_type;
    // Weight values from FastGL for orders <= 70
    constexpr const FloatType weights_2[] = {1.0};
    constexpr const FloatType weights_4[] = {0.6521451548625461426269364, 0.3478548451374538573730642};
    constexpr const FloatType weights_6[] = {0.4679139345726910473898704, 0.3607615730481386075698336, 0.1713244923791703450402969};
    constexpr const FloatType weights_8[] = {0.3626837833783619829651504, 0.3137066458778872873379622, 0.2223810344533744705443556, 0.1012285362903762591525320};
    constexpr const FloatType weights_10[] = {0.2955242247147528701738930, 0.2692667193099963550912268, 0.2190863625159820439955350, 0.1494513491505805931457764, 0.6667134430868813759356850e-1};
    constexpr const FloatType weights_12[] = {0.2491470458134027850005624, 0.2334925365383548087608498, 0.2031674267230659217490644, 0.1600783285433462263346522, 0.1069393259953184309602552, 0.4717533638651182719461626e-1};
    constexpr const FloatType weights_14[] = {0.2152638534631577901958766, 0.2051984637212956039659240, 0.1855383974779378137417164, 0.1572031671581935345696019, 0.1215185706879031846894145, 0.8015808715976020980563266e-1, 0.3511946033175186303183410e-1};
    constexpr const FloatType weights_16[] = {0.1894506104550684962853967, 0.1826034150449235888667636, 0.1691565193950025381893119, 0.1495959888165767320815019, 0.1246289712555338720524763, 0.9515851168249278480992520e-1, 0.6225352393864789286284360e-1, 0.2715245941175409485178166e-1};
    constexpr const FloatType weights_18[] = {0.1691423829631435918406565, 0.1642764837458327229860538, 0.1546846751262652449254180, 0.1406429146706506512047311, 0.1225552067114784601845192, 0.1009420441062871655628144, 0.7642573025488905652912984e-1, 0.4971454889496979645333512e-1, 0.2161601352648331031334248e-1};
    constexpr const FloatType weights_20[] = {0.1527533871307258506980843, 0.1491729864726037467878288, 0.1420961093183820513292985, 0.1316886384491766268984948, 0.1181945319615184173123774, 0.1019301198172404350367504, 0.8327674157670474872475850e-1, 0.6267204833410906356950596e-1, 0.4060142980038694133103928e-1, 0.1761400713915211831186249e-1};
    constexpr const FloatType weights_22[] = {0.1392518728556319933754102, 0.1365414983460151713525738, 0.1311735047870623707329649, 0.1232523768105124242855609, 0.1129322960805392183934005, 0.1004141444428809649320786, 0.8594160621706772741444398e-1, 0.6979646842452048809496104e-1, 0.5229333515268328594031142e-1, 0.3377490158481415479330258e-1, 0.1462799529827220068498987e-1};
    constexpr const FloatType weights_24[] = {0.1279381953467521569740562, 0.1258374563468282961213754, 0.1216704729278033912044631, 0.1155056680537256013533445, 0.1074442701159656347825772, 0.9761865210411388826988072e-1, 0.8619016153195327591718514e-1, 0.7334648141108030573403386e-1, 0.5929858491543678074636724e-1, 0.4427743881741980616860272e-1, 0.2853138862893366318130802e-1, 0.1234122979998719954680507e-1};
    constexpr const FloatType weights_26[] = {0.1183214152792622765163711, 0.1166604434852965820446624, 0.1133618165463196665494407, 0.1084718405285765906565795, 0.1020591610944254232384142, 0.9421380035591414846366474e-1, 0.8504589431348523921044770e-1, 0.7468414976565974588707538e-1, 0.6327404632957483553945402e-1, 0.5097582529714781199831990e-1, 0.3796238329436276395030342e-1, 0.2441785109263190878961718e-1, 0.1055137261734300715565387e-1};
    constexpr const FloatType weights_28[] = {0.1100470130164751962823763, 0.1087111922582941352535716, 0.1060557659228464179104165, 0.1021129675780607698142166, 0.9693065799792991585048880e-1, 0.9057174439303284094218612e-1, 0.8311341722890121839039666e-1, 0.7464621423456877902393178e-1, 0.6527292396699959579339794e-1, 0.5510734567571674543148330e-1, 0.4427293475900422783958756e-1, 0.3290142778230437997763004e-1, 0.2113211259277125975149896e-1, 0.9124282593094517738816778e-2};
    constexpr const FloatType weights_30[] = {0.1028526528935588403412856, 0.1017623897484055045964290, 0.9959342058679526706278018e-1, 0.9636873717464425963946864e-1, 0.9212252223778612871763266e-1, 0.8689978720108297980238752e-1, 0.8075589522942021535469516e-1, 0.7375597473770520626824384e-1, 0.6597422988218049512812820e-1, 0.5749315621761906648172152e-1, 0.4840267283059405290293858e-1, 0.3879919256962704959680230e-1, 0.2878470788332336934971862e-1, 0.1846646831109095914230276e-1, 0.7968192496166605615469690e-2};
    constexpr const FloatType weights_32[] = {0.9654008851472780056676488e-1, 0.9563872007927485941908208e-1, 0.9384439908080456563918026e-1, 0.9117387869576388471286854e-1, 0.8765209300440381114277140e-1, 0.8331192422694675522219922e-1, 0.7819389578707030647174106e-1, 0.7234579410884850622539954e-1, 0.6582222277636184683765034e-1, 0.5868409347853554714528360e-1, 0.5099805926237617619616316e-1, 0.4283589802222668065687810e-1, 0.3427386291302143310268716e-1, 0.2539206530926205945575196e-1, 0.1627439473090567060516896e-1, 0.7018610009470096600404748e-2};
    constexpr const FloatType weights_34[] = {0.9095674033025987361533764e-1, 0.9020304437064072957394216e-1, 0.8870189783569386928707642e-1, 0.8646573974703574978424688e-1, 0.8351309969984565518702044e-1, 0.7986844433977184473881888e-1, 0.7556197466003193127083398e-1, 0.7062937581425572499903896e-1, 0.6511152155407641137854442e-1, 0.5905413582752449319396124e-1, 0.5250741457267810616824590e-1, 0.4552561152335327245382266e-1, 0.3816659379638751632176606e-1, 0.3049138063844613180944194e-1, 0.2256372198549497008409476e-1, 0.1445016274859503541520101e-1, 0.6229140555908684718603220e-2};
    constexpr const FloatType weights_36[] = {0.8598327567039474749008516e-1, 0.8534668573933862749185052e-1, 0.8407821897966193493345756e-1, 0.8218726670433970951722338e-1, 0.7968782891207160190872470e-1, 0.7659841064587067452875784e-1, 0.7294188500565306135387342e-1, 0.6874532383573644261368974e-1, 0.6403979735501548955638454e-1, 0.5886014424532481730967550e-1, 0.5324471397775991909202590e-1, 0.4723508349026597841661708e-1, 0.4087575092364489547411412e-1, 0.3421381077030722992124474e-1, 0.2729862149856877909441690e-1, 0.2018151529773547153209770e-1, 0.1291594728406557440450307e-1, 0.5565719664245045361251818e-2};
    constexpr const FloatType weights_38[] = {0.8152502928038578669921876e-1, 0.8098249377059710062326952e-1, 0.7990103324352782158602774e-1, 0.7828784465821094807537540e-1, 0.7615366354844639606599344e-1, 0.7351269258474345714520658e-1, 0.7038250706689895473928292e-1, 0.6678393797914041193504612e-1, 0.6274093339213305405296984e-1, 0.5828039914699720602230556e-1, 0.5343201991033231997375704e-1, 0.4822806186075868337435238e-1, 0.4270315850467443423587832e-1, 0.3689408159402473816493978e-1, 0.3083950054517505465873166e-1, 0.2457973973823237589520214e-1, 0.1815657770961323689887502e-1, 0.1161344471646867417766868e-1, 0.5002880749639345675901886e-2};
    constexpr const FloatType weights_40[] = {0.7750594797842481126372404e-1, 0.7703981816424796558830758e-1, 0.7611036190062624237155810e-1, 0.7472316905796826420018930e-1, 0.7288658239580405906051074e-1, 0.7061164739128677969548346e-1, 0.6791204581523390382569024e-1, 0.6480401345660103807455446e-1, 0.6130624249292893916653822e-1, 0.5743976909939155136661768e-1, 0.5322784698393682435499678e-1, 0.4869580763507223206143380e-1, 0.4387090818567327199167442e-1, 0.3878216797447201763997196e-1, 0.3346019528254784739267780e-1, 0.2793700698002340109848970e-1, 0.2224584919416695726150432e-1, 0.1642105838190788871286396e-1, 0.1049828453115281361474434e-1, 0.4521277098533191258471490e-2};
    constexpr const FloatType weights_42[] = {0.7386423423217287999638556e-1, 0.7346081345346752826402828e-1, 0.7265617524380410488790570e-1, 0.7145471426517098292181042e-1, 0.6986299249259415976615480e-1, 0.6788970337652194485536350e-1, 0.6554562436490897892700504e-1, 0.6284355804500257640931846e-1, 0.5979826222758665431283142e-1, 0.5642636935801838164642686e-1, 0.5274629569917407034394234e-1, 0.4877814079280324502744954e-1, 0.4454357777196587787431674e-1, 0.4006573518069226176059618e-1, 0.3536907109759211083266214e-1, 0.3047924069960346836290502e-1, 0.2542295952611304788674188e-1, 0.2022786956905264475705664e-1, 0.1492244369735749414467869e-1, 0.9536220301748502411822340e-2, 0.4105998604649084610599928e-2};
    constexpr const FloatType weights_44[] = {0.7054915778935406881133824e-1, 0.7019768547355821258714200e-1, 0.6949649186157257803708402e-1, 0.6844907026936666098545864e-1, 0.6706063890629365239570506e-1, 0.6533811487918143498424096e-1, 0.6329007973320385495013890e-1, 0.6092673670156196803855800e-1, 0.5825985987759549533421064e-1, 0.5530273556372805254874660e-1, 0.5207009609170446188123162e-1, 0.4857804644835203752763920e-1, 0.4484398408197003144624282e-1, 0.4088651231034621890844686e-1, 0.3672534781380887364290888e-1, 0.3238122281206982088084682e-1, 0.2787578282128101008111450e-1, 0.2323148190201921062895910e-1, 0.1847148173681474917204335e-1, 0.1361958675557998552020491e-1, 0.8700481367524844122565470e-2, 0.3745404803112777515171456e-2};
    constexpr const FloatType weights_46[] = {0.6751868584903645882021418e-1, 0.6721061360067817586237416e-1, 0.6659587476845488737576196e-1, 0.6567727426778120737875756e-1, 0.6445900346713906958827948e-1, 0.6294662106439450817895206e-1, 0.6114702772465048101535670e-1, 0.5906843459554631480755080e-1, 0.5672032584399123581687444e-1, 0.5411341538585675449163752e-1, 0.5125959800714302133536554e-1, 0.4817189510171220053046892e-1, 0.4486439527731812676709458e-1, 0.4135219010967872970421980e-1, 0.3765130535738607132766076e-1, 0.3377862799910689652060416e-1, 0.2975182955220275579905234e-1, 0.2558928639713001063470016e-1, 0.2130999875413650105447862e-1, 0.1693351400783623804623151e-1, 0.1247988377098868420673525e-1, 0.7969898229724622451610710e-2, 0.3430300868107048286016700e-2};
    constexpr const FloatType weights_48[] = {0.6473769681268392250302496e-1, 0.6446616443595008220650418e-1, 0.6392423858464818662390622e-1, 0.6311419228625402565712596e-1, 0.6203942315989266390419786e-1, 0.6070443916589388005296916e-1, 0.5911483969839563574647484e-1, 0.5727729210040321570515042e-1, 0.5519950369998416286820356e-1, 0.5289018948519366709550490e-1, 0.5035903555385447495780746e-1, 0.4761665849249047482590674e-1, 0.4467456085669428041944838e-1, 0.4154508294346474921405856e-1, 0.3824135106583070631721688e-1, 0.3477722256477043889254814e-1, 0.3116722783279808890206628e-1, 0.2742650970835694820007336e-1, 0.2357076083932437914051962e-1, 0.1961616045735552781446139e-1, 0.1557931572294384872817736e-1, 0.1147723457923453948959265e-1, 0.7327553901276262102386656e-2, 0.3153346052305838632678320e-2};
    constexpr const FloatType weights_50[] = {0.6217661665534726232103316e-1, 0.6193606742068324338408750e-1, 0.6145589959031666375640678e-1, 0.6073797084177021603175000e-1, 0.5978505870426545750957640e-1, 0.5860084981322244583512250e-1, 0.5718992564772838372302946e-1, 0.5555774480621251762356746e-1, 0.5371062188899624652345868e-1, 0.5165570306958113848990528e-1, 0.4940093844946631492124360e-1, 0.4695505130394843296563322e-1, 0.4432750433880327549202254e-1, 0.4152846309014769742241230e-1, 0.3856875661258767524477018e-1, 0.3545983561514615416073452e-1, 0.3221372822357801664816538e-1, 0.2884299358053519802990658e-1, 0.2536067357001239044019428e-1, 0.2178024317012479298159128e-1, 0.1811556071348939035125903e-1, 0.1438082276148557441937880e-1, 0.1059054838365096926356876e-1, 0.6759799195745401502778824e-2, 0.2908622553155140958394976e-2};
    constexpr const FloatType weights_52[] = {0.5981036574529186024778538e-1, 0.5959626017124815825831088e-1, 0.5916881546604297036933200e-1, 0.5852956177181386855029062e-1, 0.5768078745252682765393200e-1, 0.5662553090236859719080832e-1, 0.5536756966930265254904124e-1, 0.5391140693275726475083694e-1, 0.5226225538390699303439404e-1, 0.5042601856634237721821144e-1, 0.4840926974407489685396032e-1, 0.4621922837278479350764582e-1, 0.4386373425900040799512978e-1, 0.4135121950056027167904044e-1, 0.3869067831042397898510146e-1, 0.3589163483509723294194276e-1, 0.3296410908971879791501014e-1, 0.2991858114714394664128188e-1, 0.2676595374650401344949324e-1, 0.2351751355398446159032286e-1, 0.2018489150798079220298930e-1, 0.1678002339630073567792252e-1, 0.1331511498234096065660116e-1, 0.9802634579462752061952706e-2, 0.6255523962973276899717754e-2, 0.2691316950047111118946698e-2};
    constexpr const FloatType weights_54[] = {0.5761753670714702467237616e-1, 0.5742613705411211485929010e-1, 0.5704397355879459856782852e-1, 0.5647231573062596503104434e-1, 0.5571306256058998768336982e-1, 0.5476873621305798630622270e-1, 0.5364247364755361127210060e-1, 0.5233801619829874466558872e-1, 0.5085969714618814431970910e-1, 0.4921242732452888606879048e-1, 0.4740167880644499105857626e-1, 0.4543346672827671397485208e-1, 0.4331432930959701544192564e-1, 0.4105130613664497422171834e-1, 0.3865191478210251683685736e-1, 0.3612412584038355258288694e-1, 0.3347633646437264571604038e-1, 0.3071734249787067605400450e-1, 0.2785630931059587028700164e-1, 0.2490274146720877305005456e-1, 0.2186645142285308594551102e-1, 0.1875752762146937791200757e-1, 0.1558630303592413170296832e-1, 0.1236332812884764416646861e-1, 0.9099369455509396948032734e-2, 0.5805611015239984878826112e-2, 0.2497481835761585775945054e-2};
    constexpr const FloatType weights_56[] = {0.5557974630651439584627342e-1, 0.5540795250324512321779340e-1, 0.5506489590176242579630464e-1, 0.5455163687088942106175058e-1, 0.5386976186571448570895448e-1, 0.5302137852401076396799152e-1, 0.5200910915174139984305222e-1, 0.5083608261779848056012412e-1, 0.4950592468304757891996610e-1, 0.4802274679360025812073550e-1, 0.4639113337300189676219012e-1, 0.4461612765269228321341510e-1, 0.4270321608466708651103858e-1, 0.4065831138474451788012514e-1, 0.3848773425924766248682568e-1, 0.3619819387231518603588452e-1, 0.3379676711561176129542654e-1, 0.3129087674731044786783572e-1, 0.2868826847382274172988602e-1, 0.2599698705839195219181960e-1, 0.2322535156256531693725830e-1, 0.2038192988240257263480560e-1, 0.1747551291140094650495930e-1, 0.1451508927802147180777130e-1, 0.1150982434038338217377419e-1, 0.8469063163307887661628584e-2, 0.5402522246015337761313780e-2, 0.2323855375773215501098716e-2};
    constexpr const FloatType weights_58[] = {0.5368111986333484886390600e-1, 0.5352634330405825210061082e-1, 0.5321723644657901410348096e-1, 0.5275469052637083342964580e-1, 0.5214003918366981897126058e-1, 0.5137505461828572547451486e-1, 0.5046194247995312529765992e-1, 0.4940333550896239286651076e-1, 0.4820228594541774840657052e-1, 0.4686225672902634691841818e-1, 0.4538711151481980250398048e-1, 0.4378110353364025103902560e-1, 0.4204886332958212599457020e-1, 0.4019538540986779688807676e-1, 0.3822601384585843322945902e-1, 0.3614642686708727054078062e-1, 0.3396262049341601079772722e-1, 0.3168089125380932732029244e-1, 0.2930781804416049071839382e-1, 0.2685024318198186847590714e-1, 0.2431525272496395254025850e-1, 0.2171015614014623576691612e-1, 0.1904246546189340865578709e-1, 0.1631987423497096505212063e-1, 0.1355023711298881214517933e-1, 0.1074155353287877411685532e-1, 0.7901973849998674754018608e-2, 0.5039981612650243085015810e-2, 0.2167723249627449943047768e-2};
    constexpr const FloatType weights_60[] = {0.5190787763122063973286496e-1, 0.5176794317491018754380368e-1, 0.5148845150098093399504444e-1, 0.5107015606985562740454910e-1, 0.5051418453250937459823872e-1, 0.4982203569055018101115930e-1, 0.4899557545575683538947578e-1, 0.4803703181997118096366674e-1, 0.4694898884891220484701330e-1, 0.4573437971611448664719662e-1, 0.4439647879578711332778398e-1, 0.4293889283593564195423128e-1, 0.4136555123558475561316394e-1, 0.3968069545238079947012286e-1, 0.3788886756924344403094056e-1, 0.3599489805108450306657888e-1, 0.3400389272494642283491466e-1, 0.3192121901929632894945890e-1, 0.2975249150078894524083642e-1, 0.2750355674992479163522324e-1, 0.2518047762152124837957096e-1, 0.2278951694399781986378308e-1, 0.2033712072945728677503268e-1, 0.1782990101420772026039605e-1, 0.1527461859678479930672510e-1, 0.1267816647681596013149540e-1, 0.1004755718228798435788578e-1, 0.7389931163345455531517530e-2, 0.4712729926953568640893942e-2, 0.2026811968873758496433874e-2};
    constexpr const FloatType weights_62[] = {0.5024800037525628168840300e-1, 0.5012106956904328807480410e-1, 0.4986752859495239424476130e-1, 0.4948801791969929252786578e-1, 0.4898349622051783710485112e-1, 0.4835523796347767283480314e-1, 0.4760483018410123227045008e-1, 0.4673416847841552480220700e-1, 0.4574545221457018077723242e-1, 0.4464117897712441429364478e-1, 0.4342413825804741958006920e-1, 0.4209740441038509664302268e-1, 0.4066432888241744096828524e-1, 0.3912853175196308412331100e-1, 0.3749389258228002998561838e-1, 0.3576454062276814128558760e-1, 0.3394484437941054509111762e-1, 0.3203940058162467810633926e-1, 0.3005302257398987007700934e-1, 0.2799072816331463754123820e-1, 0.2585772695402469802709536e-1, 0.2365940720868279257451652e-1, 0.2140132227766996884117906e-1, 0.1908917665857319873250324e-1, 0.1672881179017731628855027e-1, 0.1432619182380651776740340e-1, 0.1188739011701050194481938e-1, 0.9418579428420387637936636e-2, 0.6926041901830960871704530e-2, 0.4416333456930904813271960e-2, 0.1899205679513690480402948e-2};
    constexpr const FloatType weights_64[] = {0.4869095700913972038336538e-1, 0.4857546744150342693479908e-1, 0.4834476223480295716976954e-1, 0.4799938859645830772812614e-1, 0.4754016571483030866228214e-1, 0.4696818281621001732532634e-1, 0.4628479658131441729595326e-1, 0.4549162792741814447977098e-1, 0.4459055816375656306013478e-1, 0.4358372452932345337682780e-1, 0.4247351512365358900733972e-1, 0.4126256324262352861015628e-1, 0.3995374113272034138665686e-1, 0.3855015317861562912896262e-1, 0.3705512854024004604041492e-1, 0.3547221325688238381069330e-1, 0.3380516183714160939156536e-1, 0.3205792835485155358546770e-1, 0.3023465707240247886797386e-1, 0.2833967261425948322751098e-1, 0.2637746971505465867169136e-1, 0.2435270256871087333817770e-1, 0.2227017380838325415929788e-1, 0.2013482315353020937234076e-1, 0.1795171577569734308504602e-1, 0.1572603047602471932196614e-1, 0.1346304789671864259806029e-1, 0.1116813946013112881859029e-1, 0.8846759826363947723030856e-2, 0.6504457968978362856118112e-2, 0.4147033260562467635287472e-2, 0.1783280721696432947292054e-2};
    constexpr const FloatType weights_66[] = {0.4722748126299855484563332e-1, 0.4712209828764473218544518e-1, 0.4691156748762082774625404e-1, 0.4659635863958410362582412e-1, 0.4617717509791597547166640e-1, 0.4565495222527305612043888e-1, 0.4503085530544150021519278e-1, 0.4430627694315316190460328e-1, 0.4348283395666747864757528e-1, 0.4256236377005571631890662e-1, 0.4154692031324188131773448e-1, 0.4043876943895497912586836e-1, 0.3924038386682833018781280e-1, 0.3795443766594162094913028e-1, 0.3658380028813909441368980e-1, 0.3513153016547255590064132e-1, 0.3360086788611223267034862e-1, 0.3199522896404688727128174e-1, 0.3031819621886851919364104e-1, 0.2857351178293187118282268e-1, 0.2676506875425000190879332e-1, 0.2489690251475737263773110e-1, 0.2297318173532665591809836e-1, 0.2099819909186462577733052e-1, 0.1897636172277132593486659e-1, 0.1691218147224521718035102e-1, 0.1481026500273396017364296e-1, 0.1267530398126168187644599e-1, 0.1051206598770575465737803e-1, 0.8325388765990901416725080e-2, 0.6120192018447936365568516e-2, 0.3901625641744248259228942e-2, 0.1677653744007238599334225e-2};
    constexpr const FloatType weights_68[] = {0.4584938738725097468656398e-1,0.4575296541606795051900614e-1,0.4556032425064828598070770e-1,0.4527186901844377786941174e-1,0.4488820634542666782635216e-1,0.4441014308035275590934876e-1,0.4383868459795605201060492e-1,0.4317503268464422322584344e-1,0.4242058301114249930061428e-1,0.4157692219740291648457550e-1,0.4064582447595407614088174e-1,0.3962924796071230802540652e-1,0.3852933052910671449325372e-1,0.3734838532618666771607896e-1,0.3608889590017987071497568e-1,0.3475351097975151316679320e-1,0.3334503890398068790314300e-1,0.3186644171682106493934736e-1,0.3032082893855398034157906e-1,0.2871145102748499071080394e-1,0.2704169254590396155797848e-1,0.2531506504517639832390244e-1,0.2353519968587633336129308e-1,0.2170583961037807980146532e-1,0.1983083208795549829102926e-1,0.1791412045792315248940600e-1,0.1595973590961380007213420e-1,0.1397178917445765581596455e-1,0.1195446231976944210322336e-1,0.9912001251585937209131520e-2,0.7848711393177167415052160e-2,0.5768969918729952021468320e-2,0.3677366595011730633570254e-2,0.1581140256372912939103728e-2};
    constexpr const FloatType weights_70[] = {0.4454941715975466720216750e-1, 0.4446096841724637082355728e-1, 0.4428424653905540677579966e-1, 0.4401960239018345875735580e-1, 0.4366756139720144025254848e-1, 0.4322882250506869978939520e-1, 0.4270425678944977776996576e-1, 0.4209490572728440602098398e-1, 0.4140197912904520863822652e-1, 0.4062685273678961635122600e-1, 0.3977106549277656747784952e-1, 0.3883631648407340397900292e-1, 0.3782446156922281719727230e-1, 0.3673750969367269534804046e-1, 0.3557761890129238053276980e-1, 0.3434709204990653756854510e-1, 0.3304837223937242047087430e-1, 0.3168403796130848173465310e-1, 0.3025679798015423781653688e-1, 0.2876948595580828066131070e-1, 0.2722505481866441715910742e-1, 0.2562657090846848279898494e-1, 0.2397720788910029227868640e-1, 0.2228024045225659583389064e-1, 0.2053903782432645338449270e-1, 0.1875705709313342341545081e-1, 0.1693783637630293253183738e-1, 0.1508498786544312768229492e-1, 0.1320219081467674762507440e-1, 0.1129318464993153764963015e-1, 0.9361762769699026811498692e-2, 0.7411769363190210362109460e-2, 0.5447111874217218312821680e-2, 0.3471894893078143254999524e-2, 0.1492721288844515731042666e-2};

    constexpr const FloatType weights_1[] = {2.0000000000000000000000000};
    constexpr const FloatType weights_3[] = {0.8888888888888888888888889, 0.5555555555555555555555555};
    constexpr const FloatType weights_5[] = {0.5688888888888888888888888, 0.4786286704993664680412916, 0.2369268850561890875142644};
    constexpr const FloatType weights_7[] = {0.4179591836734693877551020, 0.3818300505051189449503698, 0.2797053914892766679014680, 0.1294849661688696932706118};
    constexpr const FloatType weights_9[] = {0.3302393550012597631645250, 0.3123470770400028400686304, 0.2606106964029354623187428, 0.1806481606948574040584721, 0.8127438836157441197189206e-1};
    constexpr const FloatType weights_11[] = {0.2729250867779006307144835, 0.2628045445102466621806890, 0.2331937645919904799185238, 0.1862902109277342514260979, 0.1255803694649046246346947, 0.5566856711617366648275374e-1};
    constexpr const FloatType weights_13[] = {0.2325515532308739101945895, 0.2262831802628972384120902, 0.2078160475368885023125234, 0.1781459807619457382800468, 0.1388735102197872384636019, 0.9212149983772844791442126e-1, 0.4048400476531587952001996e-1};
    constexpr const FloatType weights_15[] = {0.2025782419255612728806201, 0.1984314853271115764561182, 0.1861610000155622110268006, 0.1662692058169939335532006, 0.1395706779261543144478051, 0.1071592204671719350118693, 0.7036604748810812470926662e-1, 0.3075324199611726835462762e-1};
    constexpr const FloatType weights_17[] = {0.1794464703562065254582656, 0.1765627053669926463252710, 0.1680041021564500445099705, 0.1540457610768102880814317, 0.1351363684685254732863199, 0.1118838471934039710947887, 0.8503614831717918088353538e-1, 0.5545952937398720112944102e-1, 0.2414830286854793196010920e-1};
    constexpr const FloatType weights_19[] = {0.1610544498487836959791636, 0.1589688433939543476499565, 0.1527660420658596667788553, 0.1426067021736066117757460, 0.1287539625393362276755159, 0.1115666455473339947160242, 0.9149002162244999946446222e-1, 0.6904454273764122658070790e-1, 0.4481422676569960033283728e-1, 0.1946178822972647703631351e-1};
    constexpr const FloatType weights_21[] = {0.1460811336496904271919851, 0.1445244039899700590638271, 0.1398873947910731547221335, 0.1322689386333374617810526, 0.1218314160537285341953671, 0.1087972991671483776634747, 0.9344442345603386155329010e-1, 0.7610011362837930201705132e-1, 0.5713442542685720828363528e-1, 0.3695378977085249379995034e-1, 0.1601722825777433332422273e-1};
    constexpr const FloatType weights_23[] = {0.1336545721861061753514571, 0.1324620394046966173716425, 0.1289057221880821499785954, 0.1230490843067295304675784, 0.1149966402224113649416434, 0.1048920914645414100740861, 0.9291576606003514747701876e-1, 0.7928141177671895492289248e-1, 0.6423242140852585212716980e-1, 0.4803767173108466857164124e-1, 0.3098800585697944431069484e-1, 0.1341185948714177208130864e-1};
    constexpr const FloatType weights_25[] = {0.1231760537267154512039028, 0.1222424429903100416889594, 0.1194557635357847722281782, 0.1148582591457116483393255, 0.1085196244742636531160939, 0.1005359490670506442022068, 0.9102826198296364981149704e-1, 0.8014070033500101801323524e-1, 0.6803833381235691720718712e-1, 0.5490469597583519192593686e-1, 0.4093915670130631265562402e-1, 0.2635498661503213726190216e-1, 0.1139379850102628794789998e-1};
    constexpr const FloatType weights_27[] = {0.1142208673789569890450457, 0.1134763461089651486203700, 0.1112524883568451926721632, 0.1075782857885331872121629, 0.1025016378177457986712478, 0.9608872737002850756565252e-1, 0.8842315854375695019432262e-1, 0.7960486777305777126307488e-1, 0.6974882376624559298432254e-1, 0.5898353685983359911030058e-1, 0.4744941252061506270409646e-1, 0.3529705375741971102257772e-1, 0.2268623159618062319603554e-1, 0.9798996051294360261149438e-2};
    constexpr const FloatType weights_29[] = {0.1064793817183142442465111, 0.1058761550973209414065914, 0.1040733100777293739133284, 0.1010912737599149661218204, 0.9696383409440860630190016e-1, 0.9173775713925876334796636e-1, 0.8547225736617252754534480e-1, 0.7823832713576378382814484e-1, 0.7011793325505127856958160e-1, 0.6120309065707913854210970e-1, 0.5159482690249792391259412e-1, 0.4140206251868283610482948e-1, 0.3074049220209362264440778e-1, 0.1973208505612270598385931e-1, 0.8516903878746409654261436e-2};
    constexpr const FloatType weights_31[] = {0.0997205447934264514275338, 0.9922501122667230787487546e-1, 0.9774333538632872509347402e-1, 0.9529024291231951280720412e-1, 0.9189011389364147821536290e-1, 0.8757674060847787612619794e-1, 0.8239299176158926390382334e-1, 0.7639038659877661642635764e-1, 0.6962858323541036616775632e-1, 0.6217478656102842691034334e-1, 0.5410308242491685371166596e-1, 0.4549370752720110290231576e-1, 0.3643227391238546402439264e-1, 0.2700901918497942180060860e-1, 0.1731862079031058246315918e-1, 0.7470831579248775858700554e-2};
    constexpr const FloatType weights_33[] = {0.0937684461602099965673045, 0.9335642606559611616099912e-1, 0.9212398664331684621324104e-1, 0.9008195866063857723974370e-1, 0.8724828761884433760728158e-1, 0.8364787606703870761392808e-1, 0.7931236479488673836390848e-1, 0.7427985484395414934247216e-1, 0.6859457281865671280595482e-1, 0.6230648253031748003162750e-1, 0.5547084663166356128494468e-1, 0.4814774281871169567014706e-1, 0.4040154133166959156340938e-1, 0.3230035863232895328156104e-1, 0.2391554810174948035053310e-1, 0.1532170151293467612794584e-1, 0.6606227847587378058647800e-2};
    constexpr const FloatType weights_35[] = {0.0884867949071042906382073, 0.8814053043027546297073886e-1, 0.8710444699718353424332214e-1, 0.8538665339209912522594402e-1, 0.8300059372885658837992644e-1, 0.7996494224232426293266204e-1, 0.7630345715544205353865872e-1, 0.7204479477256006466546180e-1, 0.6722228526908690396430546e-1, 0.6187367196608018888701398e-1, 0.5604081621237012857832772e-1, 0.4976937040135352980519956e-1, 0.4310842232617021878230592e-1, 0.3611011586346338053271748e-1, 0.2882926010889425404871630e-1, 0.2132297991148358088343844e-1, 0.1365082834836149226640441e-1, 0.5883433420443084975750336e-2};
    constexpr const FloatType weights_37[] = {0.0837683609931389047970173, 0.8347457362586278725225302e-1, 0.8259527223643725089123018e-1, 0.8113662450846503050987774e-1, 0.7910886183752938076721222e-1, 0.7652620757052923788588804e-1, 0.7340677724848817272462668e-1, 0.6977245155570034488508154e-1, 0.6564872287275124948402376e-1, 0.6106451652322598613098804e-1, 0.5605198799827491780853916e-1, 0.5064629765482460160387558e-1, 0.4488536466243716665741054e-1, 0.3880960250193454448896226e-1, 0.3246163984752148106723444e-1, 0.2588603699055893352275954e-1, 0.1912904448908396604350259e-1, 0.1223878010030755652630649e-1, 0.5273057279497939351724544e-2};
    constexpr const FloatType weights_39[] = {0.0795276221394428524174181, 0.7927622256836847101015574e-1, 0.7852361328737117672506330e-1, 0.7727455254468201672851160e-1, 0.7553693732283605770478448e-1, 0.7332175341426861738115402e-1, 0.7064300597060876077011486e-1, 0.6751763096623126536302120e-1, 0.6396538813868238898670650e-1, 0.6000873608859614957494160e-1, 0.5567269034091629990739094e-1, 0.5098466529212940521402098e-1, 0.4597430110891663188417682e-1, 0.4067327684793384393905618e-1, 0.3511511149813133076106530e-1, 0.2933495598390337859215654e-1, 0.2336938483217816459471240e-1, 0.1725622909372491904080491e-1, 0.1103478893916459424267603e-1, 0.4752944691635101370775866e-2};
    constexpr const FloatType weights_41[] = {0.0756955356472983723187799, 0.7547874709271582402724706e-1, 0.7482962317622155189130518e-1, 0.7375188202722346993928094e-1, 0.7225169686102307339634646e-1, 0.7033766062081749748165896e-1, 0.6802073676087676673553342e-1, 0.6531419645352741043616384e-1, 0.6223354258096631647157330e-1, 0.5879642094987194499118590e-1, 0.5502251924257874188014710e-1, 0.5093345429461749478117008e-1, 0.4655264836901434206075674e-1, 0.4190519519590968942934048e-1, 0.3701771670350798843526154e-1, 0.3191821173169928178706676e-1, 0.2663589920711044546754900e-1, 0.2120106336877955307569710e-1, 0.1564493840781858853082666e-1, 0.9999938773905945338496546e-2, 0.4306140358164887684003630e-2};
    constexpr const FloatType weights_43[] = {0.0722157516937989879774623, 0.7202750197142197434530754e-1, 0.7146373425251414129758106e-1, 0.7052738776508502812628636e-1, 0.6922334419365668428229950e-1, 0.6755840222936516919240796e-1, 0.6554124212632279749123378e-1, 0.6318238044939611232562970e-1, 0.6049411524999129451967862e-1, 0.5749046195691051942760910e-1, 0.5418708031888178686337342e-1, 0.5060119278439015652385048e-1, 0.4675149475434658001064704e-1, 0.4265805719798208376380686e-1, 0.3834222219413265757212856e-1, 0.3382649208686029234496834e-1, 0.2913441326149849491594084e-1, 0.2429045661383881590201850e-1, 0.1931990142368390039612543e-1, 0.1424875643157648610854214e-1, 0.9103996637401403318866628e-2, 0.3919490253844127282968528e-2};
    constexpr const FloatType weights_45[] = {0.0690418248292320201107985, 0.6887731697766132288200278e-1, 0.6838457737866967453169206e-1, 0.6756595416360753627091012e-1, 0.6642534844984252808291474e-1, 0.6496819575072343085382664e-1, 0.6320144007381993774996374e-1, 0.6113350083106652250188634e-1, 0.5877423271884173857436156e-1, 0.5613487875978647664392382e-1, 0.5322801673126895194590376e-1, 0.5006749923795202979913194e-1, 0.4666838771837336526776814e-1, 0.4304688070916497115169120e-1, 0.3922023672930244756418756e-1, 0.3520669220160901624770010e-1, 0.3102537493451546716250854e-1, 0.2669621396757766480567536e-1, 0.2223984755057873239395080e-1, 0.1767753525793759061709347e-1, 0.1303110499158278432063191e-1, 0.8323189296218241645734836e-2, 0.3582663155283558931145652e-2};
    constexpr const FloatType weights_47[] = {0.0661351296236554796534403, 0.6599053358881047453357062e-1, 0.6555737776654974025114294e-1, 0.6483755623894572670260402e-1, 0.6383421660571703063129384e-1, 0.6255174622092166264056434e-1, 0.6099575300873964533071060e-1, 0.5917304094233887597615438e-1, 0.5709158029323154022201646e-1, 0.5476047278153022595712512e-1, 0.5218991178005714487221170e-1, 0.4939113774736116960457022e-1, 0.4637638908650591120440168e-1, 0.4315884864847953826830162e-1, 0.3975258612253100378090162e-1, 0.3617249658417495161345948e-1, 0.3243423551518475676761786e-1, 0.2855415070064338650473990e-1, 0.2454921165965881853783378e-1, 0.2043693814766842764203432e-1, 0.1623533314643305967072624e-1, 0.1196284846431232096394232e-1, 0.7638616295848833614105174e-2, 0.3287453842528014883248206e-2};
    constexpr const FloatType weights_49[] = {0.0634632814047905977182534, 0.6333550929649174859083696e-1, 0.6295270746519569947439960e-1, 0.6231641732005726740107682e-1, 0.6142920097919293629682652e-1, 0.6029463095315201730310616e-1, 0.5891727576002726602452756e-1, 0.5730268153018747548516450e-1, 0.5545734967480358869043158e-1, 0.5338871070825896852794302e-1, 0.5110509433014459067462262e-1, 0.4861569588782824027765094e-1, 0.4593053935559585354249958e-1, 0.4306043698125959798834538e-1, 0.4001694576637302136860494e-1, 0.3681232096300068981946734e-1, 0.3345946679162217434248744e-1, 0.2997188462058382535069014e-1, 0.2636361892706601696094518e-1, 0.2264920158744667649877160e-1, 0.1884359585308945844445106e-1, 0.1496214493562465102958377e-1, 0.1102055103159358049750846e-1, 0.7035099590086451473452956e-2, 0.3027278988922905077484090e-2};
    constexpr const FloatType weights_51[] = {0.0609989248412058801597976, 0.6088546484485634388119860e-1, 0.6054550693473779513812526e-1, 0.5998031577750325209006396e-1, 0.5919199392296154378353896e-1, 0.5818347398259214059843780e-1, 0.5695850772025866210007778e-1, 0.5552165209573869301673704e-1, 0.5387825231304556143409938e-1, 0.5203442193669708756413650e-1, 0.4999702015005740977954886e-1, 0.4777362624062310199999514e-1, 0.4537251140765006874816670e-1, 0.4280260799788008665360980e-1, 0.4007347628549645318680892e-1, 0.3719526892326029284290846e-1, 0.3417869320418833623620910e-1, 0.3103497129016000845442504e-1, 0.2777579859416247719599602e-1, 0.2441330057378143427314164e-1, 0.2095998840170321057979252e-1, 0.1742871472340105225950284e-1, 0.1383263400647782229668883e-1, 0.1018519129782172993923731e-1, 0.6500337783252600292109494e-2, 0.2796807171089895575547228e-2};
    constexpr const FloatType weights_53[] = {0.0587187941511643645254869, 0.5861758623272026331807196e-1, 0.5831431136225600755627570e-1, 0.5781001499171319631968304e-1, 0.5710643553626719177338328e-1, 0.5620599838173970980865512e-1, 0.5511180752393359900234954e-1, 0.5382763486873102904208140e-1, 0.5235790722987271819970160e-1, 0.5070769106929271529648556e-1, 0.4888267503269914042044844e-1, 0.4688915034075031402187278e-1, 0.4473398910367281021276570e-1, 0.4242462063452001359228150e-1, 0.3996900584354038212709364e-1, 0.3737560980348291567417214e-1, 0.3465337258353423795838740e-1, 0.3181167845901932306323576e-1, 0.2886032361782373626279970e-1, 0.2580948251075751771396152e-1, 0.2266967305707020839878928e-1, 0.1945172110763689538804750e-1, 0.1616672525668746392806095e-1, 0.1282602614424037917915135e-1, 0.9441202284940344386662890e-2, 0.6024276226948673281242120e-2, 0.2591683720567031811603734e-2};
    constexpr const FloatType weights_55[] = {0.0566029764445604254401057, 0.5651231824977200140065834e-1, 0.5624063407108436802827906e-1, 0.5578879419528408710293598e-1, 0.5515824600250868759665114e-1, 0.5435100932991110207032224e-1, 0.5336967000160547272357054e-1, 0.5221737154563208456439348e-1, 0.5089780512449397922477522e-1, 0.4941519771155173948075862e-1, 0.4777429855120069555003682e-1, 0.4598036394628383810390480e-1, 0.4403914042160658989516800e-1, 0.4195684631771876239520718e-1, 0.3974015187433717960946388e-1, 0.3739615786796554528291572e-1, 0.3493237287358988740726862e-1, 0.3235668922618583168470572e-1, 0.2967735776516104122129630e-1, 0.2690296145639627066711996e-1, 0.2404238800972562200779126e-1, 0.2110480166801645412020978e-1, 0.1809961452072906240796732e-1, 0.1503645833351178821315019e-1, 0.1192516071984861217075236e-1, 0.8775746107058528177390204e-2, 0.5598632266560767354082364e-2, 0.2408323619979788819164582e-2};
    constexpr const FloatType weights_57[] = {0.0546343287565840240628413, 0.5455280360476188648013898e-1, 0.5430847145249864313874678e-1, 0.5390206148329857464280950e-1, 0.5333478658481915842657698e-1, 0.5260833972917743244023134e-1, 0.5172488892051782472062386e-1, 0.5068707072492740865664050e-1, 0.4949798240201967899383808e-1, 0.4816117266168775126885110e-1, 0.4668063107364150378384082e-1, 0.4506077616138115779721374e-1, 0.4330644221621519659643210e-1, 0.4142286487080111036319668e-1, 0.3941566547548011408995280e-1, 0.3729083432441731735473546e-1, 0.3505471278231261750575064e-1, 0.3271397436637156854248994e-1, 0.3027560484269399945849064e-1, 0.2774688140218019232125814e-1, 0.2513535099091812264727322e-1, 0.2244880789077643807968978e-1, 0.1969527069948852038242318e-1, 0.1688295902344154903500062e-1, 0.1402027079075355617024753e-1, 0.1111576373233599014567619e-1, 0.8178160067821232626211086e-2, 0.5216533474718779390504886e-2, 0.2243753872250662909727492e-2};
    constexpr const FloatType weights_59[] = {0.0527980126219904214155123, 0.5272443385912793196130422e-1, 0.5250390264782873905094128e-1, 0.5213703364837539138398724e-1, 0.5162484939089148214644000e-1, 0.5096877742539391685024800e-1, 0.5017064634299690281072034e-1, 0.4923268067936198577969374e-1, 0.4815749471460644038814684e-1, 0.4694808518696201919315986e-1, 0.4560782294050976983186828e-1, 0.4414044353029738069079808e-1, 0.4255003681106763866730838e-1, 0.4084103553868670766020196e-1, 0.3901820301616000950303072e-1, 0.3708661981887092269183778e-1, 0.3505166963640010878371850e-1, 0.3291902427104527775751116e-1, 0.3069462783611168323975056e-1, 0.2838468020053479790515332e-1, 0.2599561973129850018665014e-1, 0.2353410539371336342527500e-1, 0.2100699828843718735046168e-1, 0.1842134275361002936061624e-1, 0.1578434731308146614732024e-1, 0.1310336630634519101831859e-1, 0.1038588550099586219379846e-1, 0.7639529453487575142699186e-2, 0.4872239168265284768580414e-2, 0.2095492284541223402697724e-2};
    constexpr const FloatType weights_61[] = {0.0510811194407862179779210, 0.5101448703869726354373512e-1, 0.5081476366881834320770052e-1, 0.5048247038679740464814450e-1, 0.5001847410817825342505160e-1, 0.4942398534673558993996884e-1, 0.4870055505641152608753004e-1, 0.4785007058509560716183348e-1, 0.4687475075080906597642932e-1, 0.4577714005314595937133982e-1, 0.4456010203508348827154136e-1, 0.4322681181249609790104358e-1, 0.4178074779088849206667564e-1, 0.4022568259099824736764020e-1, 0.3856567320700817274615216e-1, 0.3680505042315481738432126e-1, 0.3494840751653335109085198e-1, 0.3300058827590741063272390e-1, 0.3096667436839739482469792e-1, 0.2885197208818340150434184e-1, 0.2666199852415088966281066e-1, 0.2440246718754420291534050e-1, 0.2207927314831904400247522e-1, 0.1969847774610118133051782e-1, 0.1726629298761374359443389e-1, 0.1478906588493791454617878e-1, 0.1227326350781210462927897e-1, 0.9725461830356133736135366e-2, 0.7152354991749089585834616e-2, 0.4560924006012417184541648e-2, 0.1961453361670282671779431e-2};
    constexpr const FloatType weights_63[] = {0.0494723666239310208886693, 0.4941183303991817896703964e-1, 0.4923038042374756078504314e-1, 0.4892845282051198994470936e-1, 0.4850678909788384786409014e-1, 0.4796642113799513141105276e-1, 0.4730867131226891908060508e-1, 0.4653514924538369651039536e-1, 0.4564774787629260868588592e-1, 0.4464863882594139537033256e-1, 0.4354026708302759079896428e-1, 0.4232534502081582298250554e-1, 0.4100684575966639863511004e-1, 0.3958799589154409398480778e-1, 0.3807226758434955676363856e-1, 0.3646337008545728963045232e-1, 0.3476524064535587769718026e-1, 0.3298203488377934176568344e-1, 0.3111811662221981750821608e-1, 0.2917804720828052694555162e-1, 0.2716657435909793322519012e-1, 0.2508862055334498661862972e-1, 0.2294927100488993314894282e-1, 0.2075376125803909077534152e-1, 0.1850746446016127040926083e-1, 0.1621587841033833888228333e-1, 0.1388461261611561082486681e-1, 0.1151937607688004175075116e-1, 0.9125968676326656354058462e-2, 0.6710291765960136251908410e-2, 0.4278508346863761866081200e-2, 0.1839874595577084117085868e-2};
    constexpr const FloatType weights_65[] = {0.0479618493944666181207076, 0.4790669250049586203134730e-1, 0.4774134868124062155903898e-1, 0.4746619823288550315264446e-1, 0.4708187401045452224600686e-1, 0.4658925997223349830225508e-1, 0.4598948914665169696389334e-1, 0.4528394102630023065712822e-1, 0.4447423839508297442732352e-1, 0.4356224359580048653228480e-1, 0.4255005424675580271921714e-1, 0.4143999841724029302268646e-1, 0.4023462927300553381544642e-1, 0.3893671920405119761667398e-1, 0.3754925344825770980977246e-1, 0.3607542322556527393216642e-1, 0.3451861839854905862522142e-1, 0.3288241967636857498404946e-1, 0.3117059038018914246443218e-1, 0.2938706778931066806264472e-1, 0.2753595408845034394249940e-1, 0.2562150693803775821408458e-1, 0.2364812969128723669878144e-1, 0.2162036128493406284165378e-1, 0.1954286583675006282683714e-1, 0.1742042199767024849536596e-1, 0.1525791214644831034926464e-1, 0.1306031163999484633616732e-1, 0.1083267878959796862151440e-1, 0.8580148266881459893636434e-2, 0.6307942578971754550189764e-2, 0.4021524172003736347075858e-2, 0.1729258251300250898337759e-2};
    constexpr const FloatType weights_67[] = {0.0465408367035635082499002, 0.4649043816026462820831466e-1, 0.4633935168241562110844706e-1, 0.4608790448976157619721740e-1, 0.4573664116106369093689412e-1, 0.4528632245466953156805004e-1, 0.4473792366088982547214182e-1, 0.4409263248975101830783160e-1, 0.4335184649869951735915584e-1, 0.4251717006583049147154770e-1, 0.4159041091519924309854838e-1, 0.4057357620174452522725164e-1, 0.3946886816430888264288692e-1, 0.3827867935617948064763712e-1, 0.3700558746349258202313488e-1, 0.3565234972274500666133270e-1, 0.3422189694953664673983902e-1, 0.3271732719153120542712204e-1, 0.3114189901947282393742616e-1, 0.2949902447094566969584718e-1, 0.2779226166243676998720012e-1, 0.2602530708621323880370460e-1, 0.2420198760967316472069180e-1, 0.2232625219645207692279754e-1, 0.2040216337134354044925720e-1, 0.1843388845680457387216616e-1, 0.1642569062253087920472674e-1, 0.1438191982720055093097663e-1, 0.1230700384928815052195302e-1, 0.1020544003410244098666155e-1, 0.8081790299023136215346300e-2, 0.5940693177582235216514606e-2, 0.3787008301825508445960626e-2, 0.1628325035240012866460003e-2};
    constexpr const FloatType weights_69[] = {0.0452016023770799542422892, 0.4515543023614546051651704e-1, 0.4501700814039980219871620e-1, 0.4478661887831255754213528e-1, 0.4446473312204713809623108e-1, 0.4405200846590928438098588e-1, 0.4354928808292674103357578e-1, 0.4295759900230521387841984e-1, 0.4227815001128051285158270e-1, 0.4151232918565450208287406e-1, 0.4066170105406160053752604e-1, 0.3972800340176164120645862e-1, 0.3871314372049251393273936e-1, 0.3761919531164090650815840e-1, 0.3644839305070051405664348e-1, 0.3520312882168348614775456e-1, 0.3388594663083228949780964e-1, 0.3249953740964611124473418e-1, 0.3104673351789053903268552e-1, 0.2953050295790671177981110e-1, 0.2795394331218770599086132e-1, 0.2632027541686948379176090e-1, 0.2463283678454245536433616e-1, 0.2289507479074078565552120e-1, 0.2111053963987189462789068e-1, 0.1928287712884940278924393e-1, 0.1741582123196982913207401e-1, 0.1551318654340616473976910e-1, 0.1357886064907567099981112e-1, 0.1161679661067196554873961e-1, 0.9631006150415575588660562e-2, 0.7625555931201510611459992e-2, 0.5604579927870594828535346e-2, 0.3572416739397372609702552e-2, 0.1535976952792084075135094e-2};

    const std::array<const FloatType*, 71> weight_arrs = {
        nullptr, weights_1, weights_2, weights_3, weights_4, weights_5,
        weights_6, weights_7, weights_8, weights_9, weights_10, weights_11,
        weights_12, weights_13, weights_14, weights_15, weights_16, weights_17,
        weights_18, weights_19, weights_20, weights_21, weights_22, weights_23,
        weights_24, weights_25, weights_26, weights_27, weights_28, weights_29,
        weights_30, weights_31, weights_32, weights_33, weights_34, weights_35,
        weights_36, weights_37, weights_38, weights_39, weights_40, weights_41,
        weights_42, weights_43, weights_44, weights_45, weights_46, weights_47,
        weights_48, weights_49, weights_50, weights_51, weights_52, weights_53,
        weights_54, weights_55, weights_56, weights_57, weights_58, weights_59,
        weights_60, weights_61, weights_62, weights_63, weights_64, weights_65,
        weights_66, weights_67, weights_68, weights_69, weights_70
    };

    if (std::ranges::size(weights) == 0) return;
    
    if constexpr (std::same_as<Layout, PackedLayout>)
    {
        const std::size_t num_unique_nodes = std::ranges::size(weights);
        const std::size_t num_nodes = 2*num_unique_nodes - parity;
        const FloatType* weight_arr = weight_arrs[num_nodes];
        for (std::size_t i = 0; i < num_unique_nodes; ++i)
            weights[i] = weight_arr[i];
    }
    else if constexpr (std::same_as<Layout, UnpackedLayout>)
    {
        const std::size_t num_nodes = std::ranges::size(weights);
        const std::size_t m = num_nodes >> 1;
        const std::size_t parity = num_nodes & 1;
        const std::size_t num_unique_nodes = m + parity;
        const FloatType* weight_arr = weight_arrs[num_nodes];
        if (parity)
        {
            weights[m] = weight_arr[0];
            for (std::size_t i = 1; i < num_unique_nodes; ++i)
            {
                weights[m - i] = weight_arr[i];
                weights[m + i] = weight_arr[i];
            }
        }
        else
        {
            for (std::size_t i = 0; i < num_unique_nodes; ++i)
            {
                weights[m - i - 1] = weight_arr[i];
                weights[m + i] = weight_arr[i];
            }
        }
    }
}

// Table lookup of nodes and weights where the Bogaert algorithm is inaccurate.
template <gl_layout Layout, GLNodeStyle node_style_param, std::ranges::random_access_range R>
    requires std::floating_point<
        typename std::remove_reference_t<R>::value_type>
constexpr void gl_nodes_and_weights_table(
    R&& nodes, R&& weights, std::size_t parity) noexcept
{
    gl_nodes_table<Layout, node_style_param>(std::forward<R>(nodes), parity);
    gl_weights_table<Layout>(std::forward<R>(weights), parity);
}

} // namespace detail

/**
    @brief Obtain Gauss-Legendre nodes for a given number of nodes.

    @tparam Layout layout of `nodes`
    @tparam R type of the range for storing the nodes

    @param nodes range for storing the nodes
    @param parity parity of the total number of nodes

    For `num_nodes < 70` the nodes are read from a precomputed table. For greater numbers of
    nodes Bogaert's iteration-free method is used: I. Bogaert, Iteration-free computation of
    Gauss-Legendre quadrature nodes and weights, SIAM J. Sci. Comput., 36 (2014), pp. C1008-C1026).

    The nodes returned are accurate to double macine epsilon.
*/
template <gl_layout Layout, GLNodeStyle node_style_param, std::ranges::random_access_range R>requires std::floating_point<
        typename std::remove_reference_t<R>::value_type>
constexpr void gl_nodes(R&& nodes, std::size_t parity) noexcept
{
    if (nodes.size() == 0) return;
    else if (Layout::total_nodes(nodes.size(), parity) < 70)
        detail::gl_nodes_table<Layout, node_style_param>(std::forward<R>(nodes), parity);
    else
        detail::gl_nodes_bogaert<Layout, node_style_param>(std::forward<R>(nodes), parity);
}

/**
    @brief Obtain Gauss-Legendre weights for a given number of nodes.

    @tparam Layout layout of `weights`
    @tparam R type of the range for storing the weights

    @param weights range for storing the weights
    @param parity parity of the total number of nodes

    For `num_nodes < 70` the nodes are read from a precomputed table. For greater numbers of nodes
    Bogaert's iteration-free method is used: I. Bogaert, Iteration-free computation of
    Gauss-Legendre quadrature nodes and weights, SIAM J. Sci. Comput., 36 (2014), pp. C1008-C1026).

    The weights returned are accurate to double macine epsilon.
*/
template <gl_layout Layout, std::ranges::random_access_range R>
    requires std::floating_point<
        typename std::remove_reference_t<R>::value_type>
constexpr void gl_weights(R&& weights, std::size_t parity) noexcept
{
    if (weights.size() == 0) return;
    else if (Layout::total_nodes(weights.size(), parity) < 70)
        detail::gl_weights_table<Layout>(std::forward<R>(weights), parity);
    else
        detail::gl_weights_bogaert<Layout>(std::forward<R>(weights), parity);
}

/**
    @brief Obtain Gauss-Legendre nodes and weights for a given number of nodes.

    @tparam Layout layout of `nodes` and `weights`
    @tparam R type of the range for storing the nodes and weights

    @param nodes range for storing the nodes
    @param weights range for storing the weights
    @param parity parity of the total number of nodes

    For `num_nodes < 70` the nodes are read from a precomputed table. For greater numbers of
    nodes Bogaert's iteration-free method is used: I. Bogaert, Iteration-free computation of
    Gauss-Legendre quadrature nodes and weights, SIAM J. Sci. Comput., 36 (2014), pp. C1008-C1026).

    The nodes and weights returned are accurate to double macine epsilon.
*/
template <gl_layout Layout, GLNodeStyle node_style_param, std::ranges::random_access_range R>
    requires std::floating_point<
        typename std::remove_reference_t<R>::value_type>
constexpr void gl_nodes_and_weights(
    R&& nodes, R&& weights, std::size_t parity) noexcept
{
    if (nodes.size() == 0) return;
    else if (Layout::total_nodes(nodes.size(), parity) < 70)
        detail::gl_nodes_and_weights_table<Layout, node_style_param>(
                std::forward<R>(nodes), std::forward<R>(weights), parity);
    else
        detail::gl_nodes_and_weights_bogaert<Layout, node_style_param>(
                std::forward<R>(nodes), std::forward<R>(weights), parity);
}

} // namespace gl
} // namespace zest
