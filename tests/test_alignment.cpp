#include "alignment.hpp"

namespace
{

} // namespace

int main()
{
    assert(zest::detail::is_power_of_two(0));
    assert(zest::detail::is_power_of_two(1));
    assert(zest::detail::is_power_of_two(2));
    assert(!zest::detail::is_power_of_two(3));
    assert(zest::detail::is_power_of_two(4));
    assert(!zest::detail::is_power_of_two(5));
    assert(!zest::detail::is_power_of_two(6));
    assert(!zest::detail::is_power_of_two(7));
    assert(zest::detail::is_power_of_two(8));
    assert(!zest::detail::is_power_of_two(10));
    assert(!zest::detail::is_power_of_two(15));
    assert(zest::detail::is_power_of_two(16));
    assert(!zest::detail::is_power_of_two(19));
    assert(!zest::detail::is_power_of_two(28));
    assert(zest::detail::is_power_of_two(32));
    assert(!zest::detail::is_power_of_two(57));
    assert(zest::detail::is_power_of_two(64));
    assert(!zest::detail::is_power_of_two(65));
    assert(!zest::detail::is_power_of_two(100));
    assert(!zest::detail::is_power_of_two(127));
    assert(zest::detail::is_power_of_two(128));

    assert(zest::detail::next_divisible<1>(0) == 0);
    assert(zest::detail::next_divisible<1>(1) == 1);
    assert(zest::detail::next_divisible<1>(12) == 12);
    assert(zest::detail::next_divisible<1>(345) == 345);
    assert(zest::detail::next_divisible<1>(2345) == 2345);
    assert(zest::detail::next_divisible<1>(34785) == 34785);

    assert(zest::detail::next_divisible<2>(0) == 0);
    assert(zest::detail::next_divisible<2>(1) == 2);
    assert(zest::detail::next_divisible<2>(12) == 12);
    assert(zest::detail::next_divisible<2>(345) == 346);
    assert(zest::detail::next_divisible<2>(2345) == 2346);
    assert(zest::detail::next_divisible<2>(34785) == 34786);

    assert(zest::detail::next_divisible<4>(0) == 0);
    assert(zest::detail::next_divisible<4>(1) == 4);
    assert(zest::detail::next_divisible<4>(12) == 12);
    assert(zest::detail::next_divisible<4>(345) == 348);
    assert(zest::detail::next_divisible<4>(2345) == 2348);
    assert(zest::detail::next_divisible<4>(34785) == 34788);

    assert(zest::detail::next_divisible<8>(0) == 0);
    assert(zest::detail::next_divisible<8>(1) == 8);
    assert(zest::detail::next_divisible<8>(12) == 16);
    assert(zest::detail::next_divisible<8>(345) == 352);
    assert(zest::detail::next_divisible<8>(2345) == 2352);
    assert(zest::detail::next_divisible<8>(34785) == 34792);

    assert(zest::detail::next_divisible<16>(0) == 0);
    assert(zest::detail::next_divisible<16>(1) == 16);
    assert(zest::detail::next_divisible<16>(12) == 16);
    assert(zest::detail::next_divisible<16>(345) == 352);
    assert(zest::detail::next_divisible<16>(2345) == 2352);
    assert(zest::detail::next_divisible<16>(34785) == 34800);

    assert(zest::detail::next_divisible<32>(0) == 0);
    assert(zest::detail::next_divisible<32>(1) == 32);
    assert(zest::detail::next_divisible<32>(12) == 32);
    assert(zest::detail::next_divisible<32>(345) == 352);
    assert(zest::detail::next_divisible<32>(2345) == 2368);
    assert(zest::detail::next_divisible<32>(34785) == 34816);

    assert(zest::detail::next_divisible<64>(0) == 0);
    assert(zest::detail::next_divisible<64>(1) == 64);
    assert(zest::detail::next_divisible<64>(12) == 64);
    assert(zest::detail::next_divisible<64>(345) == 384);
    assert(zest::detail::next_divisible<64>(2345) == 2368);
    assert(zest::detail::next_divisible<64>(34785) == 34816);

    assert(zest::aligned_size<std::uint8_t, zest::NoAlignment>(6) == 6);
    assert(zest::aligned_size<std::uint8_t, zest::SSEAlignment>(6) == 16);
    assert(zest::aligned_size<std::uint8_t, zest::AVXAlignment>(6) == 32);
    assert(zest::aligned_size<std::uint8_t, zest::AVX512Alignment>(6) == 64);
    assert(zest::aligned_size<std::uint8_t, zest::CacheLineAlignment>(6) == 64);

    assert(zest::aligned_size<std::uint16_t, zest::NoAlignment>(6) == 12);
    assert(zest::aligned_size<std::uint16_t, zest::SSEAlignment>(6) == 16);
    assert(zest::aligned_size<std::uint16_t, zest::AVXAlignment>(6) == 32);
    assert(zest::aligned_size<std::uint16_t, zest::AVX512Alignment>(6) == 64);
    assert(zest::aligned_size<std::uint16_t, zest::CacheLineAlignment>(6) == 64);

    assert(zest::aligned_size<std::uint32_t, zest::NoAlignment>(6) == 24);
    assert(zest::aligned_size<std::uint32_t, zest::SSEAlignment>(6) == 32);
    assert(zest::aligned_size<std::uint32_t, zest::AVXAlignment>(6) == 32);
    assert(zest::aligned_size<std::uint32_t, zest::AVX512Alignment>(6) == 64);
    assert(zest::aligned_size<std::uint32_t, zest::CacheLineAlignment>(6) == 64);

    assert(zest::aligned_size<std::uint64_t, zest::NoAlignment>(6) == 48);
    assert(zest::aligned_size<std::uint64_t, zest::SSEAlignment>(6) == 48);
    assert(zest::aligned_size<std::uint64_t, zest::AVXAlignment>(6) == 64);
    assert(zest::aligned_size<std::uint64_t, zest::AVX512Alignment>(6) == 64);
    assert(zest::aligned_size<std::uint64_t, zest::CacheLineAlignment>(6) == 64);
}
