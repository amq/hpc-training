#include <vector>

namespace filters {
// pre-computed using http://dev.theomader.com/gaussian-kernel-calculator/
std::vector<float> gaussian3{
    0.077847f, 0.123317f, 0.077847f,
    0.123317f, 0.195346f, 0.123317f,
    0.077847f, 0.123317f, 0.077847f};

std::vector<float> gaussian5{
    0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f,
    0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f,
    0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f,
    0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f,
    0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f};

std::vector<float> gaussian7{
    0.000036f, 0.000363f, 0.001446f, 0.002291f, 0.001446f, 0.000363f, 0.000036f,
    0.000363f, 0.003676f, 0.014662f, 0.023226f, 0.014662f, 0.003676f, 0.000363f,
    0.001446f, 0.014662f, 0.058488f, 0.092651f, 0.058488f, 0.014662f, 0.001446f,
    0.002291f, 0.023226f, 0.092651f, 0.146768f, 0.092651f, 0.023226f, 0.002291f,
    0.001446f, 0.014662f, 0.058488f, 0.092651f, 0.058488f, 0.014662f, 0.001446f,
    0.000363f, 0.003676f, 0.014662f, 0.023226f, 0.014662f, 0.003676f, 0.000363f,
    0.000036f, 0.000363f, 0.001446f, 0.002291f, 0.001446f, 0.000363f, 0.000036f};

std::vector<float> gaussian9{
    0.000000f, 0.000001f, 0.000014f, 0.000055f, 0.000088f, 0.000055f, 0.000014f, 0.000001f, 0.000000f,
    0.000001f, 0.000036f, 0.000362f, 0.001445f, 0.002289f, 0.001445f, 0.000362f, 0.000036f, 0.000001f,
    0.000014f, 0.000362f, 0.003672f, 0.014648f, 0.023205f, 0.014648f, 0.003672f, 0.000362f, 0.000014f,
    0.000055f, 0.001445f, 0.014648f, 0.058434f, 0.092566f, 0.058434f, 0.014648f, 0.001445f, 0.000055f,
    0.000088f, 0.002289f, 0.023205f, 0.092566f, 0.146634f, 0.092566f, 0.023205f, 0.002289f, 0.000088f,
    0.000055f, 0.001445f, 0.014648f, 0.058434f, 0.092566f, 0.058434f, 0.014648f, 0.001445f, 0.000055f,
    0.000014f, 0.000362f, 0.003672f, 0.014648f, 0.023205f, 0.014648f, 0.003672f, 0.000362f, 0.000014f,
    0.000001f, 0.000036f, 0.000362f, 0.001445f, 0.002289f, 0.001445f, 0.000362f, 0.000036f, 0.000001f,
    0.000000f, 0.000001f, 0.000014f, 0.000055f, 0.000088f, 0.000055f, 0.000014f, 0.000001f, 0.000000f};

std::vector<float> gaussian(int size) {
    switch (size) {
    case 3:
        return gaussian3;
        break;
    case 5:
        return gaussian5;
        break;
    case 7:
        return gaussian7;
        break;
    case 9:
        return gaussian9;
        break;
    default:
        return std::vector<float>();
    }
}
} // namespace filters
