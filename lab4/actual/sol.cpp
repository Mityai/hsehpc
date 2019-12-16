#include <cmath>
#include <cstdio>
#include <cstdlib>

constexpr size_t STEPS = 10000;
const double PI = acos(-1);

template <typename T>
T sqr(T x) {
  return x * x;
}

double calc_value(double T0, const double K, const double T, const double L, const double x) {
  double sum = 0;
  for (size_t m = 0; m < STEPS; ++m) {
    sum += std::exp(-K * sqr(PI) * sqr(2 * m + 1) * T / sqr(L)) / (2 * m + 1) * sin(PI * (2 * m + 1) * x / L);
  }
  return sum * 4 * T0 / PI;
}

int main(int argc, char* argv[]) {
  const double T0 = atof(argv[1]);          // время
  const double K = atof(argv[2]);           // коэффициент температуропроводности
  const double T = atof(argv[3]);           // время
  const double H = atof(argv[4]);           // шаг
  const double L = 1.0;                     // длина  стержня
  const unsigned int POINTS = 1.0 / H + 1;

  for (size_t idx = 0; idx < POINTS; ++idx) {
    double x = H * idx;
    printf("x = %.2lf, value = %.8lf\n", x, calc_value(T0, K, T, L, x));
  }
}
