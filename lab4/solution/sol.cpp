#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <cstdlib>
#include <iostream>

namespace mpi = boost::mpi;

int main(int argc, char* argv[]) {
  const double K = atof(argv[1]);             // коэффициент температуропроводности
  const double TAU = atof(argv[2]);           // шаг по времени
  const double H = atof(argv[3]);             // шаг по пространству
  const unsigned int STEPS = atoi(argv[4]);   // количество шагов
  const unsigned int POINTS = 1.0 / H + 1;
  assert((POINTS - 1) * H == 1.0);

  const double coef = K * TAU / H / H;

  mpi::environment env{argc, argv};
  mpi::communicator world;

  if (world.rank() == 0) {
    std::cout << "k = " << K << ", tau = " << TAU << ", h = " << H << ", steps = " << STEPS << ", points = " << POINTS << std::endl;
  }

  const size_t points_default_num = POINTS / world.size();

  const size_t first_point = world.rank() * points_default_num + std::min(std::max(0, world.rank() - 1), static_cast<int>(POINTS % world.size()));
  const size_t last_point = first_point + points_default_num + (world.rank() < POINTS % world.size());
  const size_t points_num = last_point - first_point;

  const size_t tag_left = world.rank() % 2;
  const size_t tag_right = tag_left ^ 1;

  std::vector<std::vector<double>> values(2, std::vector<double>(points_num, 1.0));
  if (world.rank() == 0) {
    values[0][0] = 0;
    values[1][0] = 0;
  }
  if (world.rank() + 1 == world.size()) {
    values[0].back() = 0;
    values[1].back() = 0;
  }
  for (size_t step = 1; step <= STEPS; ++step) {
    size_t prev_idx = step % 2;
    size_t cur_idx = prev_idx ^ 1;

    double left_value = 0;
    double right_value = 0;

    std::vector<mpi::request> receive_requests;
    std::vector<mpi::request> send_requests;
    if (world.rank() != 0) {
      send_requests.emplace_back(world.isend(world.rank() - 1, tag_left, values[prev_idx][0]));
      receive_requests.emplace_back(world.irecv(world.rank() - 1, tag_left, left_value));
    }
    if (world.rank() + 1 != world.size()) {
      send_requests.emplace_back(world.isend(world.rank() + 1, tag_right, values[prev_idx].back()));
      receive_requests.emplace_back(world.irecv(world.rank() + 1, tag_right, right_value));
    }

    for (size_t point = 1; point + 1 < points_num; ++point) {
      values[cur_idx][point] = values[prev_idx][point] +
        coef * (values[prev_idx][point - 1] + values[prev_idx][point + 1] - 2 * values[prev_idx][point]);
    }

    mpi::wait_all(receive_requests.begin(), receive_requests.end());

    if (world.rank() != 0) {
      values[cur_idx][0] = values[prev_idx][0] +
        coef * (left_value + values[prev_idx][1] - 2 * values[prev_idx][0]);
    }
    if (world.rank() + 1 != world.size()) {
      values[cur_idx].back() = values[prev_idx].back() +
        coef * (values[prev_idx][points_num - 2] + right_value - 2 * values[prev_idx].back());
    }

    mpi::wait_all(send_requests.begin(), send_requests.end());
  }

  if (world.rank() == 0) {
    std::vector<double> result(POINTS);
    std::vector<int> sizes(world.size());
    for (size_t i = 0; i < world.size(); ++i) {
      sizes[i] = points_default_num + (i < POINTS % world.size());
    }
    mpi::gatherv(world, values[1 - STEPS % 2], result.data(), sizes, 0);
    size_t step_size = 0.1 / H;
    for (size_t idx = 0; idx < POINTS; idx += step_size) {
      printf("x = %.2lf, value = %.4lf\n", idx * 0.1, result[idx]);
    }
  } else {
    mpi::gatherv(world, values[1 - STEPS % 2], 0);
  }
}
