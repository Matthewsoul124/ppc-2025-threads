#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/frolova_e_Sobel_filter/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<int> GenRgbPicture(size_t width, size_t height) {
  std::vector<int> image(width * height * 3);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> rgb(0, 255);

  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      size_t index = (y * width + x) * 3;
      image[index] = rgb(gen);
      image[index + 1] = rgb(gen);
      image[index + 2] = rgb(gen);
    }
  }

  return image;
}

std::vector<frolova_e_sobel_filter_all::RGB> ConvertToRGB(const std::vector<int> &pict) {
  std::vector<frolova_e_sobel_filter_all::RGB> picture;
  size_t pixel_count = pict.size() / 3;

  for (size_t i = 0; i < pixel_count; i++) {
    frolova_e_sobel_filter_all::RGB pixel;
    pixel.R = pict[i * 3];
    pixel.G = pict[(i * 3) + 1];
    pixel.B = pict[(i * 3) + 2];

    picture.push_back(pixel);
  }
  return picture;
}

std::vector<int> ToGrayScaleImgNS(std::vector<frolova_e_sobel_filter_all::RGB> &color_img, size_t width,
                                  size_t height) {
  std::vector<int> gray_scale_image(width * height);
  for (int i = 0; i < static_cast<int>(width * height); i++) {
    gray_scale_image[i] =
        static_cast<int>((0.299 * color_img[i].R) + (0.587 * color_img[i].G) + (0.114 * color_img[i].B));
  }

  return gray_scale_image;
}

std::vector<int> ApplySobelFilter(const std::vector<int> &gray_scale_image, size_t width, size_t height) {
  const std::vector<int> gx = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  const std::vector<int> gy = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

  std::vector<int> result(width * height, 0);

  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      int res_x = 0;
      int res_y = 0;

      for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
          int px = static_cast<int>(x) + kx;
          int py = static_cast<int>(y) + ky;

          int pixel_value = 0;

          if (px >= 0 && px < static_cast<int>(width) && py >= 0 && py < static_cast<int>(height)) {
            pixel_value = gray_scale_image[(py * width) + px];
          }

          size_t kernel_ind = ((ky + 1) * 3) + (kx + 1);
          res_x += pixel_value * gx[kernel_ind];
          res_y += pixel_value * gy[kernel_ind];
        }
      }
      int gradient = static_cast<int>(std::sqrt((res_x * res_x) + (res_y * res_y)));
      result[(y * width) + x] = std::clamp(gradient, 0, 255);
    }
  }

  return result;
}

}  // namespace

TEST(frolova_e_sobel_filter_stl, black_image) {
  boost::mpi::communicator world;
  std::vector<int> value = {5, 5};
  std::vector<int> pict(75, 0);

  std::vector<int> res(25, 0);
  std::vector<int> reference(25, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
    task_data->inputs_count.emplace_back(value.size());

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
    task_data->inputs_count.emplace_back(pict.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(res.size());
  }

  frolova_e_sobel_filter_all::SobelFilterALL test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(reference, res);
  }
}

TEST(frolova_e_sobel_filter_stl, white_image) {
  boost::mpi::communicator world;
  std::vector<int> value = {5, 5};
  std::vector<int> pict(75, 255);

  std::vector<int> res(25, 0);

  std::vector<int> reference = {255, 255, 255, 255, 255, 255, 0,   0,   0,   255, 255, 0,  0,
                                0,   255, 255, 0,   0,   0,   255, 255, 255, 255, 255, 255};

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
    task_data->inputs_count.emplace_back(value.size());

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
    task_data->inputs_count.emplace_back(pict.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(res.size());
  }

  frolova_e_sobel_filter_all::SobelFilterALL test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(reference, res);
  }
}

TEST(frolova_e_sobel_filter_stl, sharp_vertical_edge) {
  boost::mpi::communicator world;
  std::vector<int> value = {5, 5};
  std::vector<int> pict = {0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255,

                           0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255,

                           0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255,

                           0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255,

                           0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255};

  std::vector<int> res(25, 0);

  std::vector<int> reference = {0, 255, 255, 255, 255, 0, 255, 255, 0,   255, 0,   255, 255,
                                0, 255, 0,   255, 255, 0, 255, 0,   255, 255, 255, 255};

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
    task_data->inputs_count.emplace_back(value.size());

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
    task_data->inputs_count.emplace_back(pict.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(res.size());
  }

  frolova_e_sobel_filter_all::SobelFilterALL test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(reference, res);
  }
}

TEST(frolova_e_sobel_filter_all, small_image_1) {
  boost::mpi::communicator world;
  std::vector<int> value = {3, 3};
  std::vector<int> pict = {172, 47,  117, 192, 67, 251, 195, 103, 9,  211, 21, 242, 3,  87,
                           70,  216, 88,  140, 58, 193, 230, 39,  87, 174, 88, 81,  165};

  std::vector<int> res(9, 0);

  std::vector<int> reference = {255, 255, 255, 255, 53, 255, 255, 255, 255};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
    task_data->inputs_count.emplace_back(value.size());

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
    task_data->inputs_count.emplace_back(pict.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(res.size());
  }

  // Create Task
  frolova_e_sobel_filter_all::SobelFilterALL test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(reference, res);
  }
}

TEST(frolova_e_sobel_filter_all, one_pixel) {
  boost::mpi::communicator world;
  std::vector<int> value = {1, 1};
  std::vector<int> pict = {0, 0, 0};

  std::vector<int> res(1, 0);

  std::vector<int> reference = {0};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
    task_data->inputs_count.emplace_back(value.size());

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
    task_data->inputs_count.emplace_back(pict.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(res.size());
  }

  // Create Task
  frolova_e_sobel_filter_all::SobelFilterALL test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(reference, res);
  }
}

TEST(frolova_e_sobel_filter_all, _random_picture) {
  boost::mpi::communicator world;
  std::vector<int> value = {50, 50};
  std::vector<int> pict = GenRgbPicture(50, 50);

  std::vector<int> res(2500, 0);
  std::vector<int> reference(2500, 0);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
    task_data->inputs_count.emplace_back(value.size());

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
    task_data->inputs_count.emplace_back(pict.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(res.size());
  }

  // Create Task
  frolova_e_sobel_filter_all::SobelFilterALL test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);

  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  if (world.rank() == 0) {
    std::vector<frolova_e_sobel_filter_all::RGB> picture = ConvertToRGB(pict);
    std::vector<int> gray_scale_image =
        ToGrayScaleImgNS(picture, static_cast<size_t>(value[0]), static_cast<size_t>(value[1]));

    reference = ApplySobelFilter(gray_scale_image, value[0], value[1]);
    EXPECT_EQ(reference, res);
  }
}

// FALSE_VALIDATION

TEST(frolova_e_sobel_filter_all, not_correct_value) {
  boost::mpi::communicator world;
  std::vector<int> value = {-1, 1};
  std::vector<int> pict = {100, 0, 0};

  std::vector<int> res(1, 0);
  std::vector<int> reference = {0};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
    task_data->inputs_count.emplace_back(value.size());

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
    task_data->inputs_count.emplace_back(pict.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(res.size());

    // Create Task
    frolova_e_sobel_filter_all::SobelFilterALL test_task(task_data);
    ASSERT_EQ(test_task.Validation(), false);
  }
}

TEST(frolova_e_sobel_filter_all, vector_is_not_multiple_of_three) {
  boost::mpi::communicator world;
  std::vector<int> value = {1, 1};
  std::vector<int> pict = {100, 0};

  std::vector<int> res(1, 0);

  std::vector<int> reference = {0};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
    task_data->inputs_count.emplace_back(value.size());

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
    task_data->inputs_count.emplace_back(pict.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(res.size());

    // Create Task
    frolova_e_sobel_filter_all::SobelFilterALL test_task(task_data);
    ASSERT_EQ(test_task.Validation(), false);
  }
}

TEST(frolova_e_sobel_filter_all, vector_element_is_not_included_the_range) {
  boost::mpi::communicator world;
  std::vector<int> value = {1, 1};
  std::vector<int> pict = {100, 0, 270};

  std::vector<int> res(1, 0);

  std::vector<int> reference = {0};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
    task_data->inputs_count.emplace_back(value.size());

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
    task_data->inputs_count.emplace_back(pict.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(res.size());

    // Create Task
    frolova_e_sobel_filter_all::SobelFilterALL test_task(task_data);
    ASSERT_EQ(test_task.Validation(), false);
  }
}

TEST(frolova_e_sobel_filter_all, negative_value_of_element_int_RGBvector) {
  boost::mpi::communicator world;
  std::vector<int> value = {1, 1};
  std::vector<int> pict = {100, 0, -1};

  std::vector<int> res(1, 0);

  std::vector<int> reference = {0};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
    task_data->inputs_count.emplace_back(value.size());

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
    task_data->inputs_count.emplace_back(pict.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(res.size());

    // Create Task
    frolova_e_sobel_filter_all::SobelFilterALL test_task(task_data);
    ASSERT_EQ(test_task.Validation(), false);
  }
}

TEST(frolova_e_sobel_filter_all, zero_value_of_picture) {
  boost::mpi::communicator world;
  std::vector<int> value = {0, 0};
  std::vector<int> pict = {100, 0, 0};

  std::vector<int> res(1, 0);

  std::vector<int> reference = {0};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
    task_data->inputs_count.emplace_back(value.size());

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
    task_data->inputs_count.emplace_back(pict.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(res.size());

    // Create Task
    frolova_e_sobel_filter_all::SobelFilterALL test_task(task_data);
    ASSERT_EQ(test_task.Validation(), false);
  }
}
