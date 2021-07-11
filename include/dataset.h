#pragma once
#include <dirent.h>
#include <torch/torch.h>

#include <opencv2/opencv.hpp>

torch::Tensor read_data(std::string image_path);
torch::Tensor read_labels(int label);
std::vector<torch::Tensor> process_images(std::vector<std::string> list_of_images);
std::vector<torch::Tensor> process_labels(std::vector<int> list_of_labels);
std::pair<std::vector<std::string>, std::vector<int>> load_data_from_folder(std::vector<std::string> folder_names);