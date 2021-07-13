#include <stdio.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include "dataset.h"
#include "trainer.h"
std::vector<std::string> get_image_folders(std::string path) {
    const char *PATH = path.c_str();
    DIR *dir = opendir(PATH);
    struct dirent *ent = readdir(dir);

    std::vector<std::string> folder_names;
    while ((ent = readdir(dir)) != NULL) {
        if (strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0) {
            folder_names.push_back(path + "/" + ent->d_name);
            // std::cout << (ent->d_name) << "\n";
        }
    }
    closedir(dir);
    return folder_names;
};

int main(int argc, const char *argv[]) {
    // load dataset
    std::string DATA_DIR = "/home/vinc3/Projects/libtorch_impls/skin_cancer/data";
    // "/home/vinc3/Projects/libtorch_impls/skin_cancer/data/train"
    std::string TRAIN_DIR = DATA_DIR + "/train";
    // "/home/vinc3/Projects/libtorch_impls/skin_cancer/data/val"
    std::string VAL_DIR = DATA_DIR + "/val";

    std::vector<std::string> train_folder_images = get_image_folders(TRAIN_DIR);
    std::vector<std::string> val_folder_images = get_image_folders(VAL_DIR);

    // evrything train
    std::pair<std::vector<std::string>, std::vector<int>> train_images_labels = load_data_from_folder(train_folder_images);
    std::vector<std::string> list_train_images = train_images_labels.first;
    std::vector<int> list_train_labels = train_images_labels.second;

    // trainloader
    auto train_dataset = CDataset(list_train_images, list_train_labels)
                             .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), 4);
    torch::jit::script::Module module;
    if (argc == 1) {
        module = torch::jit::load("resnet18_without_last_layer.pt");

    } else {
        module = torch::jit::load(argv[1]);
    }
    torch::nn::Linear linear(512, 3);
    torch::optim::Adam optimizer(linear->parameters(), torch::optim::AdamOptions(0.001));

    float train_size = train_dataset.size().value();
    std::cout << train_folder_images << "\n"
              << linear << train_size << std::endl;
    // train(module, linear, train_loader, optimizer, );
}
