#include <stdio.h>

#include <iostream>
#include <vector>

#include "dataset.h"
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
}
int main() {
    // load dataset
    std::string DATA_DIR = "/home/vinc3/Projects/libtorch_impls/skin_cancer/data";
    // "/home/vinc3/Projects/libtorch_impls/skin_cancer/data/train"
    std::string TRAIN_DIR = DATA_DIR + "/train";
    // "/home/vinc3/Projects/libtorch_impls/skin_cancer/data/val"
    std::string VAL_DIR = DATA_DIR + "/val";

    std::vector<std::string> train_folder_images = get_image_folders(TRAIN_DIR);
    std::vector<std::string> val_folder_images = get_image_folders(VAL_DIR);
    std::cout << train_folder_images<<"\n";

    std::pair<std::vector<std::string>, std::vector<int>> train_images_labels = load_data_from_folder(train_folder_images);
    std::cout << train_images_labels;
}