#include "dataset.h"

torch::Tensor read_data(std::string image_path) {
    cv::Mat img = cv::imread(image_path, 1);
    cv::resize(img, img, cv::Size(224, 224), cv::INTER_CUBIC);
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1});
    return img_tensor.clone();
};

torch::Tensor read_labels(int label) {
    torch::Tensor label_tensor = torch::full({1}, label);
    return label_tensor.clone();  // not sure if this is necessary since main reason for cloning the images tensors is the underlying shared memory
};

// std::vector<torch::Tensor> process_images(std::vector<std::string> list_of_images){

// };
// std::vector<torch::Tensor> process_labels(std::vector<int> list_of_labels);
std::pair<std::vector<std::string>, std::vector<int>> load_data_from_folder(std::vector<std::string> folder_names) {
    std::vector<std::string> list_of_images;
    std::vector<int> list_of_labels;
    int label = 0;

    for (auto &value : folder_names) {
        std::string base_name = value + "/";
        DIR *dir;
        struct dirent *ent;
        // if base_name not null -> "data/"
        // https://stackoverflow.com/questions/7416445/what-is-use-of-c-str-function-in-c
        if ((dir = opendir(base_name.c_str())) != NULL) {
            // iterate over the directories and contents
            while ((ent = readdir(dir)) != NULL) {
                std::string filename = ent->d_name;
                if (filename.length() > 4 && filename.substr(filename.length() - 3) == "jpg") {
                    list_of_images.push_back(base_name + ent->d_name);
                    list_of_labels.push_back(label);
                }
            }
            closedir(dir);
        } else {
            std::cout << "Couldn't open the directory" << std::endl;
        }
    }
    return std::make_pair(list_of_images, list_of_labels);
};