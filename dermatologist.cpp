#include "dermatologist.h"

torch::Tensor read_data(std::string path) {
    /**
* Returns an image tensor (shape 224,224,3) -> takes in an image path
*
* @param Image path
* @return Image tensor (shape -> 224, 224, 3)
*/
    cv::Mat img = cv::imread(path);
    cv::resize(img, img, cv::Size(224, 224), cv::INTER_CUBIC);
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1});
    return img_tensor.clone();
}
