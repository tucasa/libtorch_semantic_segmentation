#include <iostream>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#define HEIGHT 256
#define WIDTH 256
#define N_CLASS 10

std::vector<std::string> img_paths;

void get_img_paths(std::string const &path, std::vector<std::string> &img_paths) {
  img_paths.clear();
  struct dirent *entry;

  DIR *dir = opendir(path.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      std::string name = entry->d_name;
      std::string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "jpg") || (ext == "png")) {
        img_paths.push_back(path + "/" + name);
      }
    }
  }
  closedir(dir);
}

torch::Tensor preprocess(cv::Mat &img){
  cv::resize(img, img, cv::Size(WIDTH, HEIGHT));

  torch::Tensor input_tensor = torch::from_blob(img.data, {1, HEIGHT, WIDTH, 3}, at::kByte);
  input_tensor = input_tensor.permute({0,3,1,2});
  input_tensor = input_tensor.to(torch::kFloat);
  input_tensor = input_tensor.div(255);
  
  return input_tensor;
}

int main(){
  torch::DeviceType device_type = torch::kCPU;
  torch::Device device(device_type);
  
  std::string model_path = "../models/model.pt";
  torch::jit::script::Module model = torch::jit::load(model_path);
  model.to(device);
  model.eval();

  std::string img_dir_path = "../images";
  get_img_paths(img_dir_path, img_paths);
  std::sort(img_paths.begin(), img_paths.end());

  for (int i = 0; i < img_paths.size(); i++){
    std::string img_path = img_paths.at(i);
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    torch::Tensor input_tensor = preprocess(img);

    torch::NoGradGuard no_grad;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    torch::Tensor pred = model.forward({input_tensor}).toTensor();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::cout << "Processing time = " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000000.0 << " sec" << std::endl;

    pred = pred.argmax(1);
    pred = pred.squeeze();
    pred = pred.mul(int(255 / N_CLASS)).to(torch::kU8);
    pred = pred.to(torch::kCPU);

    cv::Mat pred_mat(cv::Size(WIDTH, HEIGHT), CV_8U, pred.data_ptr());
    cv::resize(pred_mat, pred_mat, cv::Size(512, 512));
    cv::namedWindow("pred", 0);
    cv::imshow("pred", pred_mat);  // monochrome image
    cv::waitKey(0);
  }
	
  return 0;

}
