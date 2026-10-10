#include <iostream>
#include <opencv2/opencv.hpp>
#include "net.h"

int main()
{
    std::cout << "===== NCNN + OpenCV Test =====" << std::endl;

    std::cout << "OpenCV version: "
              << CV_VERSION << std::endl;

    ncnn::Net net;

    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = false;
    net.opt.num_threads = 4;

    std::cout << "NCNN Net created successfully." << std::endl;
    std::cout << "Threads: 4" << std::endl;

    cv::Mat image = cv::Mat::zeros(640, 640, CV_8UC3);

    if (image.empty())
    {
        std::cerr << "OpenCV image creation failed." << std::endl;
        return -1;
    }

    std::cout << "OpenCV image test passed." << std::endl;
    std::cout << "All tests passed." << std::endl;

    return 0;
}
