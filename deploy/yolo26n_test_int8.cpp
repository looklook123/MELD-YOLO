#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "net.h"

int main()
{
    const std::string model_param =
        "/home/pi/meld-yolo/yolo26n_int8/model_int8.param";

    const std::string model_bin =
        "/home/pi/meld-yolo/yolo26n_int8/model_int8.bin";

    const std::string image_path =
        "/home/pi/datasets/test/images/Corn_Blight-84-_jpg.rf.e03cbdcd9dabfd912b2751fd2207ea73.jpg";

    std::cout << "===== YOLO26 NCNN Output Inspection =====\n";

    // ------------------------------------------------------------
    // 1. Read image
    // ------------------------------------------------------------
    cv::Mat image = cv::imread(image_path);

    if (image.empty())
    {
        std::cerr << "ERROR: Cannot read image:\n"
                  << image_path << std::endl;
        return -1;
    }

    std::cout << "Image size: "
              << image.cols << " x "
              << image.rows << std::endl;

    // ------------------------------------------------------------
    // 2. Resize to 640x640
    // ------------------------------------------------------------
    cv::Mat resized;

    cv::resize(
        image,
        resized,
        cv::Size(640, 640)
    );

    cv::cvtColor(
        resized,
        resized,
        cv::COLOR_BGR2RGB
    );

    // ------------------------------------------------------------
    // 3. Load NCNN model
    // ------------------------------------------------------------
    ncnn::Net net;

    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = false;
    net.opt.num_threads = 4;

    if (net.load_param(model_param.c_str()) != 0)
    {
        std::cerr << "ERROR: Failed to load param." << std::endl;
        return -1;
    }

    if (net.load_model(model_bin.c_str()) != 0)
    {
        std::cerr << "ERROR: Failed to load model." << std::endl;
        return -1;
    }

    std::cout << "NCNN model loaded successfully." << std::endl;

    // ------------------------------------------------------------
    // 4. Convert input
    // ------------------------------------------------------------
    ncnn::Mat in = ncnn::Mat::from_pixels(
        resized.data,
        ncnn::Mat::PIXEL_RGB,
        640,
        640
    );

    const float norm_vals[3] = {
        1.0f / 255.0f,
        1.0f / 255.0f,
        1.0f / 255.0f
    };

    in.substract_mean_normalize(
        nullptr,
        norm_vals
    );

    // ------------------------------------------------------------
    // 5. Inference
    // ------------------------------------------------------------
    ncnn::Extractor ex = net.create_extractor();

    if (ex.input("in0", in) != 0)
    {
        std::cerr << "ERROR: Failed to set input." << std::endl;
        return -1;
    }

    ncnn::Mat out;

    if (ex.extract("out0", out) != 0)
    {
        std::cerr << "ERROR: Failed to extract out0." << std::endl;
        return -1;
    }

    // ------------------------------------------------------------
    // 6. Output shape
    // ------------------------------------------------------------
    std::cout << "\n===== Output Shape =====\n";

    std::cout << "dims = " << out.dims << std::endl;
    std::cout << "w    = " << out.w << std::endl;
    std::cout << "h    = " << out.h << std::endl;
    std::cout << "c    = " << out.c << std::endl;

    if (out.dims != 2 || out.h != 8 || out.w != 8400)
    {
        std::cerr << "Unexpected output shape." << std::endl;
        return -1;
    }

    // ------------------------------------------------------------
    // 7. Read output correctly
    //
    // out.h = 8
    // out.w = 8400
    //
    // Row 0: box value 0
    // Row 1: box value 1
    // Row 2: box value 2
    // Row 3: box value 3
    // Row 4: class 0 score
    // Row 5: class 1 score
    // Row 6: class 2 score
    // Row 7: class 3 score
    // ------------------------------------------------------------

    const float* row0 = out.row(0);
    const float* row1 = out.row(1);
    const float* row2 = out.row(2);
    const float* row3 = out.row(3);
    const float* row4 = out.row(4);
    const float* row5 = out.row(5);
    const float* row6 = out.row(6);
    const float* row7 = out.row(7);

    std::cout << "\n===== First 10 Candidates =====\n";

    for (int i = 0; i < 10; ++i)
    {
        std::cout << "Candidate " << i << ": ";

        std::cout
            << std::fixed
            << std::setprecision(6)
            << row0[i] << ", "
            << row1[i] << ", "
            << row2[i] << ", "
            << row3[i] << ", "
            << row4[i] << ", "
            << row5[i] << ", "
            << row6[i] << ", "
            << row7[i]
            << std::endl;
    }

    // ------------------------------------------------------------
    // 8. Find highest class score
    // ------------------------------------------------------------
    float max_score = -1.0f;
    int best_index = -1;
    int best_class = -1;

    for (int i = 0; i < out.w; ++i)
    {
        float scores[4] = {
            row4[i],
            row5[i],
            row6[i],
            row7[i]
        };

        for (int c = 0; c < 4; ++c)
        {
            if (scores[c] > max_score)
            {
                max_score = scores[c];
                best_index = i;
                best_class = c;
            }
        }
    }

    std::cout << "\n===== Highest Class Score =====\n";

    std::cout
        << "Candidate : "
        << best_index
        << std::endl;

    std::cout
        << "Class     : "
        << best_class
        << std::endl;

    std::cout
        << "Score     : "
        << std::fixed
        << std::setprecision(6)
        << max_score
        << std::endl;

    std::cout << "\n===== Test Finished =====\n";

    return 0;
}
