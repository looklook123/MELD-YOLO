#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include "net.h"

namespace fs = std::filesystem;

struct Detection
{
    int class_id;
    float score;
    float x1;
    float y1;
    float x2;
    float y2;
};

static float IoU(const Detection& a, const Detection& b)
{
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);

    float inter = w * h;

    float area_a =
        std::max(0.0f, a.x2 - a.x1) *
        std::max(0.0f, a.y2 - a.y1);

    float area_b =
        std::max(0.0f, b.x2 - b.x1) *
        std::max(0.0f, b.y2 - b.y1);

    float uni = area_a + area_b - inter;

    if (uni <= 0.0f)
        return 0.0f;

    return inter / uni;
}

// Class-wise NMS
static std::vector<Detection> NMS(
    const std::vector<Detection>& detections,
    float iou_threshold)
{
    std::vector<Detection> result;

    for (int cls = 0; cls < 4; ++cls)
    {
        std::vector<Detection> cls_dets;

        for (const auto& d : detections)
        {
            if (d.class_id == cls)
                cls_dets.push_back(d);
        }

        std::sort(
            cls_dets.begin(),
            cls_dets.end(),
            [](const Detection& a, const Detection& b)
            {
                return a.score > b.score;
            });

        std::vector<bool> removed(cls_dets.size(), false);

        for (size_t i = 0; i < cls_dets.size(); ++i)
        {
            if (removed[i])
                continue;

            result.push_back(cls_dets[i]);

            for (size_t j = i + 1; j < cls_dets.size(); ++j)
            {
                if (removed[j])
                    continue;

                if (IoU(cls_dets[i], cls_dets[j]) > iou_threshold)
                    removed[j] = true;
            }
        }
    }

    std::sort(
        result.begin(),
        result.end(),
        [](const Detection& a, const Detection& b)
        {
            return a.score > b.score;
        });

    return result;
}

int main()
{
    // ============================================================
    // Configuration
    // ============================================================

    const std::string model_param =
        "/home/pi/meld-yolo/best_ncnn_model/model_int8.param";

    const std::string model_bin =
        "/home/pi/meld-yolo/best_ncnn_model/model_int8.bin";

    const std::string image_dir =
        "/home/pi/datasets/test/images";

    const std::string output_dir =  "/home/pi/meld-yolo/test_results_int8";


    const float conf_threshold = 0.001f;
    const float nms_threshold = 0.70f;

    const int num_classes = 4;

    // ============================================================
    // Create output directory
    // ============================================================

    fs::create_directories(output_dir);

    // ============================================================
    // Load model
    // ============================================================

    ncnn::Net net;

    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = false;
    net.opt.num_threads = 4;

    std::cout << "============================================\n";
    std::cout << " YOLO26 NCNN Raspberry Pi Test Benchmark\n";
    std::cout << "============================================\n";

    std::cout << "Model: " << model_param << "\n";
    std::cout << "Images: " << image_dir << "\n";
    std::cout << "Threads: " << net.opt.num_threads << "\n";
    std::cout << "Confidence: " << conf_threshold << "\n";
    std::cout << "NMS IoU: " << nms_threshold << "\n\n";

    if (net.load_param(model_param.c_str()) != 0)
    {
        std::cerr << "ERROR: Failed to load param.\n";
        return -1;
    }

    if (net.load_model(model_bin.c_str()) != 0)
    {
        std::cerr << "ERROR: Failed to load model.\n";
        return -1;
    }

    std::cout << "Model loaded successfully.\n\n";

    // ============================================================
    // Collect images
    // ============================================================

    std::vector<std::string> image_files;

    for (const auto& entry : fs::directory_iterator(image_dir))
    {
        if (!entry.is_regular_file())
            continue;

        std::string ext = entry.path().extension().string();

        std::transform(
            ext.begin(),
            ext.end(),
            ext.begin(),
            ::tolower);

        if (ext == ".jpg" ||
            ext == ".jpeg" ||
            ext == ".png" ||
            ext == ".bmp")
        {
            image_files.push_back(entry.path().string());
        }
    }

    std::sort(image_files.begin(), image_files.end());

    std::cout << "Test images: "
              << image_files.size()
              << "\n\n";

    if (image_files.empty())
    {
        std::cerr << "ERROR: No images found.\n";
        return -1;
    }

    // ============================================================
    // Warmup
    // ============================================================

    {
        cv::Mat image = cv::imread(
            image_files[0],
            cv::IMREAD_COLOR);

        if (image.empty())
        {
            std::cerr << "ERROR: Cannot read warmup image.\n";
            return -1;
        }

        cv::resize(
            image,
            image,
            cv::Size(640, 640));

        cv::cvtColor(
            image,
            image,
            cv::COLOR_BGR2RGB);

        ncnn::Mat in = ncnn::Mat::from_pixels(
            image.data,
            ncnn::Mat::PIXEL_RGB,
            640,
            640);

        const float norm_vals[3] = {
            1.0f / 255.0f,
            1.0f / 255.0f,
            1.0f / 255.0f
        };

        in.substract_mean_normalize(
            nullptr,
            norm_vals);

        ncnn::Extractor ex = net.create_extractor();

        ex.input("in0", in);

        ncnn::Mat out;

        ex.extract("out0", out);
    }

    std::cout << "Warmup finished.\n\n";

    // ============================================================
    // Benchmark
    // ============================================================

    double total_inference_ms = 0.0;
    double total_end_to_end_ms = 0.0;

    size_t total_detections = 0;

    size_t processed = 0;

    for (const auto& image_path : image_files)
    {
        cv::Mat image = cv::imread(
            image_path,
            cv::IMREAD_COLOR);

        if (image.empty())
        {
            std::cerr << "WARNING: Cannot read "
                      << image_path << "\n";
            continue;
        }

        // --------------------------------------------------------
        // Preprocess
        // --------------------------------------------------------

        auto total_start =
            std::chrono::steady_clock::now();

        cv::Mat resized;

        cv::resize(
            image,
            resized,
            cv::Size(640, 640));

        cv::cvtColor(
            resized,
            resized,
            cv::COLOR_BGR2RGB);

        ncnn::Mat in = ncnn::Mat::from_pixels(
            resized.data,
            ncnn::Mat::PIXEL_RGB,
            640,
            640);

        const float norm_vals[3] = {
            1.0f / 255.0f,
            1.0f / 255.0f,
            1.0f / 255.0f
        };

        in.substract_mean_normalize(
            nullptr,
            norm_vals);

        // --------------------------------------------------------
        // NCNN inference
        // --------------------------------------------------------

        ncnn::Extractor ex = net.create_extractor();

        ex.input("in0", in);

        ncnn::Mat out;

        auto infer_start =
            std::chrono::steady_clock::now();

        if (ex.extract("out0", out) != 0)
        {
            std::cerr << "WARNING: inference failed: "
                      << image_path << "\n";
            continue;
        }

        auto infer_end =
            std::chrono::steady_clock::now();

        double inference_ms =
            std::chrono::duration<double, std::milli>(
                infer_end - infer_start).count();

        // --------------------------------------------------------
        // Check output
        // --------------------------------------------------------

        if (out.dims != 2 ||
            out.w != 8400 ||
            out.h != 8)
        {
            std::cerr
                << "WARNING: unexpected output shape "
                << out.w << " x "
                << out.h << "\n";
            continue;
        }

        const float* row0 = out.row(0);
        const float* row1 = out.row(1);
        const float* row2 = out.row(2);
        const float* row3 = out.row(3);

        const float* row4 = out.row(4);
        const float* row5 = out.row(5);
        const float* row6 = out.row(6);
        const float* row7 = out.row(7);

        // --------------------------------------------------------
        // Decode candidates
        // --------------------------------------------------------

        std::vector<Detection> candidates;

        for (int i = 0; i < 8400; ++i)
        {
            float scores[4] = {
                row4[i],
                row5[i],
                row6[i],
                row7[i]
            };

            int best_class = 0;
            float best_score = scores[0];

            for (int c = 1; c < num_classes; ++c)
            {
                if (scores[c] > best_score)
                {
                    best_score = scores[c];
                    best_class = c;
                }
            }

            if (best_score < conf_threshold)
                continue;

            // NCNN output is:
            // cx, cy, w, h

            float cx = row0[i];
            float cy = row1[i];
            float w  = row2[i];
            float h  = row3[i];

            Detection det;

            det.class_id = best_class;
            det.score = best_score;

            det.x1 = cx - w * 0.5f;
            det.y1 = cy - h * 0.5f;
            det.x2 = cx + w * 0.5f;
            det.y2 = cy + h * 0.5f;

            det.x1 = std::max(0.0f,
                              std::min(640.0f, det.x1));

            det.y1 = std::max(0.0f,
                              std::min(640.0f, det.y1));

            det.x2 = std::max(0.0f,
                              std::min(640.0f, det.x2));

            det.y2 = std::max(0.0f,
                              std::min(640.0f, det.y2));

            candidates.push_back(det);
        }

        // --------------------------------------------------------
        // NMS
        // --------------------------------------------------------

        std::vector<Detection> detections =
            NMS(candidates, nms_threshold);

        total_detections += detections.size();

        // --------------------------------------------------------
        // Save results
        // Format:
        // class confidence x1 y1 x2 y2
        // --------------------------------------------------------

        fs::path image_file(image_path);

        std::string stem =
            image_file.stem().string();

        std::string txt_path =
            output_dir + "/" + stem + ".txt";

        std::ofstream fout(txt_path);

        fout << std::fixed
             << std::setprecision(6);

        for (const auto& d : detections)
        {
            fout << d.class_id << " "
                 << d.score << " "
                 << d.x1 << " "
                 << d.y1 << " "
                 << d.x2 << " "
                 << d.y2 << "\n";
        }

        fout.close();

        auto total_end =
            std::chrono::steady_clock::now();

        double total_ms =
            std::chrono::duration<double, std::milli>(
                total_end - total_start).count();

        total_inference_ms += inference_ms;
        total_end_to_end_ms += total_ms;

        processed++;

        if (processed % 10 == 0 ||
            processed == image_files.size())
        {
            std::cout
                << "\rProcessed "
                << processed
                << " / "
                << image_files.size()
                << std::flush;
        }
    }

    std::cout << "\n\n";

    // ============================================================
    // Final statistics
    // ============================================================

    if (processed == 0)
    {
        std::cerr << "No images processed.\n";
        return -1;
    }

    double avg_inference_ms =
        total_inference_ms / processed;

    double avg_total_ms =
        total_end_to_end_ms / processed;

    double inference_fps =
        1000.0 / avg_inference_ms;

    double end_to_end_fps =
        1000.0 / avg_total_ms;

    double avg_detections =
        static_cast<double>(total_detections)
        / processed;

    // ============================================================
    // Save benchmark.csv
    // ============================================================

    std::string csv_path =
        output_dir + "/benchmark.csv";

    std::ofstream csv(csv_path);

    csv << "model,precision,images,threads,"
        << "avg_inference_ms,inference_fps,"
        << "avg_total_ms,end_to_end_fps,"
        << "total_detections,avg_detections\n";

    csv << "YOLO26,FP32,"
        << processed << ","
        << net.opt.num_threads << ","
        << std::fixed
        << std::setprecision(4)
        << avg_inference_ms << ","
        << inference_fps << ","
        << avg_total_ms << ","
        << end_to_end_fps << ","
        << total_detections << ","
        << avg_detections
        << "\n";

    csv.close();

    // ============================================================
    // Print
    // ============================================================

    std::cout
        << "============================================\n";
    std::cout
        << " FP32 Raspberry Pi Benchmark Result\n";
    std::cout
        << "============================================\n";

    std::cout
        << "Images processed       : "
        << processed << "\n";

    std::cout
        << "Threads                : "
        << net.opt.num_threads << "\n";

    std::cout
        << "Average inference     : "
        << std::fixed
        << std::setprecision(4)
        << avg_inference_ms
        << " ms\n";

    std::cout
        << "Inference FPS         : "
        << std::fixed
        << std::setprecision(4)
        << inference_fps
        << "\n";

    std::cout
        << "Average total time    : "
        << avg_total_ms
        << " ms\n";

    std::cout
        << "End-to-end FPS        : "
        << end_to_end_fps
        << "\n";

    std::cout
        << "Total detections      : "
        << total_detections
        << "\n";

    std::cout
        << "Average detections    : "
        << avg_detections
        << "\n";

    std::cout
        << "\nPrediction directory:\n"
        << output_dir
        << "\n";

    std::cout
        << "\nBenchmark CSV:\n"
        << csv_path
        << "\n";

    return 0;
}

