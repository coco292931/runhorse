#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <atomic>
#include <unistd.h>

static std::atomic<bool> g_running{true};

static void signal_handler(int) {
    g_running = false;
}

static const float PAGE_WIDTH_CM = 12.0f;
static const float IMAGE_HEIGHT_CM = 12.0f;
static const float RED_HEIGHT_CM = 5.0f;
static const float EXTEND_RATIO = IMAGE_HEIGHT_CM / RED_HEIGHT_CM;
static const char* CLASS_NAMES[] = {
    "A-枪支", "B-爆炸物", "C-急救包", "D-望远镜", "E-装甲车", "F-救护车"
};
static const int NUM_CLASSES = 6;

struct Config {
    std::string mode = "webcam";
    std::string model_path = "best.onnx";
    std::string source;
    std::string camera_dev = "/dev/video0";
    int camera_width = 320;
    int camera_height = 240;
    int camera_fps = 60;
    int imgsz = 64;
    int output_width = 720;
    float conf_thres = 0.25f;
    float first_aid_conf_thres = 0.80f;
    int red_h_low1 = 0, red_h_high1 = 12;
    int red_h_low2 = 168, red_h_high2 = 180;
    int red_s_min = 80, red_v_min = 50;
    float min_red_area = 80.0f;
    float red_aspect_min = 1.0f, red_aspect_max = 6.0f;
    int morph_kernel = 5;
    float crop_scale = 1.0f;
    float crop_exposure = 1.0f;
    float crop_contrast = 1.0f;
    int frame_skip = 1;
    bool show = false;
    bool save_debug = false;
};

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--mode" && i+1 < argc) cfg.mode = argv[++i];
        else if (arg == "--model" && i+1 < argc) cfg.model_path = argv[++i];
        else if (arg == "--source" && i+1 < argc) cfg.source = argv[++i];
        else if (arg == "--dev" && i+1 < argc) cfg.camera_dev = argv[++i];
        else if (arg == "--camera-width" && i+1 < argc) cfg.camera_width = std::stoi(argv[++i]);
        else if (arg == "--camera-height" && i+1 < argc) cfg.camera_height = std::stoi(argv[++i]);
        else if (arg == "--camera-fps" && i+1 < argc) cfg.camera_fps = std::stoi(argv[++i]);
        else if (arg == "--imgsz" && i+1 < argc) cfg.imgsz = std::stoi(argv[++i]);
        else if (arg == "--output-width" && i+1 < argc) cfg.output_width = std::stoi(argv[++i]);
        else if (arg == "--conf-thres" && i+1 < argc) cfg.conf_thres = std::stof(argv[++i]);
        else if (arg == "--first-aid-conf-thres" && i+1 < argc) cfg.first_aid_conf_thres = std::stof(argv[++i]);
        else if (arg == "--red-h-low1" && i+1 < argc) cfg.red_h_low1 = std::stoi(argv[++i]);
        else if (arg == "--red-h-high1" && i+1 < argc) cfg.red_h_high1 = std::stoi(argv[++i]);
        else if (arg == "--red-h-low2" && i+1 < argc) cfg.red_h_low2 = std::stoi(argv[++i]);
        else if (arg == "--red-h-high2" && i+1 < argc) cfg.red_h_high2 = std::stoi(argv[++i]);
        else if (arg == "--red-s-min" && i+1 < argc) cfg.red_s_min = std::stoi(argv[++i]);
        else if (arg == "--red-v-min" && i+1 < argc) cfg.red_v_min = std::stoi(argv[++i]);
        else if (arg == "--min-red-area" && i+1 < argc) cfg.min_red_area = std::stof(argv[++i]);
        else if (arg == "--morph-kernel" && i+1 < argc) cfg.morph_kernel = std::stoi(argv[++i]);
        else if (arg == "--crop-scale" && i+1 < argc) cfg.crop_scale = std::stof(argv[++i]);
        else if (arg == "--crop-exposure" && i+1 < argc) cfg.crop_exposure = std::stof(argv[++i]);
        else if (arg == "--crop-contrast" && i+1 < argc) cfg.crop_contrast = std::stof(argv[++i]);
        else if (arg == "--frame-skip" && i+1 < argc) cfg.frame_skip = std::stoi(argv[++i]);
        else if (arg == "--show") cfg.show = true;
        else if (arg == "--save-debug") cfg.save_debug = true;
    }
    return cfg;
}

cv::Mat make_red_mask(const cv::Mat& frame, const Config& cfg) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, cv::Scalar(cfg.red_h_low1, cfg.red_s_min, cfg.red_v_min),
                cv::Scalar(cfg.red_h_high1, 255, 255), mask1);
    cv::inRange(hsv, cv::Scalar(cfg.red_h_low2, cfg.red_s_min, cfg.red_v_min),
                cv::Scalar(cfg.red_h_high2, 255, 255), mask2);
    cv::bitwise_or(mask1, mask2, mask);
    int ks = std::max(1, cfg.morph_kernel);
    if (ks % 2 == 0) ks++;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ks, ks));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), 2);
    return mask;
}

std::vector<cv::Point2f> order_quad(std::vector<cv::Point2f>& pts) {
    std::vector<cv::Point2f> ordered(4);
    std::vector<float> sums(4), diffs(4);
    for (int i = 0; i < 4; i++) {
        sums[i] = pts[i].x + pts[i].y;
        diffs[i] = pts[i].x - pts[i].y;
    }
    ordered[0] = pts[std::min_element(sums.begin(), sums.end()) - sums.begin()];
    ordered[2] = pts[std::max_element(sums.begin(), sums.end()) - sums.begin()];
    ordered[1] = pts[std::max_element(diffs.begin(), diffs.end()) - diffs.begin()];
    ordered[3] = pts[std::min_element(diffs.begin(), diffs.end()) - diffs.begin()];
    return ordered;
}

struct DetectResult {
    bool success = false;
    std::vector<cv::Point2f> red_quad;
    float area = 0, aspect = 0, score = 0;
    cv::Mat mask;
};

DetectResult detect_red_patch(const cv::Mat& frame, const Config& cfg) {
    DetectResult det;
    det.mask = make_red_mask(frame, cfg);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(det.mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    float best_score = -1;
    for (auto& contour : contours) {
        float area = (float)cv::contourArea(contour);
        if (area < cfg.min_red_area) continue;
        cv::RotatedRect rect = cv::minAreaRect(contour);
        float rw = rect.size.width, rh = rect.size.height;
        if (rw <= 1 || rh <= 1) continue;
        float aspect = std::max(rw, rh) / std::min(rw, rh);
        if (aspect < cfg.red_aspect_min || aspect > cfg.red_aspect_max) continue;
        float rect_area = rw * rh;
        float fill = rect_area > 0 ? area / rect_area : 0;
        if (fill < 0.25f) continue;
        float y_weight = rect.center.y / (float)std::max(1, frame.rows);
        float score = area * std::max(fill, 0.01f) * (1.0f + 0.25f * y_weight);
        if (score > best_score) {
            best_score = score;
            det.area = area;
            det.aspect = aspect;
            det.score = score;
            cv::Point2f box[4];
            rect.points(box);
            std::vector<cv::Point2f> pts(box, box + 4);
            det.red_quad = order_quad(pts);
        }
    }
    det.success = (best_score > 0);
    return det;
}

std::vector<cv::Point2f> estimate_image_quad(const std::vector<cv::Point2f>& rq) {
    cv::Point2f tl = rq[0], tr = rq[1], br = rq[2], bl = rq[3];
    cv::Point2f left_ext = (tl - bl) * EXTEND_RATIO;
    cv::Point2f right_ext = (tr - br) * EXTEND_RATIO;
    return {tl + left_ext, tr + right_ext, tr, tl};
}

cv::Mat warp_image_area(const cv::Mat& frame, const std::vector<cv::Point2f>& quad, int w) {
    std::vector<cv::Point2f> dst = {{0,0},{(float)(w-1),0},{(float)(w-1),(float)(w-1)},{0,(float)(w-1)}};
    cv::Mat M = cv::getPerspectiveTransform(quad, dst);
    cv::Mat out;
    cv::warpPerspective(frame, out, M, cv::Size(w, w));
    return out;
}

cv::Mat apply_postprocess(const cv::Mat& crop, const Config& cfg) {
    cv::Mat out = crop.clone();
    if (std::abs(cfg.crop_exposure - 1.0f) > 1e-6f)
        out.convertTo(out, -1, cfg.crop_exposure, 0);
    if (std::abs(cfg.crop_contrast - 1.0f) > 1e-6f) {
        out.convertTo(out, CV_32F);
        out = (out - 127.5f) * cfg.crop_contrast + 127.5f;
        cv::min(out, 255.0f, out);
        cv::max(out, 0.0f, out);
        out.convertTo(out, CV_8U);
    }
    return out;
}

struct ClassifyResult {
    int class_id = -1;
    std::string class_name = "unknown";
    float confidence = 0;
    bool low_confidence = true;
};

ClassifyResult classify(cv::dnn::Net& net, const cv::Mat& crop, const Config& cfg) {
    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(cfg.imgsz, cfg.imgsz));
    cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0/255.0, cv::Size(), cv::Scalar(), true);
    net.setInput(blob);
    cv::Mat output = net.forward();

    // softmax
    float* data = (float*)output.data;
    float max_val = *std::max_element(data, data + NUM_CLASSES);
    float sum = 0;
    for (int i = 0; i < NUM_CLASSES; i++) { data[i] = std::exp(data[i] - max_val); sum += data[i]; }
    for (int i = 0; i < NUM_CLASSES; i++) data[i] /= sum;

    int top1 = (int)(std::max_element(data, data + NUM_CLASSES) - data);
    float conf = data[top1];

    ClassifyResult res;
    res.class_id = top1;
    res.class_name = CLASS_NAMES[top1];
    res.confidence = conf;
    res.low_confidence = (conf < cfg.conf_thres);
    // C-急救包 special threshold
    if (top1 == 2 && conf < cfg.first_aid_conf_thres) {
        res.class_id = -1;
        res.class_name = "unknown";
        res.low_confidence = true;
    }
    return res;
}

void process_frame(cv::Mat& frame, cv::dnn::Net& net, const Config& cfg) {
    DetectResult det = detect_red_patch(frame, cfg);
    if (!det.success) {
        std::cout << "{\"success\":false,\"error\":\"red_patch_not_found\"}" << std::endl;
        return;
    }
    auto image_quad = estimate_image_quad(det.red_quad);
    cv::Mat crop = warp_image_area(frame, image_quad, cfg.output_width);
    crop = apply_postprocess(crop, cfg);
    ClassifyResult cls = classify(net, crop, cfg);
    std::cout << "{\"success\":true,\"class_id\":" << cls.class_id
              << ",\"class_name\":\"" << cls.class_name
              << "\",\"confidence\":" << cls.confidence
              << ",\"low_confidence\":" << (cls.low_confidence ? "true" : "false")
              << "}" << std::endl;
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    Config cfg = parse_args(argc, argv);

    std::cerr << "Loading model: " << cfg.model_path << std::endl;
    cv::dnn::Net net = cv::dnn::readNetFromONNX(cfg.model_path);
    if (net.empty()) {
        std::cerr << "Failed to load model: " << cfg.model_path << std::endl;
        return 1;
    }
    std::cerr << "Model loaded OK" << std::endl;

    if (cfg.mode == "image") {
        if (cfg.source.empty()) {
            std::cerr << "--source required in image mode" << std::endl;
            return 1;
        }
        cv::Mat frame = cv::imread(cfg.source);
        if (frame.empty()) {
            std::cerr << "Failed to read: " << cfg.source << std::endl;
            return 1;
        }
        process_frame(frame, net, cfg);
    } else {
        std::cerr << "Opening camera: " << cfg.camera_dev << std::endl;
        cv::VideoCapture cap(cfg.camera_dev);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open camera: " << cfg.camera_dev << std::endl;
            return 1;
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.camera_width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.camera_height);
        cap.set(cv::CAP_PROP_FPS, cfg.camera_fps);
        bool yuv_ok = cap.set(cv::CAP_PROP_FOURCC,
            cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
        if (!yuv_ok) {
            cap.set(cv::CAP_PROP_FOURCC,
                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        }

        int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(cv::CAP_PROP_FPS);
        std::cerr << "Camera: " << w << "x" << h
                  << " @ " << fps << "fps"
                  << " format=" << (yuv_ok ? "YUYV" : "MJPEG/other")
                  << std::endl;

        int frame_idx = 0;
        while (g_running.load()) {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                usleep(10000);
                continue;
            }
            frame_idx++;
            if (frame_idx % std::max(1, cfg.frame_skip) == 0)
                process_frame(frame, net, cfg);
        }
        cap.release();
        std::cerr << "Camera released, exiting." << std::endl;
    }
    return 0;
}
