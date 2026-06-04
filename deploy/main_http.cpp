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
#include <thread>
#include <mutex>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstring>

static std::atomic<bool> g_running{true};
static cv::Mat g_display_frame;
static std::mutex g_frame_mtx;

static void signal_handler(int) { g_running = false; }

static const float EXTEND_RATIO = 12.0f / 5.0f;
static const char* CLASS_NAMES[] = {
    "A", "B", "C", "D", "E", "F"
};
static const char* CLASS_NAMES_FULL[] = {
    "A-Gun", "B-Explosive", "C-FirstAid", "D-Binoculars", "E-APC", "F-Ambulance"
};
static const int NUM_CLASSES = 6;

struct Config {
    std::string model_path = "best.onnx";
    std::string camera_dev = "/dev/video0";
    int camera_width = 320;
    int camera_height = 240;
    int camera_fps = 60;
    int imgsz = 64;
    int output_width = 256;
    int http_port = 8080;
    float conf_thres = 0.25f;
    float first_aid_conf_thres = 0.80f;
    int red_h_low1 = 0, red_h_high1 = 12;
    int red_h_low2 = 168, red_h_high2 = 180;
    int red_s_min = 80, red_v_min = 50;
    float min_red_area = 80.0f;
    float red_aspect_min = 1.0f, red_aspect_max = 6.0f;
    int morph_kernel = 5;
    float crop_exposure = 1.0f;
    float crop_contrast = 1.0f;
    int frame_skip = 2;
};
Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i+1 < argc) cfg.model_path = argv[++i];
        else if (arg == "--dev" && i+1 < argc) cfg.camera_dev = argv[++i];
        else if (arg == "--camera-width" && i+1 < argc) cfg.camera_width = std::stoi(argv[++i]);
        else if (arg == "--camera-height" && i+1 < argc) cfg.camera_height = std::stoi(argv[++i]);
        else if (arg == "--camera-fps" && i+1 < argc) cfg.camera_fps = std::stoi(argv[++i]);
        else if (arg == "--imgsz" && i+1 < argc) cfg.imgsz = std::stoi(argv[++i]);
        else if (arg == "--port" && i+1 < argc) cfg.http_port = std::stoi(argv[++i]);
        else if (arg == "--conf-thres" && i+1 < argc) cfg.conf_thres = std::stof(argv[++i]);
        else if (arg == "--first-aid-conf-thres" && i+1 < argc) cfg.first_aid_conf_thres = std::stof(argv[++i]);
        else if (arg == "--frame-skip" && i+1 < argc) cfg.frame_skip = std::stoi(argv[++i]);
        else if (arg == "--min-red-area" && i+1 < argc) cfg.min_red_area = std::stof(argv[++i]);
    }
    return cfg;
}

// ============================================================
// Red detection & classification (same logic as main.cpp)
// ============================================================
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
};
DetectResult detect_red_patch(const cv::Mat& frame, const Config& cfg) {
    DetectResult det;
    cv::Mat mask = make_red_mask(frame, cfg);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

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

    float* data = (float*)output.data;
    float max_val = *std::max_element(data, data + NUM_CLASSES);
    float sum = 0;
    for (int i = 0; i < NUM_CLASSES; i++) { data[i] = std::exp(data[i] - max_val); sum += data[i]; }
    for (int i = 0; i < NUM_CLASSES; i++) data[i] /= sum;

    int top1 = (int)(std::max_element(data, data + NUM_CLASSES) - data);
    float conf = data[top1];

    ClassifyResult res;
    res.class_id = top1;
    res.class_name = CLASS_NAMES_FULL[top1];
    res.confidence = conf;
    res.low_confidence = (conf < cfg.conf_thres);
    if (top1 == 2 && conf < cfg.first_aid_conf_thres) {
        res.class_id = -1;
        res.class_name = "unknown";
        res.low_confidence = true;
    }
    return res;
}

// ============================================================
// HTTP MJPEG Server
// ============================================================
static int g_server_fd = -1;

static bool send_all(int fd, const void* data, size_t len) {
    const char* p = (const char*)data;
    while (len > 0) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) return false;
        p += n;
        len -= n;
    }
    return true;
}

static void client_thread(int client_fd) {
    // Set send timeout
    struct timeval tv{2, 0};
    setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    const char* header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=FRAME\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";
    if (!send_all(client_fd, header, strlen(header))) {
        close(client_fd); return;
    }

    while (g_running.load()) {
        std::vector<uchar> jpeg;
        {
            std::lock_guard<std::mutex> lock(g_frame_mtx);
            if (!g_display_frame.empty()) {
                cv::imencode(".jpg", g_display_frame, jpeg,
                    {cv::IMWRITE_JPEG_QUALITY, 60});
            }
        }
        if (jpeg.empty()) { usleep(30000); continue; }

        char part[128];
        int plen = snprintf(part, sizeof(part),
            "--FRAME\r\nContent-Type: image/jpeg\r\n"
            "Content-Length: %zu\r\n\r\n", jpeg.size());
        if (!send_all(client_fd, part, plen)) break;
        if (!send_all(client_fd, jpeg.data(), jpeg.size())) break;
        if (!send_all(client_fd, "\r\n", 2)) break;
        usleep(50000);
    }
    close(client_fd);
}

static void http_accept_loop(int port) {
    g_server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_server_fd < 0) { perror("socket"); return; }

    int opt = 1;
    setsockopt(g_server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(g_server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(g_server_fd); g_server_fd = -1; return;
    }
    if (listen(g_server_fd, 4) < 0) {
        perror("listen"); close(g_server_fd); g_server_fd = -1; return;
    }

    fprintf(stderr, "HTTP MJPEG server on port %d\n", port);

    while (g_running.load()) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(g_server_fd, &fds);
        struct timeval tv{1, 0};
        if (select(g_server_fd + 1, &fds, NULL, NULL, &tv) <= 0) continue;

        struct sockaddr_in cli{};
        socklen_t cli_len = sizeof(cli);
        int cfd = accept(g_server_fd, (struct sockaddr*)&cli, &cli_len);
        if (cfd < 0) continue;

        // Drain HTTP request with timeout (non-blocking)
        struct timeval rtv{0, 500000};
        setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &rtv, sizeof(rtv));
        char buf[1024];
        recv(cfd, buf, sizeof(buf), 0);

        fprintf(stderr, "Client connected\n");
        std::thread t(client_thread, cfd);
        t.detach();
    }
    close(g_server_fd);
}

// ============================================================
// Main: camera capture + inference + overlay + HTTP
// ============================================================
int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGPIPE, SIG_IGN);

    Config cfg = parse_args(argc, argv);

    fprintf(stderr, "Loading model: %s\n", cfg.model_path.c_str());
    cv::dnn::Net net = cv::dnn::readNetFromONNX(cfg.model_path);
    if (net.empty()) {
        fprintf(stderr, "Failed to load model: %s\n", cfg.model_path.c_str());
        return 1;
    }
    fprintf(stderr, "Model loaded OK\n");

    fprintf(stderr, "Opening camera: %s\n", cfg.camera_dev.c_str());
    cv::VideoCapture cap(cfg.camera_dev);
    if (!cap.isOpened()) {
        fprintf(stderr, "Failed to open camera: %s\n", cfg.camera_dev.c_str());
        return 1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.camera_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.camera_height);
    cap.set(cv::CAP_PROP_FPS, cfg.camera_fps);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y','U','Y','V'));

    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    fprintf(stderr, "Camera: %dx%d\n", w, h);

    // Start HTTP server thread
    std::thread http_thread(http_accept_loop, cfg.http_port);
    http_thread.detach();

    fprintf(stderr, "Running... open http://<board-ip>:%d in browser\n", cfg.http_port);

    int frame_idx = 0;
    while (g_running.load()) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            usleep(10000);
            continue;
        }
        frame_idx++;
        if (frame_idx % std::max(1, cfg.frame_skip) != 0) continue;

        // Run inference
        DetectResult det = detect_red_patch(frame, cfg);
        cv::Mat display = frame.clone();

        if (det.success) {
            // Draw red quad (green outline)
            for (int i = 0; i < 4; i++) {
                cv::line(display, det.red_quad[i], det.red_quad[(i+1)%4],
                         cv::Scalar(0, 255, 0), 2);
            }
            // Warp and classify
            auto image_quad = estimate_image_quad(det.red_quad);
            cv::Mat crop = warp_image_area(frame, image_quad, cfg.output_width);
            ClassifyResult cls = classify(net, crop, cfg);

            // Draw result text
            char text[128];
            snprintf(text, sizeof(text), "%s %.0f%%",
                     cls.class_name.c_str(), cls.confidence * 100);
            cv::Scalar color = cls.low_confidence ?
                cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0);
            cv::putText(display, text, cv::Point(5, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        } else {
            cv::putText(display, "No target", cv::Point(5, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }

        // Update shared frame for HTTP clients
        {
            std::lock_guard<std::mutex> lock(g_frame_mtx);
            g_display_frame = display;
        }
    }

    cap.release();
    if (g_server_fd >= 0) close(g_server_fd);
    fprintf(stderr, "Exiting.\n");
    return 0;
}
