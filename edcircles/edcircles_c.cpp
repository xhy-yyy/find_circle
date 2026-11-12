// ED_Lib EDCircles 的 C 接口，给 Python ctypes 调用（ringdet/edcircles.py）。
//
// 流程：ED（给定梯度算子和阈值）->（可选）EDPF 用 Helmholtz 原理验证边缘段 -> EDCircles 找圆和椭圆。
// ED_Lib 的 EDCircles(Mat) 就是 grad_op=PREWITT、grad_thresh=11、anchor_thresh=3、validate_segments=1 的特例。
// 圆和椭圆最后都要过 EDCircles 自己的 NFA 验证，与 validate_segments 无关。

#include <cstdint>
#include <cstring>
#include <exception>
#include <cmath>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "EDLib.h"

namespace {

GradientOperator to_op(int op) {
    switch (op) {
        case 1: return SOBEL_OPERATOR;
        case 2: return SCHARR_OPERATOR;
        case 3: return LSD_OPERATOR;
        default: return PREWITT_OPERATOR;
    }
}

}  // namespace

#define EDC_API __attribute__((visibility("default")))

extern "C" {

// img: 8 位灰度，stride 为每行字节数。
// out: 每个结果 6 个 double：cx, cy, a, b, theta(弧度), kind(0 圆 / 1 椭圆)；圆的 a = b = r。
// 返回检测总数（可能大于 max_out，只写前 max_out 个）；出错返回 -1。
EDC_API int edc_detect(const uint8_t* img, int width, int height, int stride,
               int grad_op, int grad_thresh, int anchor_thresh, int scan_interval,
               int min_path_len, double sigma, int sum_flag, int validate_segments,
               double* out, int max_out) {
    if (img == nullptr || width < 16 || height < 16 || stride < width) return 0;
    try {
        // ED_Lib 按 width 连续寻址，非连续的行要先拷贝
        cv::Mat view(height, width, CV_8UC1, const_cast<uint8_t*>(img), (size_t)stride);
        cv::Mat src = view.isContinuous() ? view : view.clone();
        ED ed(src, to_op(grad_op), grad_thresh, anchor_thresh, scan_interval, min_path_len, sigma,
              sum_flag != 0);
        EDCircles det(ed, validate_segments != 0);
        std::vector<mCircle> cs = det.getCircles();
        std::vector<mEllipse> es = det.getEllipses();
        int n = 0;
        for (const mCircle& c : cs) {
            if (n < max_out && out != nullptr) {
                double* o = out + 6 * n;
                o[0] = c.center.x; o[1] = c.center.y; o[2] = c.r; o[3] = c.r; o[4] = 0.0; o[5] = 0.0;
            }
            n++;
        }
        for (const mEllipse& e : es) {
            if (n < max_out && out != nullptr) {
                double* o = out + 6 * n;
                o[0] = e.center.x; o[1] = e.center.y; o[2] = e.axes.width; o[3] = e.axes.height;
                o[4] = e.theta; o[5] = 1.0;
            }
            n++;
        }
        return n;
    } catch (...) {
        return -1;
    }
}

EDC_API const char* edc_version() { return "ED_Lib@69b8d08"; }

EDC_API int edc_flow_fb(const uint8_t* prev, const uint8_t* current,
                       int width, int height, int prev_stride, int current_stride,
                       const float* points, int count, float* next, float* back,
                       uint8_t* forward_status, uint8_t* backward_status) {
    if (!prev || !current || !points || count <= 0) return -1;
    try {
        cv::Mat a(height, width, CV_8UC1, const_cast<uint8_t*>(prev), prev_stride);
        cv::Mat b(height, width, CV_8UC1, const_cast<uint8_t*>(current), current_stride);
        std::vector<cv::Mat> pa, pb;
        const cv::Size window(21, 21);
        // 前后向跟踪共享这两幅图的金字塔及梯度。
        cv::buildOpticalFlowPyramid(a, pa, window, 3);
        cv::buildOpticalFlowPyramid(b, pb, window, 3);
        cv::Mat p(count, 1, CV_32FC2, const_cast<float*>(points));
        cv::Mat q(count, 1, CV_32FC2, next), r(count, 1, CV_32FC2, back);
        cv::Mat sa(count, 1, CV_8U, forward_status), sb(count, 1, CV_8U, backward_status), error;
        const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
        cv::calcOpticalFlowPyrLK(pa, pb, p, q, sa, error, window, 3, criteria);
        cv::calcOpticalFlowPyrLK(pb, pa, q, r, sb, error, window, 3, criteria);
        return 0;
    } catch (...) {
        return -1;
    }
}

EDC_API int edc_huber_circle(const double* points, int count, double* circle,
                            double delta, int iterations) {
    if (!points || !circle || count < 3 || iterations < 0 || delta <= 0) return -1;
    try {
        double cx = circle[0], cy = circle[1], radius = circle[2];
        for (int k = 0; k < iterations; ++k) {
            cv::Matx33d normal = cv::Matx33d::zeros();
            cv::Vec3d rhs(0, 0, 0);
            for (int i = 0; i < count; ++i) {
                double dx = points[2 * i] - cx, dy = points[2 * i + 1] - cy;
                double distance = std::max(std::hypot(dx, dy), 1e-9);
                double residual = distance - radius;
                double weight = std::abs(residual) <= delta ? 1.0 : delta / std::max(std::abs(residual), 1e-12);
                cv::Vec3d jacobian(-dx / distance, -dy / distance, -1);
                for (int a = 0; a < 3; ++a) {
                    double weighted = jacobian[a] * weight;
                    rhs[a] += weighted * residual;
                    for (int b = 0; b < 3; ++b) normal(a, b) += weighted * jacobian[b];
                }
            }
            for (int a = 0; a < 3; ++a) normal(a, a) += 1e-9;
            cv::Vec3d step;
            if (!cv::solve(normal, -rhs, step, cv::DECOMP_LU)) break;
            cx += step[0]; cy += step[1]; radius += step[2];
            if (std::max({std::abs(step[0]), std::abs(step[1]), std::abs(step[2])}) < 1e-4) break;
        }
        circle[0] = cx; circle[1] = cy; circle[2] = radius;
        return 0;
    } catch (...) {
        return -1;
    }
}

EDC_API int edc_ray_scores(const double* profiles, int rays, int samples,
                          const int32_t* positions, int count, const double* widths, int width_count,
                          double step, double gap, double side_ratio, double* scores) {
    if (!profiles || !positions || !widths || !scores || rays <= 0 || samples < 2
            || count <= 0 || width_count <= 0 || step <= 0) return -1;
    try {
        thread_local std::vector<double> sums;
        sums.resize(static_cast<size_t>(rays) * (samples + 1));
        for (int j = 0; j < rays; ++j) {
            double* row = sums.data() + j * (samples + 1);
            row[0] = 0;
            for (int i = 0; i < samples; ++i) row[i + 1] = row[i] + profiles[j * samples + i];
        }
        for (int k = 0; k < width_count; ++k) {
            double width = widths[k];
            int nb = 2 * static_cast<int>(std::floor(width / (2 * step))) + 1;
            int ng = static_cast<int>(std::floor(gap / step + 0.5));
            int nl = std::max(2, static_cast<int>(std::floor(std::max(2.0, 0.5 * width) / step + 0.5)));
            if (nb > samples || nl > samples) return -1;
            for (int j = 0; j < rays; ++j) {
                const double* row = sums.data() + j * (samples + 1);
                auto average = [&](int first, int length) {
                    first = std::max(0, std::min(first, samples - length));
                    return (row[first + length] - row[first]) / length;
                };
                for (int i = 0; i < count; ++i) {
                    int first = positions[i] - nb / 2;
                    double line = average(first, nb);
                    double a = line - average(first - ng - nl, nl);
                    double b = line - average(first + nb + ng, nl);
                    double score = std::min(a, b);
                    if (std::abs(a - b) > side_ratio * std::max(score, 0.0)) score = 0;
                    scores[(k * rays + j) * count + i] = score;
                }
            }
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

EDC_API int edc_ellipse_hypotheses(const double* points, int count, const int32_t* samples,
                                 int iterations, double radius, double tolerance,
                                 double axis_ratio, double* ellipse, uint8_t* mask) {
    if (!points || !samples || count < 5 || iterations <= 0) return -1;
    int best = 0;
    std::vector<uint8_t> inliers(count);
    std::vector<cv::Point2f> sample(5);
    // 保持输入假设顺序，内点数相同时保留先出现的结果。
    for (int k = 0; k < iterations; ++k) {
        for (int j = 0; j < 5; ++j) {
            int index = samples[5 * k + j];
            if (index < 0 || index >= count) return -1;
            sample[j] = cv::Point2f(points[2 * index], points[2 * index + 1]);
        }
        cv::RotatedRect e;
        try { e = cv::fitEllipseDirect(sample); } catch (...) { continue; }
        double a = e.size.width / 2.0, b = e.size.height / 2.0;
        if (!std::isfinite(a) || !std::isfinite(b) || std::min(a, b) < 0.5 * radius
                || std::max(a, b) > 1.6 * radius || std::min(a, b) < axis_ratio * std::max(a, b)) continue;
        double angle = e.angle * CV_PI / 180.0;
        double cs = std::cos(angle), sn = std::sin(angle);
        int hits = 0;
        for (int i = 0; i < count; ++i) {
            double x = points[2 * i] - e.center.x, y = points[2 * i + 1] - e.center.y;
            double qx = (x * cs + y * sn) / a, qy = (-x * sn + y * cs) / b;
            double residual = (std::hypot(qx, qy) - 1) * (a + b) / 2;
            inliers[i] = std::abs(residual) <= tolerance;
            hits += inliers[i];
        }
        if (hits > best) {
            best = hits;
            ellipse[0] = e.center.x; ellipse[1] = e.center.y;
            ellipse[2] = a; ellipse[3] = b; ellipse[4] = e.angle;
            std::copy(inliers.begin(), inliers.end(), mask);
        }
    }
    return best;
}

}  // extern "C"
