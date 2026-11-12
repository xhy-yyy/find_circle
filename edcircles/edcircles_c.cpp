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
#include <limits>
#include <vector>

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

void huber_circle_impl(const double* points, int count, double* circle, double delta, int iterations) {
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
        huber_circle_impl(points, count, circle, delta, iterations);
        return 0;
    } catch (...) {
        return -1;
    }
}

// 圆模型 RANSAC + Huber 精修（与 verify._fit_circle 的 numpy 实现同序）。
// samples: iterations x 3 个不重复点下标。返回最终内点数；0 = 没有合格拟合；-1 输入非法。
EDC_API int edc_ransac_circle(const double* points, int count, const int32_t* samples, int iterations,
                             double tau, double r_min, double r_max, double* circle, uint8_t* mask) {
    if (!points || !samples || !circle || !mask || count < 5 || iterations <= 0 || tau <= 0) return -1;
    try {
        std::vector<uint8_t> usable(iterations, 0);
        std::vector<double> centers(static_cast<size_t>(iterations) * 2);
        std::vector<double> radii(iterations);
        for (int k = 0; k < iterations; ++k) {
            const int ia = samples[3 * k], ib = samples[3 * k + 1], ic = samples[3 * k + 2];
            if (ia < 0 || ia >= count || ib < 0 || ib >= count || ic < 0 || ic >= count) return -1;
            double ax = points[2 * ia], ay = points[2 * ia + 1];
            double bx = points[2 * ib], by = points[2 * ib + 1];
            double cx = points[2 * ic], cy = points[2 * ic + 1];
            double d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
            if (std::abs(d) < 1e-9) continue;
            double a2 = ax * ax + ay * ay, b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
            double ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
            double uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
            double radius = std::hypot(ax - ux, ay - uy);
            if (!std::isfinite(radius) || radius < r_min || radius > r_max) continue;
            usable[k] = 1;
            centers[2 * k] = ux; centers[2 * k + 1] = uy;
            radii[k] = radius;
        }
        int best = -1, best_hits = 0;
        std::vector<uint8_t> best_mask(count, 0);
        for (int k = 0; k < iterations; ++k) {
            if (!usable[k]) continue;
            int hits = 0;
            for (int i = 0; i < count; ++i) {
                double res = std::abs(std::hypot(points[2 * i] - centers[2 * k],
                                                 points[2 * i + 1] - centers[2 * k + 1]) - radii[k]);
                hits += res <= tau;
            }
            if (hits > best_hits) {
                best_hits = hits;
                best = k;
            }
        }
        if (best < 0) return 0;
        auto fill = [&](std::vector<uint8_t>& out) {
            for (int i = 0; i < count; ++i) {
                out[i] = std::abs(std::hypot(points[2 * i] - centers[2 * best],
                                             points[2 * i + 1] - centers[2 * best + 1]) - radii[best]) <= tau;
            }
        };
        fill(best_mask);
        if (best_hits < 5) return 0;
        std::vector<double> subset(static_cast<size_t>(count) * 2);
        auto gather = [&](const std::vector<uint8_t>& inliers, std::vector<double>& out) {
            int n = 0;
            for (int i = 0; i < count; ++i) {
                if (!inliers[i]) continue;
                out[2 * n] = points[2 * i]; out[2 * n + 1] = points[2 * i + 1];
                n++;
            }
            return n;
        };
        double cen[2] = {centers[2 * best], centers[2 * best + 1]};
        double radius = radii[best];
        int used = gather(best_mask, subset);
        double cylinder[3] = {cen[0], cen[1], radius};
        huber_circle_impl(subset.data(), used, cylinder, tau / 2, 10);
        cen[0] = cylinder[0]; cen[1] = cylinder[1]; radius = cylinder[2];
        for (int step = 0; step < 2; ++step) {
            if (!(radius >= r_min && radius <= r_max) || !std::isfinite(cen[0]) || !std::isfinite(cen[1])) return 0;
            int hits = 0;
            for (int i = 0; i < count; ++i) {
                best_mask[i] = std::abs(std::hypot(points[2 * i] - cen[0], points[2 * i + 1] - cen[1]) - radius) <= tau;
                hits += best_mask[i];
            }
            if (hits < 5) return 0;
            used = gather(best_mask, subset);
            double next[3] = {cen[0], cen[1], radius};
            huber_circle_impl(subset.data(), used, next, tau / 2, 10);
            cen[0] = next[0]; cen[1] = next[1]; radius = next[2];
        }
        if (!(radius >= r_min && radius <= r_max) || !std::isfinite(cen[0]) || !std::isfinite(cen[1])) return 0;
        int hits = 0;
        for (int i = 0; i < count; ++i) {
            mask[i] = std::abs(std::hypot(points[2 * i] - cen[0], points[2 * i + 1] - cen[1]) - radius) <= tau;
            hits += mask[i];
        }
        circle[0] = cen[0]; circle[1] = cen[1]; circle[2] = radius;
        return hits;
    } catch (...) {
        return -1;
    }
}

// medians 非空时，额外输出每档线宽"每条射线最高分"的中位数（inside 为射线有效掩码，可为空）。
// _peaks 用它选线宽档：每帧约 30 次小数组 np.median，Python 侧约 2.5 ms，原生一次算完。
EDC_API int edc_ray_scores(const double* profiles, int rays, int samples,
                          const int32_t* positions, int count, const double* widths, int width_count,
                          double step, double gap, double side_ratio, const uint8_t* inside,
                          double* scores, double* medians) {
    if (!profiles || !positions || !widths || !scores || rays <= 0 || samples < 2
            || count <= 0 || width_count <= 0 || step <= 0) return -1;
    try {
        std::vector<double> sums(static_cast<size_t>(rays) * (samples + 1));
        for (int j = 0; j < rays; ++j) {
            double* row = sums.data() + j * (samples + 1);
            row[0] = 0;
            for (int i = 0; i < samples; ++i) row[i + 1] = row[i] + profiles[j * samples + i];
        }
        int fail = 0;
        for (int k = 0; k < width_count; ++k) {
            double width = widths[k];
            int nb = 2 * static_cast<int>(std::floor(width / (2 * step))) + 1;
            int ng = static_cast<int>(std::floor(gap / step + 0.5));
            int nl = std::max(2, static_cast<int>(std::floor(std::max(2.0, 0.5 * width) / step + 0.5)));
            if (nb > samples || nl > samples) {
                fail = 1;
                continue;
            }
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
        if (fail) return -1;
        if (medians != nullptr) {
            for (int k = 0; k < width_count; ++k) {
                std::vector<double> values;
                values.reserve(rays);
                for (int j = 0; j < rays; ++j) {
                    if (inside != nullptr && !inside[j]) continue;
                    const double* s = scores + (static_cast<size_t>(k) * rays + j) * count;
                    double m = s[0];
                    for (int i = 1; i < count; ++i) m = std::max(m, s[i]);
                    values.push_back(m);
                }
                const size_t n = values.size();
                if (n == 0) {
                    medians[k] = std::numeric_limits<double>::quiet_NaN();
                } else if (n & 1) {
                    std::nth_element(values.begin(), values.begin() + n / 2, values.end());
                    medians[k] = values[n / 2];
                } else {
                    std::nth_element(values.begin(), values.begin() + n / 2, values.end());
                    double hi = values[n / 2];
                    std::nth_element(values.begin(), values.begin() + n / 2 - 1, values.begin() + n / 2);
                    medians[k] = 0.5 * (values[n / 2 - 1] + hi);
                }
            }
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

// 椭圆假设 + 内点精修一次算完：60 个 5 点假设并行评估（结果按原顺序归约，与串行逐位一致），
// 最优假设的内点上再做两轮 fitEllipseDirect 精修。
// 返回 >=0 为最终内点数；-1 输入非法（Python 退回纯 Python 路径）；-2 精修失败（调用方沿用上一轮结果）。
EDC_API int edc_ellipse_fit(const double* points, int count, const int32_t* samples,
                           int iterations, double radius, double tolerance,
                           double axis_ratio, double* ellipse, uint8_t* mask) {
    if (!points || !samples || !ellipse || !mask || count < 5 || iterations <= 0) return -1;
    try {
        std::vector<int> hits(iterations, 0);
        std::vector<double> params(static_cast<size_t>(iterations) * 5);
        std::vector<uint8_t> masks(static_cast<size_t>(iterations) * count);
        int bad_input = 0;
        #pragma omp parallel for schedule(static) num_threads(4)
        for (int k = 0; k < iterations; ++k) {
            cv::Point2f sample[5];
            for (int j = 0; j < 5; ++j) {
                int index = samples[5 * k + j];
                if (index < 0 || index >= count) {
                    bad_input = 1;
                    break;
                }
                sample[j] = cv::Point2f(points[2 * index], points[2 * index + 1]);
            }
            if (bad_input) continue;
            cv::RotatedRect e;
            try { e = cv::fitEllipseDirect(cv::Mat(5, 1, CV_32FC2, sample)); } catch (...) { continue; }
            double a = e.size.width / 2.0, b = e.size.height / 2.0;
            if (!std::isfinite(a) || !std::isfinite(b) || std::min(a, b) < 0.5 * radius
                    || std::max(a, b) > 1.6 * radius || std::min(a, b) < axis_ratio * std::max(a, b)) continue;
            double angle = e.angle * CV_PI / 180.0;
            double cs = std::cos(angle), sn = std::sin(angle);
            uint8_t* inliers = masks.data() + static_cast<size_t>(k) * count;
            int hit = 0;
            for (int i = 0; i < count; ++i) {
                double x = points[2 * i] - e.center.x, y = points[2 * i + 1] - e.center.y;
                double qx = (x * cs + y * sn) / a, qy = (-x * sn + y * cs) / b;
                double residual = (std::hypot(qx, qy) - 1) * (a + b) / 2;
                inliers[i] = std::abs(residual) <= tolerance;
                hit += inliers[i];
            }
            hits[k] = hit;
            double* p = params.data() + 5 * k;
            p[0] = e.center.x; p[1] = e.center.y; p[2] = a; p[3] = b; p[4] = e.angle;
        }
        if (bad_input) return -1;
        // 内点数相同时保留先出现的假设
        int best = 0, best_k = -1;
        for (int k = 0; k < iterations; ++k) {
            if (hits[k] > best) { best = hits[k]; best_k = k; }
        }
        if (best_k < 0) return 0;
        std::copy(masks.begin() + static_cast<size_t>(best_k) * count,
                  masks.begin() + static_cast<size_t>(best_k + 1) * count, mask);
        std::copy(params.begin() + 5 * best_k, params.begin() + 5 * (best_k + 1), ellipse);
        if (best < 12) return best;
        // 内点上最小二乘精修两次
        for (int refine = 0; refine < 2; ++refine) {
            std::vector<cv::Point2f> selected;
            selected.reserve(count);
            for (int i = 0; i < count; ++i) {
                if (mask[i]) selected.push_back(cv::Point2f(points[2 * i], points[2 * i + 1]));
            }
            if (selected.size() < 5) return -2;
            cv::RotatedRect e;
            try { e = cv::fitEllipseDirect(selected); } catch (...) { return -2; }
            double a = e.size.width / 2.0, b = e.size.height / 2.0;
            if (!std::isfinite(a) || !std::isfinite(b) || std::min(a, b) <= 0
                    || std::min(a, b) < axis_ratio * std::max(a, b)) return -2;
            double angle = e.angle * CV_PI / 180.0;
            double cs = std::cos(angle), sn = std::sin(angle);
            int hit = 0;
            for (int i = 0; i < count; ++i) {
                double x = points[2 * i] - e.center.x, y = points[2 * i + 1] - e.center.y;
                double qx = (x * cs + y * sn) / a, qy = (-x * sn + y * cs) / b;
                double residual = (std::hypot(qx, qy) - 1) * (a + b) / 2;
                mask[i] = std::abs(residual) <= tolerance;
                hit += mask[i];
            }
            if (hit < 12) return -2;
            ellipse[0] = e.center.x; ellipse[1] = e.center.y;
            ellipse[2] = a; ellipse[3] = b; ellipse[4] = e.angle;
            best = hit;
        }
        return best;
    } catch (...) {
        return -1;
    }
}

// 每条射线的峰：列最大值（并列取最早）+ 抛物线亚像素，输出峰位、半径偏移后的距离与有效性。
// scores 为某一档线宽的分数矩阵（rays x count），s_iw 为每个列位置对应的相对半径偏移。
EDC_API int edc_peak_pick(const double* scores, int rays, int count,
                          const double* r_ray, const double* s_iw,
                          double step, double threshold, const uint8_t* inside,
                          int32_t* index, double* peak, double* distance, uint8_t* valid) {
    if (!scores || !r_ray || !s_iw || !index || !peak || !distance || !valid
            || rays <= 0 || count <= 0 || step <= 0) return -1;
    try {
        for (int j = 0; j < rays; ++j) {
            const double* row = scores + static_cast<size_t>(j) * count;
            int k = 0;
            double best = row[0];
            for (int i = 1; i < count; ++i) {
                if (row[i] > best) { best = row[i]; k = i; }
            }
            double y0 = row[k > 0 ? k - 1 : 0];
            double y2 = row[k < count - 1 ? k + 1 : count - 1];
            double den = y0 - 2 * best + y2;
            double off = 0.0;
            if (den < 0 && k > 0 && k < count - 1) off = 0.5 * (y0 - y2) / den;
            if (off < -0.5) off = -0.5;
            else if (off > 0.5) off = 0.5;
            index[j] = k;
            peak[j] = best;
            distance[j] = r_ray[j] + s_iw[k] + off * step;
            valid[j] = (inside == nullptr || inside[j]) && (best >= threshold);
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

}  // extern "C"
