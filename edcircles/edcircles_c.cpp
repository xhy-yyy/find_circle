// ED_Lib EDCircles 的 C 接口，给 Python ctypes 调用（ringdet/edcircles.py）。
//
// 流程：ED（给定梯度算子和阈值）->（可选）EDPF 用 Helmholtz 原理验证边缘段 -> EDCircles 找圆和椭圆。
// ED_Lib 的 EDCircles(Mat) 就是 grad_op=PREWITT、grad_thresh=11、anchor_thresh=3、validate_segments=1 的特例。
// 圆和椭圆最后都要过 EDCircles 自己的 NFA 验证，与 validate_segments 无关。

#include <cstdint>
#include <cstring>
#include <exception>

#include <opencv2/core.hpp>

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

}  // extern "C"
