# ED_Lib（第三方代码）

来源：https://github.com/CihanTopal/ED_Lib
提交：69b8d081bd6d28192d816ec0ed02aff9186d73c1（2025-11-07）
许可：MIT（见 LICENSE）

只拷了库文件（ED / EDPF / EDLines / EDCircles / EDColor / NFA），没拷测试程序和图片。
本目录里对原代码的修改都记在下面，其余文件与上游一致。

## 修改记录

1. EDColor.cpp：删除调试用的 `imwrite("newGrad.pgm", gradImage)`。它每次调用都往当前目录写文件，还额外依赖 imgcodecs。
2. EDPF.h/.cpp、EDCircles.h/.cpp：新增 `EDPF(ED obj, bool validateSegments)` 和 `EDCircles(ED obj, bool validateSegments = true)`。
   false 时跳过边缘段的 Helmholtz 验证（等同 OpenCV 移植版的 PFmode=false），高空细环的弱边缘不会被提前删掉；
   圆和椭圆本身仍走 EDCircles 的 NFA 验证。默认 true，与上游行为一致。
3. EDCircles.h：`mEllipse::axes` 由 `cv::Size`（int）改为 `cv::Size2d`，半轴不再截断成整数；drawResult 画图处显式转回 `cv::Size`。
