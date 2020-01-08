#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>


#include <net.h>

using namespace std;

int main() {

    string img_path = "1.jpg";
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::Mat img2;
    int input_width = 227;
    int input_height = 227;
    // resize图片大小 resize到alexnet的输入尺寸
    cv::resize(img, img2, cv::Size(input_width, input_height));

    // 加载转换并且量化后的alexnet网络
    ncnn::Net net;
    net.load_param("alexnet.param");
    net.load_model("alexnet.bin");

    // 把opencv的mat转换成ncnn的mat
    ncnn::Mat input = ncnn::Mat::from_pixels(img2.data, ncnn::Mat::PIXEL_BGR, img2.cols, img2.rows);
    // ncnn前向计算
    ncnn::Extractor extractor = net.create_extractor();

    extractor.input("actual_input_1", input);

    ncnn::Mat output;
    extractor.extract("output1", output);
    // 输出预测结果
    ncnn::Mat out_flatterned = output.reshape(output.w * output.h * output.c);
    std::vector<float> scores;
    scores.resize(out_flatterned.w);
    for (int j=0; j<out_flatterned.w; j++)
    {
        scores[j] = out_flatterned[j];
        // cout << "scores " << j << ":" << scores[j] <<endl;
        printf("scores[%d]: %f\n", j, scores[j]);
    }

    cout<<"end"<<endl;


    // cout << "Built with OpenCV " << CV_VERSION << endl;
    // Mat image;
    // VideoCapture capture;
    // capture.open(0);
    // if(capture.isOpened())
    // {
    //     cout << "Capture is opened" << endl;
    //     for(;;)
    //     {
    //         capture >> image;
    //         if(image.empty())
    //             break;
    //         drawText(image);
    //         imshow("Sample", image);
    //         if(waitKey(10) >= 0)
    //             break;
    //     }
    // }
    // else
    // {
    //     cout << "No capture" << endl;
    //     image = Mat::zeros(480, 640, CV_8UC1);
    //     drawText(image);
    //     imshow("Sample", image);
    //     waitKey(0);
    // }
    return 0;
}