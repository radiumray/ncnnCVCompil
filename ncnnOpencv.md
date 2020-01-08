# ncnn和opencv编译安装环境搭建

## 安装opencv:
https://www.cnblogs.com/raina/p/11365854.html


## CMakeLists.txt

    # 最低版本要求
    cmake_minimum_required(VERSION 3.4.1)

    project(ncnnOpencv)

    # 设置C++编译版本
    set(CMAKE_CXX_STANDARD 11)

    # ncnn项目所在路径，需要替换
    set(NCNN_DIR /home/ray/ncnn)

    # 分别设置ncnn的链接库和头文件
    set(NCNN_LIBS ${NCNN_DIR}/build/install/lib/libncnn.a)
    set(NCNN_INCLUDE_DIRS ${NCNN_DIR}/build/install/include/ncnn)

    # 配置OpenCV
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})

    include_directories(${NCNN_INCLUDE_DIRS})

    # 配置OpenMP
    FIND_PACKAGE( OpenMP REQUIRED)  
    if(OPENMP_FOUND)  
        message("OPENMP FOUND")  
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
    endif()  

    # 建立链接依赖
    add_executable(ncnnOpencv Main.cpp)
    target_link_libraries(ncnnOpencv ${NCNN_LIBS})
    target_link_libraries(ncnnOpencv ${OpenCV_LIBS})


## Main.cpp

    #include <iostream>
    #include <fstream>
    #include <algorithm>
    #include <opencv2/opencv.hpp>
    #include <opencv2/core.hpp>
    #include <opencv2/imgproc.hpp>
    #include <opencv2/highgui.hpp>
    #include <opencv2/videoio.hpp>

    #include <stdio.h>
    #include <vector>

    #include "platform.h"
    #include "net.h"
    #if NCNN_VULKAN
    #include "gpu.h"
    #endif // NCNN_VULKAN

    using namespace std;
    using namespace cv;

    struct Object
    {
        cv::Rect_<float> rect;
        int label;
        float prob;
    };

    static int detect_yolov3(const cv::Mat& bgr, std::vector<Object>& objects)
    {
        ncnn::Net yolov3;

    #if NCNN_VULKAN
        yolov3.opt.use_vulkan_compute = true;
    #endif // NCNN_VULKAN

        // original pretrained model from https://github.com/eric612/MobileNet-YOLO
        // param : https://drive.google.com/open?id=1V9oKHP6G6XvXZqhZbzNKL6FI_clRWdC-
        // bin : https://drive.google.com/open?id=1DBcuFCr-856z3FRQznWL_S5h-Aj3RawA
        // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
        yolov3.load_param("mobilenet_yolo.param");
        yolov3.load_model("mobilenet_yolo.bin");

        const int target_size = 352;

        int img_w = bgr.cols;
        int img_h = bgr.rows;

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};
        in.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = yolov3.create_extractor();
        ex.set_num_threads(4);

        ex.input("data", in);

        ncnn::Mat out;
        ex.extract("detection_out", out);

    //     printf("%d %d %d\n", out.w, out.h, out.c);
        objects.clear();
        for (int i=0; i<out.h; i++)
        {
            const float* values = out.row(i);

            Object object;
            object.label = values[0];
            object.prob = values[1];
            object.rect.x = values[2] * img_w;
            object.rect.y = values[3] * img_h;
            object.rect.width = values[4] * img_w - object.rect.x;
            object.rect.height = values[5] * img_h - object.rect.y;

            objects.push_back(object);
        }

        return 0;
    }

    static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
    {
        static const char* class_names[] = {"background",
            "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"};

        cv::Mat image = bgr.clone();

        for (size_t i = 0; i < objects.size(); i++)
        {
            const Object& obj = objects[i];

            fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                    obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

            cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

            char text[256];
            sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = obj.rect.x;
            int y = obj.rect.y - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > image.cols)
                x = image.cols - label_size.width;

            cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                        cv::Size(label_size.width, label_size.height + baseLine)),
                        cv::Scalar(255, 255, 255), -1);

            cv::putText(image, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        cv::imshow("image", image);
        // cv::waitKey(0);
    }

    void drawText(Mat & image)
    {
        putText(image, "Hello OpenCV",
                Point(20, 50),
                FONT_HERSHEY_COMPLEX, 1, // font face and scale
                Scalar(255, 255, 255), // white
                1, LINE_AA); // line thickness and type
    }

    int main()
    {

        Mat image;
        VideoCapture capture;
        capture.open(0);
        if(capture.isOpened())
        {
            cout << "Capture is opened" << endl;
            for(;;)
            {
                capture >> image;
                if(image.empty())
                    break;
                // drawText(image);

                std::vector<Object> objects;
                detect_yolov3(image, objects);

                draw_objects(image, objects);

                // imshow("Sample", image);
                if(waitKey(10) >= 0)
                    break;
            }
        }
        else
        {
            cout << "No capture" << endl;
            image = Mat::zeros(480, 640, CV_8UC1);
            drawText(image);
            imshow("Sample", image);
            waitKey(0);
        }

        return 0;
    }


## 执行cmake和make:

    mkdir build
    cd build
    cmake ..
    make

## c_cpp_properties.json

    {
        "configurations": [
            {
                "name": "Linux",
                "includePath": [
                    "${workspaceFolder}/**",
                    "/usr/local/include/opencv4",
                    "/home/ray/ncnn/build/install/include/ncnn"
                ],
                "defines": [],
                "compilerPath": "/usr/bin/gcc",
                "cStandard": "c11",
                "cppStandard": "c++17",
                "intelliSenseMode": "clang-x64"
            }
        ],
        "version": 4
    }

