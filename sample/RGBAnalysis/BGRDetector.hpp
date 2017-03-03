#ifndef EP_BGRDetector_HPP_
#define EP_BGRDetector_HPP_


#include <opencv2/opencv.hpp>


class BGRDetector
{
public:
    BGRDetector(int bgCnt) {};

    bool needBg() {return false;};
    bool inputBg(cv::Mat& bgr) {return true;};

    bool judge(cv::Mat& bgr) {return true;};
};


#endif
