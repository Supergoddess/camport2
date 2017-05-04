#include "../common/common.hpp"
#include "BGRDetector.hpp"


using namespace cv;


static char buffer[1024*1024];
static int  n;
static volatile bool exit_main;


struct CallbackData {
    int             index;
    TY_DEV_HANDLE   hDevice;
    DepthRender*    render;
    BGRDetector*    bgrDetector;
};


void fixColor(Mat& color)
{
    for(int i = 0; i < color.rows; i++){
        // remove g
        uint8_t* p = &color.at<uint8_t>(i, 1);
        uint8_t min = 0xff;
        for(int j = 0; j <color.cols; j++){
            if(p[j*3] < min){
                min = p[j*3];
            }
        }
        if(min){
            for(int j = 0; j <color.cols; j++){
                p[j*3] -= min;
            }
        }
    }
}


void drawHist(cv::Mat& src)
{
    /// Separate the image in 3 places ( B, G and R )
    std::vector<Mat> bgr_planes;
    split( src, bgr_planes );

    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat b_hist, g_hist, r_hist;

    /// Compute the histograms:
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar(0,0,0) );

    double bminVal = 0, bmaxVal = 0;
    minMaxIdx(b_hist, &bminVal, &bmaxVal);
    double gminVal = 0, gmaxVal = 0;
    minMaxIdx(g_hist, &gminVal, &gmaxVal);
    double rminVal = 0, rmaxVal = 0;
    minMaxIdx(r_hist, &rminVal, &rmaxVal);
    // float wmax = max(max(bmaxVal, gmaxVal), rmaxVal);
    float wmax = cv::sum(g_hist).val[0] / 256 * 2;
    b_hist *= hist_h / wmax;
    g_hist *= hist_h / wmax;
    r_hist *= hist_h / wmax;

    std::vector<cv::Mat> hist_image_rgb;
    hist_image_rgb.resize(3);
    for (int idx = 0; idx < 3; idx++){
        hist_image_rgb[idx].create(hist_h, hist_w, CV_8UC1);
        hist_image_rgb[idx].setTo(0);
    }
    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        if(cvRound(g_hist.at<float>(i)) > 0){
            line( hist_image_rgb[1], Point( bin_w*(i), hist_h) ,
                         Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                         Scalar::all(0xff), 1, 4, 0  );
            line( hist_image_rgb[0], Point( bin_w*(i), hist_h) ,
                         Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                         Scalar::all(0xff), 1, 4, 0  );

            line( hist_image_rgb[2], Point( bin_w*(i), hist_h) ,
                         Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                         Scalar::all(0xff), 1, 4, 0  );
        }
    }

    merge(hist_image_rgb, histImage);
    /// Display
    namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
    imshow("calcHist Demo", histImage );
}


void frameCallback(TY_FRAME_DATA* frame, void* userdata)
{
    CallbackData* pData = (CallbackData*) userdata;
    LOGD("=== Get frame %d", ++pData->index);

    cv::Mat depth;
    cv::Mat color;
    for( int i = 0; i < frame->validCount; i++ ){
        // get & show depth image
        if(frame->image[i].componentID == TY_COMPONENT_DEPTH_CAM){
            depth = cv::Mat(frame->image[i].height, frame->image[i].width
                    , CV_16U, frame->image[i].buffer);
        }
        // get & show RGB
        if(frame->image[i].componentID == TY_COMPONENT_RGB_CAM){
            cv::Mat bgr;
            if (frame->image[i].pixelFormat == TY_PIXEL_FORMAT_YUV422){
                cv::Mat yuv(frame->image[i].height, frame->image[i].width
                            , CV_8UC2, frame->image[i].buffer);
                cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_YVYU);
            }
            else{
                cv::Mat rgb(frame->image[i].height, frame->image[i].width
                            , CV_8UC2, frame->image[i].buffer);
                cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
            }
            color = bgr;
        }
    }

    if(pData->bgrDetector->needBg()){
        pData->bgrDetector->inputBg(color);
    } else {
        bool ret = pData->bgrDetector->judge(color);
        LOGI("************************ color ret: %d\n", ret);
    }

    if(!depth.empty()) {
        // imshow("depth", depth);
        cv::Mat colorDepth = pData->render->Compute(depth);
        cv::imshow("ColorDepth", colorDepth);
    }
    if(!color.empty()) {
        // fixColor(color);
        imshow("bgr", color);
        drawHist(color);
    }

    int key = cv::waitKey(1);
    switch(key){
        case -1:
            break;
        case 'q': case 1048576 + 'q':
            exit_main = true;
            break;
        default:
            LOGD("Pressed key %d", key);
    }

    LOGD("=== Callback: Re-enqueue buffer(%p, %d)", frame->userBuffer, frame->bufferSize);
    ASSERT_OK( TYEnqueueBuffer(pData->hDevice, frame->userBuffer, frame->bufferSize) );
}

int main()
{
    LOGD("=== Init lib");
    ASSERT_OK( TYInitLib() );
    TY_VERSION_INFO* pVer = (TY_VERSION_INFO*)buffer;
    ASSERT_OK( TYLibVersion(pVer) );
    LOGD("     - lib version: %d.%d.%d", pVer->major, pVer->minor, pVer->patch);

    LOGD("=== Get device info");
    ASSERT_OK( TYGetDeviceNumber(&n) );
    LOGD("     - device number %d", n);

    TY_DEVICE_BASE_INFO* pBaseInfo = (TY_DEVICE_BASE_INFO*)buffer;
    ASSERT_OK( TYGetDeviceList(pBaseInfo, 100, &n) );

    if(n == 0){
        LOGD("=== No device got");
        return -1;
    }

    LOGD("=== Open device 0");
    TY_DEV_HANDLE hDevice;
    ASSERT_OK( TYOpenDevice(pBaseInfo[0].id, &hDevice) );

    int32_t allComps;
    ASSERT_OK( TYGetComponentIDs(hDevice, &allComps) );
    if(allComps & TY_COMPONENT_RGB_CAM){
        LOGD("=== Has RGB camera, open RGB cam");
        ASSERT_OK( TYEnableComponents(hDevice, TY_COMPONENT_RGB_CAM) );
    }

    LOGD("=== Configure components, open depth cam");
    int32_t componentIDs = TY_COMPONENT_DEPTH_CAM;
    ASSERT_OK( TYEnableComponents(hDevice, componentIDs) );

    LOGD("=== Configure feature, set resolution to 640x480.");
    LOGD("Note: DM460 resolution feature is in component TY_COMPONENT_DEVICE,");
    LOGD("      other device may lays in some other components.");
    int err = TYSetEnum(hDevice, TY_COMPONENT_DEPTH_CAM, TY_ENUM_IMAGE_MODE, TY_IMAGE_MODE_640x480);
    ASSERT(err == TY_STATUS_OK || err == TY_STATUS_NOT_PERMITTED);

    LOGD("=== Prepare image buffer");
    int32_t frameSize;
    ASSERT_OK( TYGetFrameBufferSize(hDevice, &frameSize) );
    LOGD("     - Get size of framebuffer, %d", frameSize);
    ASSERT( frameSize >= 640*480*2 );

    LOGD("     - Allocate & enqueue buffers");
    char* frameBuffer[2];
    frameBuffer[0] = new char[frameSize];
    frameBuffer[1] = new char[frameSize];
    LOGD("     - Enqueue buffer (%p, %d)", frameBuffer[0], frameSize);
    ASSERT_OK( TYEnqueueBuffer(hDevice, frameBuffer[0], frameSize) );
    LOGD("     - Enqueue buffer (%p, %d)", frameBuffer[1], frameSize);
    ASSERT_OK( TYEnqueueBuffer(hDevice, frameBuffer[1], frameSize) );

    LOGD("=== Register callback");
    LOGD("Note: Callback may block internal data receiving,");
    LOGD("      so that user should not do long time work in callback.");
    LOGD("      To avoid copying data, we pop the framebuffer from buffer queue and");
    LOGD("      give it back to user, user should call TYEnqueueBuffer to re-enqueue it.");
    DepthRender render;
    BGRDetector bgrDetector(5);
    CallbackData cb_data;
    cb_data.index = 0;
    cb_data.hDevice = hDevice;
    cb_data.render = &render;
    cb_data.bgrDetector = &bgrDetector;
    // ASSERT_OK( TYRegisterCallback(hDevice, frameCallback, &cb_data) );

    LOGD("=== Disable trigger mode");
    ASSERT_OK( TYSetBool(hDevice, TY_COMPONENT_DEVICE, TY_BOOL_TRIGGER_MODE, false) );

    LOGD("=== Start capture");
    ASSERT_OK( TYStartCapture(hDevice) );

    LOGD("=== While loop to fetch frame");
    exit_main = false;
    TY_FRAME_DATA frame;

    int index = -1;
    while(!exit_main){
#if 0
        index = (index+1) % 16;
        std::ostringstream filename;
        filename << "/home/tyrael/Desktop/nanjing-pic/" << index << ".jpg";
        Mat sampleImage = imread(filename.str().c_str());
        imshow("sample", sampleImage);
        drawHist(sampleImage);
        waitKey();
#else
        int err = TYFetchFrame(hDevice, &frame, -1);
        if( err != TY_STATUS_OK ){
            LOGD("... Drop one frame");
            continue;
        }

        frameCallback(&frame, &cb_data);
#endif
    }

    ASSERT_OK( TYStopCapture(hDevice) );
    ASSERT_OK( TYCloseDevice(hDevice) );
    ASSERT_OK( TYDeinitLib() );
    // MSLEEP(10); // sleep to ensure buffer is not used any more
    delete frameBuffer[0];
    delete frameBuffer[1];

    LOGD("=== Main done!");
    return 0;
}
