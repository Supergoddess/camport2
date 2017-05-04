#include "../common/common.hpp"

static char buffer[1024*1024];
static int  n;
static volatile bool exit_main;


struct CallbackData {
    int             index;
    TY_DEV_HANDLE   hDevice;
    DepthRender*    render;
    bool            saveFrame;
    int             saveIdx;
};

void frameCallback(TY_FRAME_DATA* frame, void* userdata)
{
    CallbackData* pData = (CallbackData*) userdata;
    LOGD("=== Get frame %d", ++pData->index);

    cv::Mat depth, leftIR, rightIR, color;
    for( int i = 0; i < frame->validCount; i++ ){
        // get & show depth image
        if(frame->image[i].componentID == TY_COMPONENT_DEPTH_CAM){
            depth = cv::Mat(frame->image[i].height, frame->image[i].width
                    , CV_16U, frame->image[i].buffer);
            cv::Mat colorDepth = pData->render->Compute(depth);
            {
                std::stringstream ss;
                ss << "center depth: " << depth.at<uint16_t>(depth.rows/2, depth.cols/2);
                putText(colorDepth, ss.str(), cv::Point(0,25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 2);
            }
            cv::imshow("ColorDepth", colorDepth);
        }
        // get & show left ir image
        if(frame->image[i].componentID == TY_COMPONENT_IR_CAM_LEFT){
            leftIR = cv::Mat(frame->image[i].height, frame->image[i].width
                    , CV_8U, frame->image[i].buffer);
            cv::imshow("LeftIR", leftIR);
        }
        // get & show right ir image
        if(frame->image[i].componentID == TY_COMPONENT_IR_CAM_RIGHT){
            rightIR = cv::Mat(frame->image[i].height, frame->image[i].width
                    , CV_8U, frame->image[i].buffer);
            cv::imshow("RightIR", rightIR);
        }
        // get & show left ir image
        // get point3D
        if(frame->image[i].componentID == TY_COMPONENT_POINT3D_CAM){
            cv::Mat point3D(frame->image[i].height, frame->image[i].width
                    , CV_32FC3, frame->image[i].buffer);
        }
        // get & show RGB
        if(frame->image[i].componentID == TY_COMPONENT_RGB_CAM){
            if (frame->image[i].pixelFormat == TY_PIXEL_FORMAT_YUV422){
                LOGD("Color format TY_PIXEL_FORMAT_YUV422");
                cv::Mat yuv(frame->image[i].height, frame->image[i].width
                            , CV_8UC2, frame->image[i].buffer);
                cv::cvtColor(yuv, color, cv::COLOR_YUV2BGR_YVYU);
            } else {
                LOGD("Color format TY_PIXEL_FORMAT_RGB");
                color = cv::Mat(frame->image[i].height, frame->image[i].width
                        , CV_8UC3, frame->image[i].buffer);
                cv::cvtColor(color, color, cv::COLOR_RGB2BGR);
            }
            cv::imshow("bgr", color);
        }
    }

    int key = cv::waitKey(1);
    switch(key){
        case -1:
            break;
        case 'q': case 1048576 + 'q':
            exit_main = true;
            break;
        case 's': case 1048576 + 's':
            pData->saveFrame = true;
            break;
        default:
            LOGD("Pressed key %d", key);
    }

    if(pData->saveFrame && !depth.empty() && !leftIR.empty() && !rightIR.empty()){
        LOGI(">>>> save frame %d", pData->saveIdx);
        char f[32];
        sprintf(f, "%d.img", pData->saveIdx++);
        FILE* fp = fopen(f, "w");
        fwrite(depth.data, 2, depth.size().area(), fp);
        fwrite(color.data, 3, color.size().area(), fp);
        // fwrite(leftIR.data, 1, leftIR.size().area(), fp);
        // fwrite(rightIR.data, 1, rightIR.size().area(), fp);
        fclose(fp);

        pData->saveFrame = false;
    }

    LOGD("=== Callback: Re-enqueue buffer(%p, %d)", frame->userBuffer, frame->bufferSize);
    ASSERT_OK( TYEnqueueBuffer(pData->hDevice, frame->userBuffer, frame->bufferSize) );
}

int main(int argc, char* argv[])
{
    const char* IP = NULL;
    const char* ID = NULL;
    TY_DEV_HANDLE hDevice;

    for(int i = 1; i < argc; i++){
        if(strcmp(argv[i], "-id") == 0){
            ID = argv[++i];
        }else if(strcmp(argv[i], "-ip") == 0){
            IP = argv[++i];
        }else if(strcmp(argv[i], "-h") == 0){
            LOGI("Usage: SimpleView_Callback [-h] [-ip <IP>]");
            return 0;
        }
    }
    
    LOGD("=== Init lib");
    ASSERT_OK( TYInitLib() );
    TY_VERSION_INFO* pVer = (TY_VERSION_INFO*)buffer;
    ASSERT_OK( TYLibVersion(pVer) );
    LOGD("     - lib version: %d.%d.%d", pVer->major, pVer->minor, pVer->patch);

    if(IP) {
        LOGD("=== Open device %s", IP);
        ASSERT_OK( TYOpenDeviceWithIP(IP, &hDevice) );
    } else {
        if(ID == NULL){
            LOGD("=== Get device info");
            ASSERT_OK( TYGetDeviceNumber(&n) );
            LOGD("     - device number %d", n);

            TY_DEVICE_BASE_INFO* pBaseInfo = (TY_DEVICE_BASE_INFO*)buffer;
            ASSERT_OK( TYGetDeviceList(pBaseInfo, 100, &n) );

            if(n == 0){
                LOGD("=== No device got");
                return -1;
            }
            ID = pBaseInfo[0].id;
        }

        LOGD("=== Open device: %s", ID);
        ASSERT_OK( TYOpenDevice(ID, &hDevice) );
    }

    int32_t allComps;
    ASSERT_OK( TYGetComponentIDs(hDevice, &allComps) );
    if(allComps & TY_COMPONENT_RGB_CAM){
        LOGD("=== Has RGB camera, open RGB cam");
        ASSERT_OK( TYEnableComponents(hDevice, TY_COMPONENT_RGB_CAM) );
    }

    LOGD("=== Configure components, open depth cam");
    int32_t componentIDs = TY_COMPONENT_DEPTH_CAM | TY_COMPONENT_IR_CAM_LEFT | TY_COMPONENT_IR_CAM_RIGHT;
    // int32_t componentIDs = TY_COMPONENT_DEPTH_CAM;
    // int32_t componentIDs = TY_COMPONENT_DEPTH_CAM | TY_COMPONENT_IR_CAM_LEFT;
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
    CallbackData cb_data;
    cb_data.index = 0;
    cb_data.hDevice = hDevice;
    cb_data.render = &render;
    cb_data.saveFrame = false;
    cb_data.saveIdx = 0;
    ASSERT_OK( TYRegisterCallback(hDevice, frameCallback, &cb_data) );

    LOGD("=== Disable trigger mode");
    ASSERT_OK( TYSetBool(hDevice, TY_COMPONENT_DEVICE, TY_BOOL_TRIGGER_MODE, false) );

    LOGD("=== Start capture");
    ASSERT_OK( TYStartCapture(hDevice) );

    LOGD("=== Wait for callback");
    exit_main = false;
    while(!exit_main){
        MSLEEP(100);
    }

    ASSERT_OK( TYStopCapture(hDevice) );
    ASSERT_OK( TYCloseDevice(hDevice) );
    ASSERT_OK( TYDeinitLib() );
    delete frameBuffer[0];
    delete frameBuffer[1];

    LOGD("=== Main done!");
    return 0;
}
