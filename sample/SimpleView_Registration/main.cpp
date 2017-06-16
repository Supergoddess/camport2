#include "../common/common.hpp"

static char buffer[1024*1024*5];
static int  n;
static volatile bool exit_main;
static volatile bool save_frame;

struct CallbackData {
    int             index;
    TY_DEV_HANDLE   hDevice;
    DepthRender*    render;
    cv::Mat         colorM;
    cv::Mat         colorD;
};

void frameCallback(TY_FRAME_DATA* frame, void* userdata)
{
    CallbackData* pData = (CallbackData*) userdata;
    LOGD("=== Get frame %d", ++pData->index);

    cv::Mat depth, irl, irr, color, point3D;
    parseFrame(*frame, &depth, &irl, &irr, &color, &point3D);
    if(!depth.empty()){
        cv::Mat colorDepth = pData->render->Compute(depth);
        cv::imshow("ColorDepth", colorDepth);
    }
    if(!irl.empty()){ cv::imshow("LeftIR", irl); }
    if(!irr.empty()){ cv::imshow("RightIR", irr); }
    if(!color.empty()){
        if(!pData->colorM.empty()){
            cv::Mat u;
            cv::undistort(color, u, pData->colorM, pData->colorD, pData->colorM);
            // cv::fisheye::undistortImage(color, color, pData->colorM, pData->colorD, pData->colorM, color.size());
            color = u;
        }
        cv::Mat resizedColor;
        cv::resize(color, resizedColor, depth.size(), 0, 0, CV_INTER_LINEAR);
        cv::imshow("color", resizedColor);
    }

    // do Registration
    cv::Mat newDepth;
    if(!point3D.empty() && !color.empty()) {
        ASSERT_OK( TYRegisterWorldToColor(pData->hDevice, (TY_VECT_3F*)point3D.data, 0
                    , point3D.cols * point3D.rows, (uint16_t*)buffer, sizeof(buffer)
                    ));
        newDepth = cv::Mat(color.rows, color.cols, CV_16U, (uint16_t*)buffer);
        cv::Mat resized_color;
        cv::Mat temp;
        //you may want to use median filter to fill holes in projected depth image or do something else here
        cv::medianBlur(newDepth,temp,5);
        newDepth = temp;
        //resize to the same size for display
        cv::resize(newDepth, newDepth, depth.size(), 0, 0, 0);
        cv::resize(color, resized_color, depth.size());
        cv::Mat depthColor = pData->render->Compute(newDepth);
        depthColor = depthColor / 2 + resized_color / 2;
        cv::imshow("projected depth", depthColor);
    }

    if(save_frame){
        LOGD(">>>>>>>>>> write images");
        imwrite("/tmp/depth.png", newDepth);
        imwrite("/tmp/color.png", color);
        save_frame = false;
    }

    int key = cv::waitKey(1);
    switch(key){
        case -1:
            break;
        case 'q': case 1048576 + 'q':
            exit_main = true;
            break;
        case 's': case 1048576 + 's':
            save_frame = true;
            break;
        default:
            LOGD("Pressed key %d", key);
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
    if(!(allComps & TY_COMPONENT_RGB_CAM)){
        LOGE("=== Has no RGB camera, cant do registration");
    }

    LOGD("=== Configure components");
    int32_t componentIDs = TY_COMPONENT_POINT3D_CAM | TY_COMPONENT_RGB_CAM;
    ASSERT_OK( TYEnableComponents(hDevice, componentIDs) );

    LOGD("=== Prepare image buffer");
    int32_t frameSize;
    ASSERT_OK( TYGetFrameBufferSize(hDevice, &frameSize) );
    LOGD("     - Get size of framebuffer, %d", frameSize);

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
    ASSERT_OK( TYRegisterCallback(hDevice, frameCallback, &cb_data) );

    LOGD("=== Disable trigger mode");
    ASSERT_OK( TYSetBool(hDevice, TY_COMPONENT_DEVICE, TY_BOOL_TRIGGER_MODE, false) );

    LOGD("=== Start capture");
    ASSERT_OK( TYStartCapture(hDevice) );

    LOGD("=== Read color rectify matrix");
    {
        TY_CAMERA_DISTORTION color_dist;
        TY_CAMERA_INTRINSIC color_intri;
        TY_STATUS ret = TYGetStruct(hDevice, TY_COMPONENT_RGB_CAM, TY_STRUCT_CAM_DISTORTION, &color_dist, sizeof(color_dist));
        ret |= TYGetStruct(hDevice, TY_COMPONENT_RGB_CAM, TY_STRUCT_CAM_INTRINSIC, &color_intri, sizeof(color_intri));
        if (ret == TY_STATUS_OK)
        {
            cb_data.colorM.create(3, 3, CV_32FC1);
            cb_data.colorD.create(5, 1, CV_32FC1);
            memcpy(cb_data.colorM.data, color_intri.data, sizeof(color_intri.data));
            memcpy(cb_data.colorD.data, color_dist.data, sizeof(float)*cb_data.colorD.rows);
        }
        else
        {//let's try  to load from file...
            cv::FileStorage fs("color_intri.xml", cv::FileStorage::READ);
            if (fs.isOpened())
            {
                fs["M"] >> cb_data.colorM;
                fs["D"] >> cb_data.colorD;
                cb_data.colorD = cb_data.colorD.colRange(0, 8);
            }
        }
    }

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
