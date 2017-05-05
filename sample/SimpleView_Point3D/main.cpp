#include <limits>
#include <cassert>
#include <cmath>
#include "../common/common.hpp"

static char buffer[1024*1024];
static int  n;
static volatile bool exit_main;

bool hasColor = false;
static TY_CAMERA_INTRINSIC m_colorIntrinsic;


struct CallbackData {
    int             index;
    TY_DEV_HANDLE   hDevice;
    DepthRender*    render;
    PointCloudViewer* pcviewer;
};

cv::Point3f depthToWorld(float* intr, int x, int y, int z)
{
    cv::Point3f world;
    world.x = (x - intr[2]) * z / intr[0];
    world.y = (y - intr[5]) * z / intr[4];
    world.z = z;

    return world;
}


void frameHandler(TY_FRAME_DATA* frame, void* userdata)
{
    CallbackData* pData = (CallbackData*) userdata;
    LOGD("=== Get frame %d", ++pData->index);

    cv::Mat point3D, color;
    for( int i = 0; i < frame->validCount; i++ ){
#if 0
        // get & show depth image
        if(frame->image[i].componentID == TY_COMPONENT_DEPTH_CAM){
            cv::Mat depth(frame->image[i].height, frame->image[i].width
                    , CV_16U, frame->image[i].buffer);
            cv::Mat colorDepth = pData->render->Compute(depth);
            cv::imshow("ColorDepth", colorDepth);
        }
        // get & show left ir image
        if(frame->image[i].componentID == TY_COMPONENT_IR_CAM_LEFT){
            cv::Mat leftIR(frame->image[i].height, frame->image[i].width
                    , CV_8U, frame->image[i].buffer);
            cv::imshow("LeftIR", leftIR);
        }
        // get & show right ir image
        if(frame->image[i].componentID == TY_COMPONENT_IR_CAM_RIGHT){
            cv::Mat rightIR(frame->image[i].height, frame->image[i].width
                    , CV_8U, frame->image[i].buffer);
            cv::imshow("RightIR", rightIR);
        }
#endif
        // get point3D
        if(frame->image[i].componentID == TY_COMPONENT_POINT3D_CAM){
            point3D = cv::Mat(frame->image[i].height, frame->image[i].width
                    , CV_32FC3, frame->image[i].buffer);
        }
        // get & show RGB
        if(frame->image[i].componentID == TY_COMPONENT_RGB_CAM){
            if (frame->image[i].pixelFormat == TY_PIXEL_FORMAT_YUV422){
                cv::Mat yuv(frame->image[i].height, frame->image[i].width
                            , CV_8UC2, frame->image[i].buffer);
                cv::cvtColor(yuv, color, cv::COLOR_YUV2BGR_YVYU);
            } else {
                color = cv::Mat(frame->image[i].height, frame->image[i].width
                        , CV_8UC3, frame->image[i].buffer);
                cv::cvtColor(color, color, cv::COLOR_RGB2BGR);
            }
            cv::imshow("color", color);
        }
    }

    if(!point3D.empty()){
        pData->pcviewer->show(point3D, "Point3D");
        if(pData->pcviewer->isStopped("Point3D")){
            exit_main = true;
            return;
        }
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

int main(int argc, char* argv[])
{
    const char* IP = NULL;
    const char* ID = NULL;
    const char* file = NULL;
    bool save3d = false;
    bool ir2jpg = false;
    TY_DEV_HANDLE hDevice;

    for(int i = 1; i < argc; i++){
        if(strcmp(argv[i], "-id") == 0){
            ID = argv[++i];
        }else if(strcmp(argv[i], "-ip") == 0){
            IP = argv[++i];
        }else if(strcmp(argv[i], "-h") == 0){
            LOGI("Usage: SimpleView_Callback [-ir] [-t] [-f <file>] [-h] [-ip <IP>] [-id <ID>]");
            return 0;
        } else if(!strcmp(argv[i], "-f")){
            file = argv[++i];
        } else if(!strcmp(argv[i], "-t")){
            save3d = true;
        } else if(!strcmp(argv[i], "-ir")){
            ir2jpg = true;
        }
    }

    if(file){
        FILE* fp = fopen(file, "r");
        cv::Mat depth(480, 640, CV_16U);
        cv::Mat irl(960, 1280, CV_8U);
        cv::Mat irr(960, 1280, CV_8U);
        fread(depth.data, 2, depth.size().area(), fp);
        fread(irl.data, 1, irl.size().area(), fp);
        fread(irr.data, 1, irr.size().area(), fp);
        // cv::imshow("depth", depth*64);
        fclose(fp);

        if(ir2jpg && !irl.empty() && !irr.empty()){
            char f[64];
            sprintf(f, "%s-irl.jpg", file);
            cv::imwrite(f, irl);
            sprintf(f, "%s-irr.jpg", file);
            cv::imwrite(f, irr);
        }

        FILE* fp3d = NULL;
        if(save3d){
            char f3d[64];
            sprintf(f3d, "%s-p3d.txt", file);
            fp3d = fopen(f3d, "w");
        }

        const float intrinsic[] = {
            1157.47473,        0.0, 627.822876,
                   0.0, 1157.47473, 464.911407,
                   0.0,        0.0,        1.0,
        };
        float inv_fx = 1.0 / intrinsic[0];
        float inv_fy = 1.0 / intrinsic[4];
        cv::Mat point3D = cv::Mat::zeros(480, 640, CV_32FC3);
        cv::Point3f* p = (cv::Point3f*)point3D.data;
        for(int r = 0; r < 480; r++){
        for(int c = 0; c < 640; c++){
            uint16_t v = depth.at<uint16_t>(r, c);
            if(v > 0){
                point3D.at<cv::Point3f>(r, c).x = (c * 2 - intrinsic[2]) * v * inv_fx;
                point3D.at<cv::Point3f>(r, c).y = (r * 2 - intrinsic[5]) * v * inv_fy;
                point3D.at<cv::Point3f>(r, c).z = v;

                if(fp3d){
                    fprintf(fp3d, "%f %f %f\n", point3D.at<cv::Point3f>(r, c).x, point3D.at<cv::Point3f>(r, c).y, point3D.at<cv::Point3f>(r, c).z);
                }
            }
        }
        }

        if(fp3d){
            fclose(fp3d);
        }

        PointCloudViewer pcviewer;
        pcviewer.show(point3D, "Point3D");
        while(1){
            if(pcviewer.isStopped("Point3D")){
                LOGI("Exit");
                return 0;
            }
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
    } else if (ID){
        LOGD("=== Open device %s", ID);
        ASSERT_OK( TYOpenDevice(ID, &hDevice) );
    } else {
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
        ASSERT_OK( TYOpenDevice(pBaseInfo[0].id, &hDevice) );
    }

    LOGD("=== Configure components, open point3d cam");
    // int32_t componentIDs = TY_COMPONENT_POINT3D_CAM;
    int32_t componentIDs = TY_COMPONENT_POINT3D_CAM | TY_COMPONENT_RGB_CAM;
    ASSERT_OK( TYEnableComponents(hDevice, componentIDs) );

    int err = TYGetStruct(hDevice, TY_COMPONENT_RGB_CAM, TY_STRUCT_CAM_INTRINSIC, (void*)&m_colorIntrinsic, sizeof(m_colorIntrinsic));
    if(err != TY_STATUS_OK){ 
        LOGE("Get camera RGB intrinsic failed: %s", TYErrorString(err));
    } else {
        hasColor = true;
    }

    LOGD("=== Configure feature, set resolution to 640x480.");
    LOGD("Note: DM460 resolution feature is in component TY_COMPONENT_DEVICE,");
    LOGD("      other device may lays in some other components.");
    err = TYSetEnum(hDevice, TY_COMPONENT_DEPTH_CAM, TY_ENUM_IMAGE_MODE, TY_IMAGE_MODE_640x480);
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
    PointCloudViewer pcviewer;
    CallbackData cb_data;
    cb_data.index = 0;
    cb_data.hDevice = hDevice;
    cb_data.render = &render;
    cb_data.pcviewer = &pcviewer;
    // ASSERT_OK( TYRegisterCallback(hDevice, frameHandler, &cb_data) );

    LOGD("=== Disable trigger mode");
    ASSERT_OK( TYSetBool(hDevice, TY_COMPONENT_DEVICE, TY_BOOL_TRIGGER_MODE, false) );

    LOGD("=== Start capture");
    ASSERT_OK( TYStartCapture(hDevice) );

    LOGD("=== While loop to fetch frame");
    exit_main = false;
    TY_FRAME_DATA frame;

    while(!exit_main){
        int err = TYFetchFrame(hDevice, &frame, -1);
        if( err != TY_STATUS_OK ){
            LOGD("... Drop one frame");
            continue;
        }

        frameHandler(&frame, &cb_data);
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

