#ifndef SAMPLE_COMMON_COMMON_HPP_
#define SAMPLE_COMMON_COMMON_HPP_

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include "TY_API.h"

#define ASSERT(x)   do{ \
                if(!(x)) { \
                    LOGD("Assert failed at %s:%d", __FILE__, __LINE__); \
                    LOGD("    : " #x ); \
                    abort(); \
                } \
            }while(0)

#define ASSERT_OK(x)    do{ \
                int err = (x); \
                if(err != TY_STATUS_OK) { \
                    LOGD("Assert failed: error %d(%s) at %s:%d", err, TYErrorString(err), __FILE__, __LINE__); \
                    LOGD("    : " #x ); \
                    abort(); \
                } \
            }while(0)


#define LOGD(fmt,...)  printf(fmt "\n", ##__VA_ARGS__)
#define LOGI(fmt,...)  printf(fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt,...)  printf("Error: " fmt "\n", ##__VA_ARGS__)
#define xLOGD(fmt,...)
#define xLOGI(fmt,...)
#define xLOGE(fmt,...)


#ifdef _WIN32
#  include <windows.h>
#  define MSLEEP(x)     Sleep(x)
   // windows defined macro max/min
#  ifdef max
#    undef max
#  endif
#  ifdef min
#    undef min
#  endif
#else
#  include <unistd.h>
#  include <sys/time.h>
#  define MSLEEP(x)     usleep((x)*1000)
#endif


#include "DepthRender.hpp"
#include "PointCloudViewer.hpp"

#endif
