# OpenCV 3.4.0 설치
  
#### 1.기존 버전 제거
#### 2. 패키지 업그레이드
#### 3. OpenCV required 패키지 설치
#### 4. OpenCV 설치
#### 5. 설치 확인
  
<br/>

## 1. 기존 버전 삭제
기존에 OpenCV 3.4.0보다 낮은 버전이 설치되어 있다면 새로 설치하는 OpenCV 3.4.0 버전이 제대로 동작하지 않기 때문에 제거해야 합니다.

```sh
$ pkg-config --modversion opencv
Package opencv was not found in the pkg-config search path.
Perhaps you should add the directory containing `opencv.pc'
to the PKG_CONFIG_PATH environment variable
No package 'opencv' found
```
위와 같이 나온다면 기존에 OpenCV가 설치되지 않은 상태입니다.
기존 3.2버전이 설치되어 있는 경우 아래와 같이 버전이 표시됩니다.

```sh
$ pkg-config --modversion opencv
3.2.0.0
```

기존에 이미 설치되어 있는 경우 아래의 명령으로 OpenCV 패키지를 삭제합니다.

```sh
$ sudo apt-get purge  libopencv* python-opencv
$ sudo apt-get autoremove
```
  
<br/>

## 2. 패키지 업그레이드
기존에 설치된 패키지들을 업그레이드 해주는 작업입니다.  
Ubuntu Repository로부터 패키지 리스트를 업데이트합니다.
```sh
$ sudo apt-get update
```
  
기존에 설치된 패키지의 업데이트를 진행합니다.
```sh
$ sudo apt-get upgrade
```
  
<br/>

## 3. OpenCV required 패키지 설치
```sh
$ sudo apt-get install build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev libv4l-dev v4l-utils libgstreamer1.0-dev libgstreamlibqt4-dev er-plugins-base1.0-dev libqt4-dev mesa-utils libgl1-mesa-dri libqt4-opengl-dev libatlas-base-dev gfortran libeigen3-dev python2.7-dev python3-dev python-numpy python3-numpy qmake
```
  
<br/>

  - 위 패키지들에 관한 설명
  
```sh
build-essential cmake
```
build-essential 패키지에는 C/C++ 컴파일러와 관련 라이브러리, make 같은 도구들이 포함되어 있습니다.  
cmake는 컴파일 옵션이나 빌드된 라이브러리에 포함시킬 OpenCV 모듈 설정등을 위해 필요합니다.  
  
<br/>

```sh
pkg-config
```
pkg-config는 프로그램 컴파일 및 링크시 필요한 라이브러리에 대한 정보를 메타파일(확장자가 .pc 인 파일)로부터 가져오는데 사용됩니다.
  
<br/>

```sh
libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
```
특정 포맷의 이미지 파일을 불러오거나 기록하기 위해 필요한 패키지들입니다.  
  
<br/>

```sh
libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev
```
특정 코덱의 비디오 파일을 읽어오거나 기록하기 위해 필요한 패키지들입니다  
  
<br/>

```sh
libv4l-dev v4l-utils
```
Video4Linux 패키지는 리눅스에서 실시간 비디오 캡처를 지원하기 위한 디바이스 드라이버와 API를 포함하고 있습니다.  
  
<br/>

```sh
libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev 
```
OpenCV에서는 highgui 모듈을 사용하여 자체적으로 윈도우 생성하여 이미지나 비디오들을 보여줄 수 있습니다.  
  
<br/>

```sh
mesa-utils libgl1-mesa-dri libqt4-opengl-dev 
```
OpenGL 지원하기 위해 필요한 라이브러리입니다.  
  
<br/>

```sh
libatlas-base-dev gfortran libeigen3-dev
```
OpenCV 최적화를 위해 사용되는 라이브러리들입니다.
  
<br/>

```sh
python2.7-dev python3-dev python-numpy python3-numpy
```
python2.7-dev와 python3-dev 패키지는 OpenCV-Python 바인딩을 위해 필요한 패키지들입니다.  
Numpy는 매트릭스 연산등을 빠르게 처리할 수 있어서 OpenCV에서 사용됩니다.  
  
<br/>

```sh
libqt4-dev
```
윈도우 생성 등의 GUI를 위해 gtk 또는 qt를 선택해서 사용가능합니다.
  
  
<br/>

## 4. OpenCV 설치
소스코드를 설치할 디렉토리를 생성한 후 진행합니다.
```sh
~$ mkdir opencv
~$ cd openc
~/opencv$
```
  
OpenCV 3.4.0 소스를 다운로드 한 후 압축을 해제합니다.
```sh
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.0.zip
$ unzip opencv.zip
```
  
opencv_contrib 소스코드를 다운로드 받은 후 압축을 해제합니다.
```sh
$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.0.zip
$ unzip opencv_contrib.zip
```
  
opencv-3.4.0 에서 build 디렉토리를 생성하고 build 디렉토리로 이동합니다.
```sh
~/opencv$ cd opencv-3.4.0/
~/opencv/opencv-3.4.0$ mkdir build
~/opencv/opencv-3.4.0$ cd build
~/opencv/opencv-3.4.0/build$ 
```
  
cmake를 사용하여 OpenCV 컴파일 설정을 해줍니다.
```sh
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=OFF \
-D WITH_IPP=OFF \
-D WITH_1394=OFF \
-D BUILD_WITH_DEBUG_INFO=OFF \
-D BUILD_DOCS=OFF \
-D INSTALL_C_EXAMPLES=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D WITH_QT=ON \
-D WITH_OPENGL=ON \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.0/modules \
-D WITH_V4L=ON  \
-D WITH_FFMPEG=ON \
-D WITH_XINE=ON \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
../
```
  
다음과 같은 메세지가 보이면 정상적으로 설정된 것입니다.
```sh
-- Configuring done
-- Generating done
-- Build files have been written to: /home/<username>/opencv/opencv-3.4.0/build
```
  
cmake 실행 결과 문구입니다.
```sh
-- General configuration for OpenCV 3.4.0 =====================================
--   Version control:               unknown
-- 
--   Extra modules:
--     Location (extra):            /home/<username>/opencv/opencv_contrib-3.4.0/modules
--     Version control (extra):     unknown
-- 
--   Platform:
--     Timestamp:                   2017-12-27T06:32:42Z
--     Host:                        Linux 4.10.0-42-generic x86_64
--     CMake:                       3.5.1
--     CMake generator:             Unix Makefiles
--     CMake build tool:            /usr/bin/make
--     Configuration:               RELEASE
-- 
--   CPU/HW features:
--     Baseline:                    SSE SSE2 SSE3
--       requested:                 SSE3
--     Dispatched code generation:  SSE4_1 SSE4_2 FP16 AVX AVX2
--       requested:                 SSE4_1 SSE4_2 AVX FP16 AVX2
--       SSE4_1 (3 files):          + SSSE3 SSE4_1
--       SSE4_2 (1 files):          + SSSE3 SSE4_1 POPCNT SSE4_2
--       FP16 (1 files):            + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 AVX
--       AVX (5 files):             + SSSE3 SSE4_1 POPCNT SSE4_2 AVX
--       AVX2 (9 files):            + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 FMA3 AVX AVX2
-- 
--   C/C++:
--     Built as dynamic libs?:      YES
--     C++ Compiler:                /usr/bin/c++  (ver 5.4.0)
--     C++ flags (Release):         -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Winit-self -Wno-narrowing -Wno-delete-non-virtual-dtor -Wno-comment -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -O3 -DNDEBUG  -DNDEBUG
--     C++ flags (Debug):           -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Winit-self -Wno-narrowing -Wno-delete-non-virtual-dtor -Wno-comment -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -g  -O0 -DDEBUG -D_DEBUG
--     C Compiler:                  /usr/bin/cc
--     C flags (Release):           -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Winit-self -Wno-narrowing -Wno-comment -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -O3 -DNDEBUG  -DNDEBUG
--     C flags (Debug):             -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Winit-self -Wno-narrowing -Wno-comment -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -g  -O0 -DDEBUG -D_DEBUG
--     Linker flags (Release):
--     Linker flags (Debug):
--     ccache:                      NO
--     Precompiled headers:         YES
--     Extra dependencies:         dl m pthread rt /usr/lib/x86_64-linux-gnu/libGLU.so /usr/lib/x86_64-linux-gnu/libGL.so
--     3rdparty dependencies:
-- 
--   OpenCV modules:
--     To be built:                 aruco bgsegm bioinspired calib3d ccalib core datasets dnn dpm face features2d flann fuzzy highgui img_hash imgcodecs imgproc line_descriptor ml objdetect optflow phase_unwrapping photo plot python2 python3 python_bindings_generator reg rgbd saliency shape stereo stitching structured_light superres surface_matching text tracking video videoio videostab xfeatures2d ximgproc xobjdetect xphoto
--     Disabled:                    js world
--     Disabled by dependency:      -
--     Unavailable:                 cnn_3dobj cudaarithm cudabgsegm cudacodec cudafeatures2d cudafilters cudaimgproc cudalegacy cudaobjdetect cudaoptflow cudastereo cudawarping cudev cvv dnn_modern freetype hdf java matlab ovis sfm ts viz
--     Applications:                apps
--     Documentation:               NO
--     Non-free algorithms:         NO
-- 
--   GUI: 
--     QT:                          YES (ver 4.8.7 EDITION = OpenSource)
--       QT OpenGL support:         YES (/usr/lib/x86_64-linux-gnu/libQtOpenGL.so)
--     GTK+:                        NO
--     OpenGL support:             YES (/usr/lib/x86_64-linux-gnu/libGLU.so /usr/lib/x86_64-linux-gnu/libGL.so)
--     VTK support:                 NO
-- 
--   Media I/O: 
--     ZLib:                        /usr/lib/x86_64-linux-gnu/libz.so (ver 1.2.8)
--     JPEG:                        /usr/lib/x86_64-linux-gnu/libjpeg.so (ver )
--     WEBP:                        build (ver encoder: 0x020e)
--     PNG:                         /usr/lib/x86_64-linux-gnu/libpng.so (ver 1.2.54)
--     TIFF:                        /usr/lib/x86_64-linux-gnu/libtiff.so (ver 42 / 4.0.6)
--     JPEG 2000:                   /usr/lib/x86_64-linux-gnu/libjasper.so (ver 1.900.1)
--     OpenEXR:                     build (ver 1.7.1)
-- 
--   Video I/O:
--     FFMPEG:                      YES
--       avcodec:                   YES (ver 56.60.100)
--       avformat:                  YES (ver 56.40.101)
--       avutil:                    YES (ver 54.31.100)
--       swscale:                   YES (ver 3.1.101)
--       avresample:                NO
--     GStreamer:                   
--       base:                      YES (ver 1.8.3)
--       video:                     YES (ver 1.8.3)
--       app:                       YES (ver 1.8.3)
--       riff:                      YES (ver 1.8.3)
--       pbutils:                   YES (ver 1.8.3)
--     libv4l/libv4l2:              NO
--     v4l/v4l2:                    linux/videodev2.h
--     Xine:                        YES (ver 1.2.6)
--     gPhoto2:                     NO
-- 
--   Parallel framework:            pthreads
-- 
--   Trace:                         YES (with Intel ITT)
-- 
--   Other third-party libraries:
--     Lapack:                      NO
--     Eigen:                       YES (ver 3.2.92)
--     Custom HAL:                  NO
-- 
--   NVIDIA CUDA:                   NO
-- 
--   OpenCL:                        YES (no extra features)
--     Include path:                /home/<username>/opencv/opencv-3.4.0/3rdparty/include/opencl/1.2
--     Link libraries:              Dynamic load
-- 
--   Python 2:
--     Interpreter:                 /usr/bin/python2.7 (ver 2.7.12)
--     Libraries:                   /usr/lib/x86_64-linux-gnu/libpython2.7.so (ver 2.7.12)
--     numpy:                       /usr/lib/python2.7/dist-packages/numpy/core/include (ver 1.11.0)
--     packages path:               lib/python2.7/dist-packages
-- 
--   Python 3:
--     Interpreter:                 /usr/bin/python3 (ver 3.5.2)
--     Libraries:                   /usr/lib/x86_64-linux-gnu/libpython3.5m.so (ver 3.5.2)
--     numpy:                       /usr/lib/python3/dist-packages/numpy/core/include (ver 1.11.0)
--     packages path:               lib/python3.5/dist-packages
-- 
--   Python (for build):            /usr/bin/python2.7
-- 
--   Java:
--     ant:                         NO
--     JNI:                         NO
--     Java wrappers:               NO
--     Java tests:                  NO
-- 
--   Matlab:                        NO
-- 
--   Install to:                    /usr/local
-- -----------------------------------------------------------------
-- 
-- Configuring done
-- Generating done
-- Build files have been written to: /home/<username>/opencv/opencv-3.4.0/build
```
  
컴파일을 싱글코어로 진행할 경우 오래걸리기 때문에 사용중인 컴퓨터의 CPU 코어의 갯수를 확인합니다.
```sh
cat /proc/cpuinfo | grep processor | wc -l
6
```
  
make 명령을 사용하여 컴파일을 진행합니다.
```sh
~/opencv/opencv-3.4.0/build$ make -j4
```
  
컴파일 된 결과물을 설치합니다.
```sh
~/opencv/opencv-3.4.0/build$ sudo make install
```

/etc/ld.so.conf.d/ 디렉토리에 /usr/local/lib를 포함하는 설정파일이 있는지 확인합니다.
```sh
~/opencv/opencv-3.4.0/build$ cat /etc/ld.so.conf.d/*
/usr/lib/x86_64-linux-gnu/libfakeroot
# libc default configuration
/usr/local/lib
# Multiarch support
/lib/x86_64-linux-gnu
/usr/lib/x86_64-linux-gnu
/usr/lib/x86_64-linux-gnu/mesa-egl
/usr/lib/x86_64-linux-gnu/mesa
```

/usr/local/lib이 출력되지 않았다면 다음 명령을 추가로 실행해야합니다.
```sh
~/opencv/opencv-3.4.0/build$  sudo sh -c 'echo '/usr/local/lib' > /etc/ld.so.conf.d/opencv.conf'
```
  
/usr/local/lib을 찾은 경우나 못찾아서 추가한 작업을 한 경우 모두 컴파일시 opencv  라이브러리를 찾을 수 있도록 다음 명령을 실행합니다.
```sh
~/opencv/opencv-3.4.0/build$ sudo ldconfig
```
  
<br/>

## 5. 설치 확인
  
#### 1. C/C++
  
C/C++를 위해 OpenCV 라이브러리 사용가능 여부를 확인합니다.  
문제 없으면 설치된 OpenCV 라이브러리의 버전이 출력됩니다. 
```sh
~/opencv/opencv-3.4.0/build$ pkg-config --modversion opencv
3.4.0
```
  
아래처럼 opencv를 찾을 수 없다고 나오면  추가 작업이 필요합니다.
```sh
~/opencv/opencv-3.4.0/build$ pkg-config --modversion opencv
Package opencv was not found in the pkg-config search path.
Perhaps you should add the directory containing `opencv.pc'
to the PKG_CONFIG_PATH environment variable
No package 'opencv' found
```
  
pkg-config 명령이 /usr/local/lib/pkgconfig 경로에 있는 opencv.pc 파일을 찾을 수 있도록 해줘야 합니다. 
```sh
$ sudo sh -c 'echo PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig >> /etc/bash.bashrc'
$ sudo sh -c 'echo export PKG_CONFIG_PATH >> /etc/bash.bashrc'
$ source ~/.bashrc
```
  
<br/>

#### 2. Python
python에서는 import를 통해 확인합니다.
```sh
~/opencv/opencv-3.4.0/build$ python3
Python 3.5.2 (default, Sep 26 2018, 17:37:01) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
>>> cv2.__version__
'3.4.0'
>>> 
```
