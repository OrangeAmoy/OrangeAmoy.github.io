---
title: 网页播放视频流的去插件化实现
date: 2019-04-02 18:00:00
tags:
categories:
- 技术文档
- 视频应用
---

### 1. 项目需求  
开发一套服务端的全景拼接系统，需具备如下功能：  
1. 支持获取前端网络摄像头视频流，并进行解码。  
2. 在本机对多路视频流进行实时图像拼接，拼接成全景视频流。  
3. 进行视频编码，并提供全景视频流服务，便于后端访问与显示。  
4. 提供一个免插件的视频流实时显示服务。  

### 2. 技术架构  
```
                        GStreamer + DeepStream
视频源 -> 视频流获取 -> 视频流解码 -> 视频分析(拼接) -> RTSP视频流服务

                        Kurento + WebRTC
RTSP视频流服务 -> H.264 MJPEG解码 -> VP8编码 -> WebRTC -> 浏览器
```

<!-- more -->

### 3. 技术简介  
[GStreamer](https://gstreamer.freedesktop.org/)  
&emsp;&emsp;GStreamer是用来构建流媒体应用的开源多媒体框架，其目标是简化音频/视频应用程序开发。GStreamer采用了基于插件(plugin)和管道(pipeline)的体系结构，框架通过一个数据管道实现多媒体应用程序，根据实际需求将一个个的插件安装在管道中，所有插件通过管道机制进行统一的数据交换，每个插件都是独立的，不与其他插件耦合，可以方便的安装与卸载。其特点之一是动态管线技术，能够根据实际数据流情况动态地调整管线内插件的组装方式，灵活性大，使用户不需要关心底层的数据流动，更专注于流媒体应用的开发。  
&emsp;&emsp;在本项目中，借助GStreamer实现视频流的获取与管理。特别在提供视频流服务部分，借助gst-rtsp-server将处理完的视频以RTSP的形式对外提供。  
&emsp;&emsp;参考资料：  
&emsp;&emsp;[用 GStreamer 简化 Linux 多媒体开发](https://www.cnblogs.com/yxling/p/6599065.html)  

[DeepStream](https://developer.nvidia.com/deepstream-sdk)  
&emsp;&emsp;DeepStream是[NVIDIA](https://www.nvidia.cn/)推出的视频分析应用框架，借助DeepStream SDK，开发人员能够充分挖掘NVIDIA Tesla GPU的硬件特性，快速构建高效，高性能的视频分析应用程序。更重要的是，随着深度学习的发展，在视频图像领域，卷积神经网络已经能够处理大部分的任务，诸如：目标检测、目标识别、目标跟踪等。DeepStream的神经网络推理模块利用了自家的[TensorRT](https://developer.nvidia.com/tensorrt)技术，进一步利用GPU特性，提高了神经网络推理速度。  
&emsp;&emsp;在以往的视频分析应用程序开发过程中，开发者往往需要考虑：视频流的获取、视频编解码、神经网络推理实现、检测目标的跟踪、目标在视频上的显示以及视频流显示等。为了实现这些功能，可能需要同时借助多种技术，典型的有：[OpenCV](https://opencv.org/) + [Caffe](http://caffe.berkeleyvision.org/)， 且需要开发者自行实现数据在CPU和GPU之间的搬运。DeepStream本质是基于GStreamer进行开发，基于插件和管道的体系结构，实现了以上视频分析应用程序开发中需要考虑的功能。主要的区别在于充分利用了自家的显卡资源与技术优势，重新对视频分析应用的各个环节进行整合，统一接口，形成DeepStream SDK，帮助开发者快速部署视频分析应用程序。当开发者遇到DeepStream本身不支持的神经网络层或者功能时，可自行根据官方提供的插件模板实现自定义功能。  
&emsp;&emsp;在本项目中，借助DeepStream实现视频流的获取，视频流管理，视频编解码，神经网络的推理(可选)，视频拼接(基于插件实现)。  
&emsp;&emsp;参考资料：  
&emsp;&emsp;[DeepStream: Next-Generation Video Analytics for Smart Cities](https://devblogs.nvidia.com/deepstream-video-analytics-smart-cities/)  

[WebRTC](https://webrtc.org/)  
&emsp;&emsp;WebRTC全称 Web Real-Time Communication。它并不是单一的协议，包含了媒体、加密、传输层等在内的多个协议标准以及一套基于JavaScript的API。通过简单易用的JavaScript API，在不安装任何插件的情况下，让浏览器拥有了P2P音视频和数据分享的能力。简单理解就是借助WebRTC，让浏览器可以在不安装任何插件的情况下播放音视频。  
&emsp;&emsp;在本项目中，借助WebRTC实现浏览器播放视频的功能，且不需要安装插件。  
&emsp;&emsp;参考资料：  
&emsp;&emsp;[知乎-又拍云](https://www.zhihu.com/question/22301898/answer/207430200)
    
[Kurento](https://doc-kurento.readthedocs.io/en/6.9.0/index.html#)  
&emsp;&emsp;Kurento是一个WebRTC媒体服务器，并且包含一个客户端API集合，用以简化WWW和移动平台上的高级视频应用程序的开发。Kurento的功能包括组通信，转码，记录，混音，广播和routing of audiovisual flows。Kurento同样提供高级的媒体处理能力，包括计算机视觉，视频检索，虚拟现实和语音分析。Kurento模块化的架构使得其集成第三方媒体处理算法(如语音识别，场景分析，人脸识别等)很简单，而且它可以被应用程序开发者视为透明。Kurento的核心组成是Kurento媒体服务器，它用来负责媒体传输，处理，加载和记录。它是基于GStreamer，优化了资源消耗来实现的。它提供的功能有：网络流协议，包括HTTP(作为客户端和服务端工作)，RTP和WebRTC；组通信(MCU和SFU功能)，支持媒体混合和媒体路由/分发;原生支持计算机视觉和虚拟现实滤镜；媒体存储，支持WebM和MP4的写操作，能播放GStreamer支持的所有格式；自动的媒体转换，支持GStreamer提供的所有codec，包括VP8, H.264, H.263, , AMR, OPUS, Speex, G.711,等。  
&emsp;&emsp;在本项目中，借助Kurento实现RTSP视频流从服务端到浏览器端的转发与转码。  
&emsp;&emsp;参考资料：  
&emsp;&emsp;[Kurento应用开发指南](https://blog.csdn.net/liuweihui521/article/details/79885324)

### 4. 技术实现  
#### 4.1 项目背景  
&emsp;&emsp;在2017年5月的时候，我们针对公司的全景拼接相机一代产品进行了优化，那时候刚开始接触的全景拼接。全景一代产品基于PC端开发，以客户端的形式展现给客户，客户端中包含了相机参数配置、视频流获取、拼接算法、全景视频展示等。目前看来有几个弊端：第一，需要将相机配置文件存于全景相机中，客户端读取配置文件并初始化，较为繁琐。第二，需要单独安装客户端，拼接算法受限于客户端的硬件性能，产品性能无法保证。第三，需要根据不同系统开发不同的客户端，维护较为麻烦。  
&emsp;&emsp;在这之后，我们开始了全景二代产品的研发工作。在全景拼接算法方面，精简了代码，提高了拼接质量与拼接速度，具体优化内容后续单独说明。除了算法的优化，更多的是从整个系统层面的优化。首先将全景拼接系统移植到[Nvidia Jetson TX2](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems-dev-kits-modules/)上，且最终以web页面的方式提供给客户。通过这种改变，一个是固定了开发环境与硬件，能够提供稳定的全景拼接服务，另一个是客户直接通过web页面查看全景视频流，简单方便。但是，虽然不需要安装客户端，但是需要web插件，因为无法直接通过web页面显示全景视频流。当时的主流全景厂商，提供全景视频流的展示方式无非是客户端与web插件。经过后期的实践证明，web插件依然相对繁琐，且存在问题。  
&emsp;&emsp;近阶段，重新启动全景拼接项目，但近期的工作主要考虑的是从系统层面去重新审视原来的工作，最主要的改变就是去插件化的全景视频流web页面显示。因此，主要的工作是进行原理验证，验证方法的可行性。通过阶段性的调研与摸索，目前选中了基于GStreamer+DeepStream+Kurento+WebRTC技术来实现本项目。选用GStreamer主要用来进行前端网络摄像头视频流的获取、管理与分发。选用DeepStream主要借助Nvidia的相关技术栈，完成多路视频流同步、全景拼接算法以及后续的集成视频结构化算法。选用Kurento充当流媒体服务器，用于实现RTSP视频流到网页端视频的转码及推送。选用WebRTC主要用来实现去插件化的全景视频流页面展示。基于以上技术，能够简单的构建一个去插件话的全景拼接系统。  
#### 4.2 流媒体服务器  
&emsp;&emsp;前面提到WebRTC能够通过JavaScript API实现网页浏览器间的实时通信，而不用通过任何类型的媒体中继，如图1所示：  

<div align=center>
<img src=https://s2.ax1x.com/2019/04/02/A6rsN6.png width=450/>

图1
</div>

&emsp;&emsp;虽然这种点对点的方式能够足以实现一些基本应用，但是诸如：组通信、流媒体录制、媒体广播或者媒体转码是难以实现的，还是需要有媒体服务器支持。  
&emsp;&emsp;在概念上，WebRTC媒体服务器是一种多媒体中继(它位于两个通信端的中间)。媒体服务器能处理媒体流，并有各种功能，包括组通信(分发一个端生成的媒体流到多个接收端，如像Multi-Conference Unit, MCU的工作方式)，混合(转换多个输入流合成一个组合流)，转码(在不兼容的客户端间选择codec和格式)， 录制等。  
&emsp;&emsp;Kurento架构的核心是媒体服务器，它被命名为Kurento媒体服务器(KMS)。Kurento媒体服务器是基于GStreamer开发的，与GStreamer一样是插件式的，所有的功能都是插件模块，可以被激活与关闭。Kurento媒体服务器能够提供即时可用的组通信，混合，转码，录制和播放等功能。另外，Kurento媒体服务器还提供一些高级的媒体处理模块，包括有计算机视觉，虚拟现实，透镜等。Kurento媒体服务器功能如图2所示：  

<div align=center>
<img src=https://s2.ax1x.com/2019/04/02/A6ry4K.png width=450/>
  
图2
</div>

&emsp;&emsp;关于媒体服务器的简单例子，如图3所示：  
<div align=center>
<img src=https://s2.ax1x.com/2019/04/02/A6rrAx.png width=450/>

图3
</div>

#### 4.3 工程搭建  
4.3.1 基于WebRTC的网页客户端搭建  
&emsp;&emsp;网页客户端主要的功能为用户交互与视频流显示，网页客户端源码参考GitHub:[lulop-k/kurento-rtsp2webrtc](https://github.com/lulop-k/kurento-rtsp2webrtc)，能够去插件化地在网页端播放rtsp视频流。  
&emsp;&emsp;源码实现  
```bash
# 1. 下载源码
$ git clone https://github.com/lulop-k/kurento-rtsp2webrtc.git
# 2. 安装程序依赖基础软件
# 2.1 安装Node.js
$ curl -sL https://deb.nodesource.com/setup | sudo bash -
$ sudo apt-get install -y nodejs
# 2.2 安装Bower软件包管理器
$ sudo npm install -g bower
# 2.3 安装HTTP服务器
$ sudo npm install -g http-server
# 3. 通过Bower安装程序运行所需依赖包
$ cd kurento-rtsp2webrtc
$ bower install
# 4. 启动程序
$ http-server
```
&emsp;&emsp;最后，可以通过浏览器打开 http://localhost:8080/ 来访问客户端。通过在 Set source URL 栏中输入 RTSP 或者 HTTP 视频流地址，点击"start"按钮进行视频播放。  
&emsp;&emsp;**特别说明：需要在安装Kurento媒体服务器后，才能正常播放视频。**  
&emsp;&emsp;启动界面如图4所示：

<div align=center>
<img src=https://s2.ax1x.com/2019/04/02/A6rg3D.png />

图4
</div>

4.3.2 Kurento媒体服务器搭建  
&emsp;&emsp;Kurento媒体服务器搭建方式可参照[官网教程](https://doc-kurento.readthedocs.io/en/6.9.0/user/installation.html#local-installation)。主要可分为利用[Docker](https://www.docker.com/)安装与本地安装，这里简单介绍利用Docker image进行安装，Docker安装另行参考[Docker官网](https://docs.docker.com/install/linux/docker-ce/ubuntu/#prerequisites)。  
&emsp;&emsp;通过Docker image安装与运行  
```bash
# 从docker hub 上下载镜像
$ docker pull kurento/kurento-media-server:xenial-latest
# 运行
$ docker run -d --name kms -p 8888:8888 kurento/kurento-media-server:xenial-latest
```

4.3.3 Kurento + WebRTC 测试结果  
测试结果如图5所示：  

<div align=center>
<img src=https://s2.ax1x.com/2019/04/02/A6rc9O.png />

图5
</div>

4.3.4 通过GStreamer提供RTSP视频流服务  
&emsp;&emsp;暂无，后续单独补充，可参考gst-rtsp-server相关参考例子。  
4.3.5 通过DeepStream插件实现全景视频拼接  
&emsp;&emsp;暂无，后续单独补充。  

&emsp;&emsp;最后，再次附上相关链接：  
&emsp;&emsp;RTSP转WebRTC源码：[kurento-rtsp2webrtc](https://github.com/lulop-k/kurento-rtsp2webrtc)  
&emsp;&emsp;Kurento流媒体服务器：[kurento-docker](https://hub.docker.com/r/kurento/kurento-media-server/)  
&emsp;&emsp;NVIDIA DeepStream SDK：[DeepStream SDK](https://developer.nvidia.com/compute/machine-learning/deepstream-downloads)  
&emsp;&emsp;GStreamer RTSP Server：[GStreamer RTSP](https://gstreamer.freedesktop.org/documentation/rtp.html)
    

### 5. 写在后面  
&emsp;&emsp;本次主要验证的是网页播放RTSP视频流的去插件化，主要是验证技术路线是否可行，由于本人非前端开发人员，暂时未对上述所述的相关技术进行深入研究与解析，才疏学浅请见谅。上述表述如有错误还望各位大佬批评指正。  
&emsp;&emsp;开展这方面的工作有些时间，但均未深入，主要的时间可能会花在解决一些莫名其妙的bug上，比如在安装Kurento媒体服务器(KMS)的时候，当每次启动网页客户端访问KMS的时候，KMS总是莫名崩溃，然后反复审查代码，最终将KMS回退一个版本就解决了。类似的事情还很多，毕竟挖坑填坑是程序员的必修课。  
&emsp;&emsp;这仅是一个demo，真正要商用路还很长，如果各位大佬有更优秀的方案还请不吝赐教。  
&emsp;&emsp;虽然项目依托是全景拼接系统，但本文主要描述的是网页播放RTSP视频流的去插件化实现，全景拼接不是本文重点。  
&emsp;&emsp;后续关于GStreamer和DeepStream的开发与使用会单独成文分享，主要分享如何自定义插件。  

### 6. 问题记录  
待后续补充。  

### 7. 补充记录  

