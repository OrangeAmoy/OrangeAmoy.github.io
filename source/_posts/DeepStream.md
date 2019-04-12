---
title: NVIDIA DeepStream 简介
date: 2019-04-12 18:00:00
tags:
categories:
- 技术文档
- 视频应用
---

# 1. 技术简介  
&emsp;&emsp;DeepStream是NVIDIA推出的一个针对智能视频分析应用和多传感器处理的工具包，其主要特性为能够利用硬件加速技术将深度神经网络与其他的复杂处理任务带入流处理管道，让用户更专注于构建神经网络任务，而不是从头开始搭建端到端的解决方案。在以前，开发一个智能视频分析应用，用户除了需要构建神经网络，还需要考虑前端视频数据的获取、视频编解码、视频渲染等问题。然而这些问题的解决可能需要同时引入多个不同的开发工具包，诸如OpenCV，FFmpeg等。现在，通过DeepStream SDK便可以解决这些问题。  
&emsp;&emsp;DeepStream的典型应用架构如图所示：  

<div align=center>
<img src=https://s2.ax1x.com/2019/04/12/AbLT1O.png />
</div>  

<!-- more -->

&emsp;&emsp;从本质上来说DeepStream是NVIDIA基于[GStreamer](https://gstreamer.freedesktop.org/)的插件系统开发的，自然也继承了GStreamer的特性。NVIDIA将自家的技术，如：TensorRT, cuDNN，CUDA, Video SDK等以插件的形式集成进GStreamer当中，以管线的形式进行智能视频分析应用的开发，将各个功能封装成组件，通过将对应功能的组件插入管线中，启动管线使数据按照要求在管线内流动，数据经过解析、编解码、预处理、算法处理后进行图像渲染或者发送到云端。下面对上述架构中的模块进行简要说明：  
- Video Decode: 视频解码模块。支持摄像头输入，RTSP格式输入，视频文件输入。  
- Stream Mux: 数据聚合模块。能够将多路视频数据图像进行聚合，以批量的形式提供给下游推理引擎进行算法推理。  
- Primary Detector: 一级检测。基于TensorRT的神经网络推理引擎，主要实现对图像中的目标进行检测。  
- Object Tracker: 目标跟踪。对检测的目标进行跟踪，基于OpenCV实现。  
- Secondary Classifiers: 二级分类。基于TensorRT的神经网络推理引擎，在一级检测的基础上对目标进行分类，或者结构化信息提取。  
- Tiler: 视频阵列。将多个视频以阵列的形式进行组合，形成照片墙。  
- OnScrenn Display: 视频结构化信息叠加。能够将结构化信息叠加在视频上，例如画矩形框，叠加文字信息等。  
- Renderer: 视频渲染。对接受到的视频帧进行渲染并显示。  
- Message Converter: 消息转换。将视频结构化信息转换成对应的消息格式，如JSON。  
- Message Broker: 消息发送。将视频结构化信息等发送到云端。  

&emsp;&emsp;参考资料：  
&emsp;&emsp;[DeepStream 3.0](https://developer.nvidia.com/deepstream-sdk#github)  

# 2. 安装教程  
## 2.1 依赖环境  
- Ubuntu 16.04  
- Gstreamer 1.8.3  
- NVIDIA driver 410+  
- CUDA 10  
- cuDNN 7.3  
- TensorRT 5.0  

## 2.2 移除旧版本DeepStream(可选)  
```bash
$ sudo rm -rf /usr/local/deepstream \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnv* \
    /usr/bin/deepstream-* \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libnvdsgst*
```

## 2.3 安装步骤  
1.安装必要的第三方开发包  
```bash
$ sudo apt install \
    libssl1.0.0 \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstrtspserver-1.0-0 \
    libjansson4
```

2.下载并安装CUDA 10，cuDNN 7.3，TensorRT 5.0  
&emsp;&emsp;关于CUDA 10，cuDNN 7.3，TensorRT 5.0等安装教程本文暂不讨论，请另行参考对应教程。  

3.建立 libnvcuvid.so 软链接  
```bash
$ sudo ln -s /usr/lib/nvidia-<driver-version>/libnvcuvid.so \
                /usr/lib/x86_64-linux-gnu/libnvcuvid.so
```
实际在/usr/lib/x86_64-linux-gnu/中已经有libnvcuvid.so文件，故暂未理解此操作意图。  

4.安装 librdkafka  
```bash
# 1. 从GitHub下载源码
$ git clone https://github.com/edenhill/librdkafka.git
# 2. 配置与编译
$ cd librdkafka
$ git reset --hard 7101c2310341ab3f4675fc565f64f0967e135a6a
$ ./configure
$ make
$ sudo make install
# 3. 拷贝生成的库文件至deepstream安装目录
$ sudo mkdir -p /usr/local/deepstream
$ sudo cp /usr/local/lib/librdkafka* /usr/local/deepstream
```

5.安装deepstream  
&emsp;&emsp;从官网下载[DeepStream SDK](https://developer.nvidia.com/compute/machine-learning/deepstream-downloads)，并解压，进入文件根目录运行  
```bash
$ sudo tar -xvpf binaries.tbz2 -C /
```

## 2.4 验证安装  
&emsp;&emsp;命令行输入：deepstream-app -c < path to config.txt >，参考例子：
```bash
$ deepstream-app -c samples/configs/deepstream-app/source30_720p_dec_infer-resnet_tiled_display_int8.txt
```
如运行成功则有对应的日志及视频输出。  

# 3. 参考例子  
&emsp;&emsp;关于DeepStream使用的例子可自行参考DeepStream中对应的源码，本文不做重点描述。  
&emsp;&emsp;本文着重描述如何利用插件进行自定义功能的开发。DeepStream插件参考源码见SDK目录下sources/gst-plugins/gst-dsexample文件夹。由于DeepStream是基于GStreamer开发的，因此DeepStream插件的开发也必然遵循GStreamer插件开发的约束。不过，一些简单的插件，特别是大部分神经网络推理类别的插件，即那种没有对原始视频或者图像进行操作，只是在其基础上使用算法进行特定功能分析的情况，可直接参考gst-dsexample，并进行简单的修改。如果需要对原始视频或者图像进行操作，则需要进一步深入研究GStreamer的原理与插件开发规则等。接下来主要从三个方面论述如何利用gst-dsexample进行插件开发。  
## 3.1 关于gst-dsexample  
### 3.1.1 gst-dsexample简介  
&emsp;&emsp;gst-dsexample基于GStreamer开发，继承至[GstBaseTransform](https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-libs/html/GstBaseTransform.html)类。gst-dsexample中共包含四个文件：gstdsexample.h，gstdsexample.cpp，dsexample_lib.h，dsexample_lib.c。gstdsexample主要实现插件的初始化，插件的注册等功能。dsexample_lib主要负责插件具体功能的实现，如目标检测与目标分类。每个插件均可单独配置不同的参数，用于对插件功能进行设置，如配置处理图像的宽和高、使用GPU的ID等。插件按数据流处理方式可分为直通式与非直通式。直通模式下，数据流直接通过插件，插件仅对流过的数据进行处理，并将处理后的结果附加在数据上传递给下游插件。如获取图像数据并通过算法进行目标检测，将目标检测结果附加在数据中，传递给下游组件，便于后续的分析。gst-dsexample就是直通式插件，获取图像数据并经过算法分析后，将分析结果附加在gst-buffer里，传递给下游组件。非直通模式下，数据流被插件截取并处理后再传递给下游插件，传递给下游的数据流可能由本插件生成，而不再是原始数据。例如图像缩放插件，当图像被放大时，图像数据空间已经改变，必须由缩放插件自行开辟数据空间，并将放大后的图像数据传递给下游。直通与非直通的区别，核心就在于数据流管理方式的不同。下面将对gst-dsexample的插件源码进行解析，关于GStreamer插件相关知识参考GStreamer官方介绍或者后续技术总结，本文会忽略相关知识。  
### 3.1.2 gst-dsexample源码解析  
gstdsexample.h
```c++
#ifndef __GST_DSEXAMPLE_H__
#define __GST_DSEXAMPLE_H__

#include <gst/base/gstbasetransform.h>
...
#include "dsexample_lib/dsexample_lib.h"

/* 插件描述信息，其中PACKAGE字段指定了插件的名称，通过该名称实现插件的调用。 */
#define PACKAGE "dsexample"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "NVIDIA example plugin for integration with DeepStream on DGPU"
#define BINARY_PACKAGE "NVIDIA DeepStream 3rdparty IP integration example plugin"
#define URL "http://nvidia.com/"

G_BEGIN_DECLS
/* 固定结构，声明插件实例与插件类 */
typedef struct _GstDsExample GstDsExample;
typedef struct _GstDsExampleClass GstDsExampleClass;

/* 固定结构 */
#define GST_TYPE_DSEXAMPLE (gst_dsexample_get_type())
#define GST_DSEXAMPLE(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DSEXAMPLE,GstDsExample))
#define GST_DSEXAMPLE_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DSEXAMPLE,GstDsExampleClass))
#define GST_DSEXAMPLE_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DSEXAMPLE, GstDsExampleClass))
#define GST_IS_DSEXAMPLE(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DSEXAMPLE))
#define GST_IS_DSEXAMPLE_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DSEXAMPLE))
#define GST_DSEXAMPLE_CAST(obj)  ((GstDsExample *)(obj))

struct _GstDsExample
{
    GstBaseTransform base_trans;

    // 自定义算法实现
    DsExampleCtx *dsexamplelib_ctx;

    // 自定义变量
    guint unique_id;

};

struct _GstDsExampleClass
{
    GstBaseTransformClass parent_class;
};

GType gst_dsexample_get_type (void);

G_END_DECLS
#endif /* __GST_DSEXAMPLE_H__ */
```
_GstDsExample为类的实例对象，里面可以存放运行时所需的变量。_GstDsExampleClass则为类，主要定义成员函数。可以看出GstBaseTransform继承至GstBaseTransform。程序所需的自定义函数或者变量可直接在_GstDsExample中添加，其余参数可视为固定模板，可不做修改。  

gstdsexample.cpp
```c++
#include "gstdsexample.h"
GST_DEBUG_CATEGORY_STATIC (gst_dsexample_debug);
#define GST_CAT_DEFAULT gst_dsexample_debug

static GQuark _dsmeta_quark = 0;

/* 插件属性的枚举变量 */
enum
{
    PROP_0,
    PROP_UNIQUE_ID
};

/* 插件属性默认值 */
#define DEFAULT_UNIQUE_ID 15

#define RGB_BYTES_PER_PIXEL 3

/* 设置插件sink和src端的连接方式与属性，这里设置数据存放于GPU中，所以内存指定为memory:NVMM */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_dsexample_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA }")));

static GstStaticPadTemplate gst_dsexample_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_dsexample_parent_class parent_class
G_DEFINE_TYPE (GstDsExample, gst_dsexample, GST_TYPE_BASE_TRANSFORM);

/* 
* 插件的配置函数，配置子类的实现、插件属性、插件描述信息等。
*/
static void
gst_dsexample_class_init (GstDsExampleClass * klass)
{
    GObjectClass *gobject_class;
    GstElementClass *gstelement_class;
    GstBaseTransformClass *gstbasetransform_class;

    gobject_class = (GObjectClass *) klass;
    gstelement_class = (GstElementClass *) klass;
    gstbasetransform_class = (GstBaseTransformClass *) klass;

    /* 重构类的实现函数 */
    gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_dsexample_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_dsexample_get_property);

    gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_dsexample_set_caps);
    gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_dsexample_start);
    gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_dsexample_stop);
    // 默认是直通模式下，所以子类重构的是transform_ip函数。
    gstbasetransform_class->transform_ip =
        GST_DEBUG_FUNCPTR (gst_dsexample_transform_ip);

    /* 注册插件属性 */
    g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
        g_param_spec_uint ("unique-id",
            "Unique ID",
            "Unique ID for the element. Can be used to identify output of the"
            " element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID, (GParamFlags)
            (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    /* 设置sink和src的pad capabilities属性 */
    gst_element_class_add_pad_template (gstelement_class,
        gst_static_pad_template_get (&gst_dsexample_src_template));
    gst_element_class_add_pad_template (gstelement_class,
        gst_static_pad_template_get (&gst_dsexample_sink_template));

    /* 设置插件描述信息 */
    gst_element_class_set_details_simple (gstelement_class,
        "DsExample plugin",
        "DsExample Plugin",
        "Process a 3rdparty example algorithm on objects / full frame",
        "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
        "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_dsexample_init (GstDsExample * dsexample)
{
    GstBaseTransform *btrans = GST_BASE_TRANSFORM (dsexample);

    /* 设置是否基于同一内存空间处理buffer，在直通模式下，inbuf和outbuf是同一内存空间的。 */
    gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
    /* 设置是否是直通模式，直通模式下调用的是transform_ip函数，非直通模式下调用transform */
    gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

    /* 设置插件属性默认值 */
    dsexample->unique_id = DEFAULT_UNIQUE_ID;

    /* This quark is required to identify NvDsMeta when iterating through
    * the buffer metadatas */
    if (!_dsmeta_quark)
        _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* 
* 设置插件属性信息
*/
static void
gst_dsexample_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
    GstDsExample *dsexample = GST_DSEXAMPLE (object);
    switch (prop_id) {
        case PROP_UNIQUE_ID:
            dsexample->unique_id = g_value_get_uint (value);
        break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

/* 
* 获取插件属性信息
*/
static void
gst_dsexample_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
    GstDsExample *dsexample = GST_DSEXAMPLE (object);

    switch (prop_id) {
        case PROP_UNIQUE_ID:
        g_value_set_uint (value, dsexample->unique_id);
        break;
        default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

/**
* 初始化资源，如算法初始化。
*/
static gboolean
gst_dsexample_start (GstBaseTransform * btrans)
{
    GstDsExample *dsexample = GST_DSEXAMPLE (btrans);
    DsExampleInitParams init_params =
        { dsexample->processing_width, dsexample->processing_height,
        dsexample->process_full_frame
    };

    /* Algorithm specific initializations and resource allocation. */
    dsexample->dsexamplelib_ctx = DsExampleCtxInit (&init_params);

    GST_DEBUG_OBJECT (dsexample, "ctx lib %p \n", dsexample->dsexamplelib_ctx);
    ...
    return TRUE;
    error:
    if (dsexample->dsexamplelib_ctx)
        DsExampleCtxDeinit (dsexample->dsexamplelib_ctx);
    return FALSE;
}

/**
* 释放资源。
*/
static gboolean
gst_dsexample_stop (GstBaseTransform * btrans)
{
    GstDsExample *dsexample = GST_DSEXAMPLE (btrans);
    // 释放算法内存空间
    DsExampleCtxDeinit (dsexample->dsexamplelib_ctx);
    dsexample->dsexamplelib_ctx = NULL;
    ...
    return TRUE;
}

/**
* 获取协商好的caps参数，并进行相应的处理。如获取输入图像的宽高等信息。
*/
static gboolean
gst_dsexample_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
    GstDsExample *dsexample = GST_DSEXAMPLE (btrans);
    ...
    return TRUE;
    error:
    return FALSE;
}

/*
* 直通模式下，获取上游数据并进行算法处理后，将检测后的结果叠加在GstBuffer里面传递给下游。
*/
static GstFlowReturn
gst_dsexample_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
    GstDsExample *dsexample = GST_DSEXAMPLE (btrans);
    GstMapInfo in_map_info;
    GstFlowReturn flow_ret = GST_FLOW_ERROR;
    gdouble scale_ratio;
    DsExampleOutput *output;

    NvBufSurface *surface = NULL;
    GstNvStreamMeta *streamMeta = NULL;

    memset (&in_map_info, 0, sizeof (in_map_info));
    if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
        g_print ("Error: Failed to map gst buffer\n");
        goto error;
    }

    surface = (NvBufSurface *) in_map_info.data;

    /* 从inbuf里面获取nvidia的streamMeta数据结构 */
    streamMeta = gst_buffer_get_nvstream_meta (inbuf);

        /* 对一级检测来说，处理的是整张图片，对二级分类来说，处理的是目标小图 */
    if (dsexample->process_full_frame) {
        for (guint i = 0; i < batch_size; i++) {
            NvOSD_RectParams rect_params;

            // 定义缩放图像的大小
            rect_params.left = 0;
            rect_params.top = 0;
            rect_params.width = dsexample->video_info.width;
            rect_params.height = dsexample->video_info.height;

            // 对图像数据进行尺度缩放与空间转换等预处理操作
            if (get_converted_mat_dgpu (dsexample, surface->buf_data[i], &rect_params,
                    scale_ratio, dsexample->video_info.width,
                    dsexample->video_info.height) != GST_FLOW_OK) {
                goto error;
            }
            // 运行检测算法，获取检测结果。
            output =
                DsExampleProcess (dsexample->dsexamplelib_ctx,
                dsexample->cvmat->data);
            // 将检测结果叠加在inbuf里面传递给下游组件。
            attach_metadata_full_frame (dsexample, inbuf, scale_ratio, output, i);
            free (output);
        }
    } else {
        /* 针对二级分类进行处理 */
        ...
    }
    flow_ret = GST_FLOW_OK;
    error:
    gst_buffer_unmap (inbuf, &in_map_info);
    return flow_ret;
}

/*
* 释放申请的内存空间。
*/
static void
free_ds_meta (gpointer meta_data)
{
    NvDsFrameMeta *params = (NvDsFrameMeta *) meta_data;
    for (guint i = 0; i < params->num_rects; i++) {
        g_free (params->obj_params[i].text_params.display_text);
    }
    g_free (params->obj_params);
    g_free (params);
}

/*
* 添加一级检测的结果，如目标位置框信息。
*/
static void
attach_metadata_full_frame (GstDsExample * dsexample, GstBuffer * inbuf,
    gdouble scale_ratio, DsExampleOutput * output, guint batch_id)
{
    NvDsMeta *dsmeta;
    NvDsFrameMeta *bbparams = (NvDsFrameMeta *) g_malloc0 (sizeof (NvDsFrameMeta));
    // 根据检测到的目标数量申请对应的目标参数内存空间。
    bbparams->obj_params =
        (NvDsObjectParams *) g_malloc0 (sizeof (NvDsObjectParams) *
        output->numObjects);
    GST_DEBUG_OBJECT (dsexample, "Attaching metadata %d\n", output->numObjects);
    ...
    // 将NvDsFrameMeta以NvDsMeta的方式添加到buffer中。
    dsmeta = gst_buffer_add_nvds_meta (inbuf, bbparams, free_ds_meta);
    // 指定NvDsMeta的数据内容为NVDS_META_FRAME_INFO。
    dsmeta->meta_type = NVDS_META_FRAME_INFO;
}

/*
* 添加二级分类的结果。
*/
static void
attach_metadata_object (GstDsExample * dsexample, NvDsObjectParams * obj_param,
    DsExampleOutput * output)
{
    ...
}

/*
* 用于向GStreamer注册插件
*/
static gboolean
dsexample_plugin_init (GstPlugin * plugin)
{
    GST_DEBUG_CATEGORY_INIT (gst_dsexample_debug, "dsexample", 0,
        "dsexample plugin");

    return gst_element_register (plugin, "dsexample", GST_RANK_PRIMARY,
        GST_TYPE_DSEXAMPLE);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    dsexample,
    DESCRIPTION, dsexample_plugin_init, "3.0", LICENSE, BINARY_PACKAGE, URL)

```
&emsp;&emsp;这里仅作简单注释，为了减少篇幅也对源码进行了裁剪，详细源码参考官方。该文件可理解为插件的接口实现函数，可以对插件进行定制化开发，配置算法所需的参数文件、实现算法的初始化、配置插件属性、进行插件注册等工作。对于自定义算法的处理结果如何传递给下游插件，参考关于NvDsMeta的描述便可，NvDsMeta支持多种功能的数据格式。  
&emsp;&emsp;这里插件的名字指定为dsexample，当然也可以根据自己的需求修改插件名字，官方有提供一些工具进行修改，我这里的话之间简单的将插件名字进行替换(对整个插件源代码进行替换)，也满足要求。  
&emsp;&emsp;关于部署。插件编译成功部署的时候，仅需要把生成的.so文件放在/usr/lib/x86_64-linux-gnu/gstreamer-1.0/目录下。通过make install命令也可以实现同样的效果。  
&emsp;&emsp;dsexample_lib相关文件为算法的具体实现文件，分为算法的初始化与算法的实现两部分，比较简单，本文不做过多阐述。  


## 3.2 人体姿态估计算法插件  
&emsp;&emsp;在实践过程中，以之前的人体姿态估计算法相关工作为切入点，完成了基于gst-dsexample的人体姿态估计算法插件开发。总体设计思想基于上文所述，配置了相关接口，将人体姿态估计算法嵌入插件中，实现功能。在后续考虑在git中上传源码，本文不做过多描述，仅贴出通过gst-inspect-1.0 dspose命令运行后的插件描述结果。  
```bash
Factory Details:
Rank                     primary (256)
Long-name                DsPose plugin
Klass                    DsPose Plugin
Description              Process a PoseEstimation algorithm on full frame
Author                   xiaochengliu.prc@foxmail.com

Plugin Details:
Name                     nvdspose
Description              pose plugin for integration with DeepStream on DGPU
Filename                 /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnvdspose.so
Version                  3.0
License                  Proprietary
Source module            dspose
Binary package           NVIDIA DeepStream 3rdparty IP integration pose plugin
Origin URL               http://nvidia.com/

GObject
+----GInitiallyUnowned
    +----GstObject
            +----GstElement
                +----GstBaseTransform
                        +----GstDsPose

Pad Templates:
SRC template: 'src'
    Availability: Always
    Capabilities:
    video/x-raw(memory:NVMM)
                format: { NV12, RGBA }
                width: [ 1, 2147483647 ]
                height: [ 1, 2147483647 ]
            framerate: [ 0/1, 2147483647/1 ]

SINK template: 'sink'
    Availability: Always
    Capabilities:
    video/x-raw(memory:NVMM)
                format: { NV12, RGBA }
                width: [ 1, 2147483647 ]
                height: [ 1, 2147483647 ]
            framerate: [ 0/1, 2147483647/1 ]


Element Flags:
no flags set

Element Implementation:
Has change_state() function: gst_element_change_state_func

Element has no clocking capabilities.
Element has no URI handling capabilities.

Pads:
SINK: 'sink'
    Pad Template: 'sink'
SRC: 'src'
    Pad Template: 'src'

Element Properties:
name                : The name of the object
                        flags: readable, writable
                        String. Default: "dspose0"
parent              : The parent of the object
                        flags: readable, writable
                        Object of type "GstObject"
qos                 : Handle Quality-of-Service events
                        flags: readable, writable
                        Boolean. Default: false
unique-id           : Unique ID for the element. 
                        flags: readable, writable
                        Unsigned Integer. Range: 0 - 4294967295 Default: 15 
processing-width    : Width of the input buffer to algorithm
                        flags: readable, writable
                        Integer. Range: 1 - 2147483647 Default: 1920 
processing-height   : Height of the input buffer to algorithm
                        flags: readable, writable
                        Integer. Range: 1 - 2147483647 Default: 1080 
full-frame          : Enable to process full frame
                        flags: readable, writable
                        Boolean. Default: true
gpu-id              : Set GPU Device ID
                        flags: readable, writable, changeable only in NULL or READY state
                        Unsigned Integer. Range: 0 - 4294967295 Default: 0 
model-file          : Set model file location
                        flags: readable, writable
                        String. Default: null
net-input-width     : Set Net input width
                        flags: readable, writable
                        Integer. Range: -1 - 2147483647 Default: -1 
net-input-height    : Set Net input height
                        flags: readable, writable
                        Integer. Range: -1 - 2147483647 Default: 368 
```
## 3.3 全景拼接插件  
&emsp;&emsp;暂无

# 4. 写在后面  
&emsp;&emsp;在接触DeepStream的过程中，会碰到莫名的一些坑，踏平这些坑花了不少时间。DeepStream个人感觉还是蛮好用的，不过好用在于能够快速部署和使用nvidia的一些技术，但是在实际项目使用过程中，由于DeepStream现在并没有开源，当中的插件功能可能并不能满足一些实际使用场景的需求，反而又需要自己去开发，那么可能最终演变成自己基于GStreamer编写基于NVIDIA生态的组件，相当于自己写一个DeepStream SDK。由于时间关系，没有办法仔细展开，待后续补充，虽然这个后续不知道要鸽到什么时候。  
&emsp;&emsp;在一些单机的基于视频流的视频结构化分析应用场景中，DeepStream还是一个很好的框架的，可以快速完成原型部署，后面接kafka或者webrtc等就可以作为服务器往客户端传输结构化数据和视频图像啦。  