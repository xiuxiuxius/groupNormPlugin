#pragma once

/* --------------------------------------------------------
 * These configs are related to tensorrt model, if these are changed,
 * please re-compile and re-serialize the tensorrt model.
 * --------------------------------------------------------*/

// For INT8, you need prepare the calibration dataset, please refer to
// https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5#int8-quantization
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32

// These are used to define input/output tensor names,
// you can set them to whatever you want.
const static char* kInputTensorName = "data";
const static char* kOutputTensorName = "prob";

// Detection model and Segmentation model' number of classes
constexpr static int kNumClass = 80;

// Classfication model's number of classes
constexpr static int kClsNumClass = 1000;

constexpr static int kBatchSize = 1;
constexpr static int kBatchSize_infer = 1;

// Yolo's input width and height must by divisible by 32
constexpr static int kInputH = 64;
constexpr static int kInputW = 64;
// [03/21/2023-11:19:31] [F] [TRT] Assertion failed: params.c % params.cPerBlock == 0 && params.hw % params.hwPerBlock == 0
constexpr static int kChannel = 128;

// Classfication model's input shape
constexpr static int kClsInputH = 224;
constexpr static int kClsInputW = 224;

// Maximum number of output bounding boxes from yololayer plugin.
// That is maximum number of output bounding boxes before NMS.
constexpr static int kMaxNumOutputBbox = 1000;

constexpr static int kNumAnchor = 3;

// The bboxes whose confidence is lower than kIgnoreThresh will be ignored in yololayer plugin.
constexpr static float kIgnoreThresh = 0.1f;

/* --------------------------------------------------------
 * These configs are NOT related to tensorrt model, if these are changed,
 * please re-compile, but no need to re-serialize the tensorrt model.
 * --------------------------------------------------------*/

// NMS overlapping thresh and final detection confidence thresh
const static float kNmsThresh = 0.45f;
const static float kConfThresh = 0.5f;

const static int kGpuId = 0;

// If your image size is larger than 4096 * 3112, please increase this value
const static int kMaxInputImageSize = 2048 * 2048;


