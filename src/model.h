#include <NvInfer.h>
#include <string>

nvinfer1::ICudaEngine* build_det_engine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder,
                                        nvinfer1::IBuilderConfig* config, nvinfer1::IOptimizationProfile* profile, nvinfer1::DataType dt);