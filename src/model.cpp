#include "model.h"
#include "config.h"

#include <iostream>
#include <fstream>
#include <map>
#include <cassert>
#include <cmath>
#include <cstring>
#include "NvInfer.h"
#include "groupNormPlugin.h"

using namespace nvinfer1;

static IPluginV2Layer* addGNLayer(INetworkDefinition *network, ITensor *input) {

  auto creator = getPluginRegistry()->getPluginCreator("GroupNorm", "1");

  // std::cout << "77777777777777" << std::endl;

  // PluginField plugin_fields[1];

  const PluginFieldCollection* pluginData = creator->getFieldNames();
  // std::cout << "plugin_data.nbFields : " << pluginData->nbFields << std::endl;
  // std::cout << "plugin_data.fields : " << pluginData->fields << std::endl;
  // IPluginV2 *plugin_obj = creator->createPlugin("gnlayer", pluginData);
  ITensor* inputTensors[] = {input};
  // auto gn = network->addPluginV2(inputTensors, kBatchSize, *plugin_obj);
  // auto gn_m = network->addElementWise(*input,*gn_ms->getOutput(0),ElementWiseOperation::kSUB);
  // auto gn = network->addElementWise(*gn_m->getOutput(0),*gn_ms->getOutput(1),ElementWiseOperation::kDIV);

  // const float epsilon = 1e-5F;
  // plugin_fields[0].data = &epsilon;
  // plugin_fields[0].length = 1;
  // plugin_fields[0].name = "test1";
  // plugin_fields[0].type = PluginFieldType::kFLOAT32;

  // int32_t bSwish = 0;
  // plugin_fields[1].data = &bSwish;
  // plugin_fields[1].length = 1;
  // plugin_fields[1].name = "test2";
  // plugin_fields[1].type = PluginFieldType::kINT32;

  // int netinfo[3] = {kChannel, kInputW, kInputH};
  // plugin_fields[0].data = netinfo;
  // plugin_fields[0].length = 3;
  // plugin_fields[0].name = "netinfo";
  // plugin_fields[0].type = PluginFieldType::kFLOAT32;

  // PluginFieldCollection plugin_data;
  // plugin_data.nbFields = 1;
  // plugin_data.fields = plugin_fields;
  IPluginV2 *plugin_obj = creator->createPlugin("gnlayer", pluginData);

  // std::cout << "8888888888888" <<(plugin_obj==nullptr)<< std::endl;

  // std::vector<ITensor*> input_tensors;
  // input_tensors.push_back(&input);

  auto gn = network->addPluginV2(inputTensors, kBatchSize, *plugin_obj);

  // std::cout << "999999999999999" << std::endl;

  return gn;
}

ICudaEngine* build_det_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config,IOptimizationProfile* profile, DataType dt) 
{
    // std::cout << " build_det_engine    begin" << std::endl;
    INetworkDefinition* network = builder->createNetworkV2(1U);
    // NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
    // Create input tensor of shape {3, kInputH, kInputW}
    ITensor* data = network->addInput(kInputTensorName, dt, Dims4{-1, kChannel, kInputH, kInputW });

    assert(data);

    // std::cout << " 4444444444444444444" << std::endl;
    
    auto gn = addGNLayer(network, data);

    // std::cout << " ##################" << std::endl;
    // auto ret = cudaGetLastError();
    // std::cout<< "1 : " <<cudaGetErrorString(ret)<<std::endl;
    // std::cout<< "(gn==nullptr) : " << (gn==nullptr)<<std::endl;
    // gn->getOutput(0);
    // ret = cudaGetLastError();
    // std::cout<<"2 : " <<cudaGetErrorString(ret)<<std::endl;
    // std::cout << " *********************" << std::endl;
    // auto nboutputs = gn->getNbOutputs();
    // std::cout << "nboutputs : " << nboutputs << std::endl;

    gn->getOutput(0)->setName(kOutputTensorName);
    // auto d = gn->getOutput(0)->getDimensions();
    // size_t d_size = 1;
    // for(int i = 0;i < d.nbDims;++ i){
    //   std::cout << d.d[i] << std::endl;
    //   d_size *= d.d[i];
    // }
    // std::cout <<" d_size : "<< d_size << std::endl;

    network->markOutput(*gn->getOutput(0));


    // std::cout << " 55555555555555555555" << std::endl;

    // Engine config
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
    config->setFlag(BuilderFlag::kFP16); // fp16
    profile->setDimensions(kInputTensorName, nvinfer1::OptProfileSelector::kMIN, Dims4{1, kChannel, kInputH, kInputW});
    profile->setDimensions(kInputTensorName, nvinfer1::OptProfileSelector::kOPT, Dims4{std::max(1,int(kBatchSize/2)), kChannel, kInputH, kInputW});
    profile->setDimensions(kInputTensorName, nvinfer1::OptProfileSelector::kMAX, Dims4{kBatchSize, kChannel, kInputH, kInputW});
    config->addOptimizationProfile(profile);
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if(engine != nullptr)
      std::cout << "Build engine successfully!" << std::endl;
    else
      std::cout << "Build engine failed!" << std::endl;
    // Don't need the network any more
    network->destroy();
    return engine;
}

