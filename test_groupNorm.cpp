#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"

using namespace nvinfer1;
static Logger gLogger;

const static int kOutputSize = kChannel * kInputH * kInputW;

void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer) {
  assert(engine->getNbBindings() == 2);
  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex = engine->getBindingIndex(kOutputTensorName);
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  // Create GPU buffers on device
  // auto odims = engine->getBindingDimensions(outputIndex);
  // size_t o_size = 1;
  // for(int i=0;i<odims.nbDims;++i){
  //   o_size *= odims.d[i];
  // }
  // std::cout<<"output size"<<o_size<<std::endl;
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

  *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize) {
    // cudaSetDevice(kGpuId);
    
    bool ret = context.enqueue(batchsize, gpu_buffers, stream, nullptr);
    // std::cout << "type : "  << typeid( gpu_buffers[1] ).name() << std::endl;
    // std::cout << "******66666   : " << sizeof(gpu_buffers) <<std::endl;
    // std::cout << "******66666   : " << sizeof(gpu_buffers[0]) <<std::endl;
    // std::cout << "******66666   : " << sizeof(gpu_buffers[1]) <<std::endl;
    // assert(ret==true);
    // std::cout << "ret:"  << ret << std::endl;
    // auto ret = cudaGetLastError();
    // CUDA_CHECK(ret)
    // std::cout<< "55 : " <<cudaGetErrorString(ret)<<std::endl;
    // ret = cudaMemset(gpu_buffers[1],0,kBatchSize * kOutputSize*4);
    // std::cout<< "58 : " <<cudaGetErrorString(ret)<<std::endl;
    // memset(output, 0, kBatchSize * kOutputSize * 4); //4个字节
    // std::cout<< "59 : " << batchsize <<std::endl;
    // float* a = new float[kOutputSize];
    // std::cout<<"addr1 : "<<output<<std::endl;
    // cudaSetDevice(kGpuId);
    // cudaStreamSynchronize(stream);
    // std::cout << "kBatchSize * kOutputSize: " << kBatchSize * kOutputSize << std::endl;
    // for(int i = 1; i < kOutputSize*1000; i ++)
    // {
    //   std::cout << "i : " << i << std::endl;
    //   ret = cudaMemcpyAsync(output, gpu_buffers[1], i, cudaMemcpyDeviceToHost, stream);
    //   if(ret != cudaSuccess) continue;
    //   else{
    //     std::cout << "i : " << i << std::endl;
    //     break;
    //   }
    // }
    CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    // ret = cudaGetLastError();
    // std::cout<< "66 : " <<cudaGetErrorString(ret)<<std::endl;
    cudaStreamSynchronize(stream);
    // std::cout << "kOutputSize : " << kOutputSize << std::endl;
    // std::ofstream cpp_input;
    // cpp_input.open("infer.txt");
    // for (int i = 0; i < kInputH; ++i)
    // {
    //     for (int j = 0; j < kInputW; ++j)
    //     {
    //         for (int c = 0; c < kChannel; c++)
    //         { 
    //             cpp_input << output[i * kInputW + j] << ",";  
    //         }
    //     }
    //     cpp_input << "\n";
    // }
    // cpp_input.close();


}

void serialize_engine(unsigned int max_batchsize, std::string& engine_name) {
  // Create builder
  IBuilder* builder = createInferBuilder(gLogger);
  IBuilderConfig* config = builder->createBuilderConfig();
  IOptimizationProfile* profile = builder->createOptimizationProfile();

  // Create model to populate the network, then set the outputs and create an engine
  ICudaEngine *engine = nullptr;
  engine = build_det_engine(max_batchsize, builder, config, profile, DataType::kFLOAT);
  assert(engine != nullptr);
  // Serialize the engine
  IHostMemory* serialized_engine = engine->serialize();
  assert(serialized_engine != nullptr);

  // Save engine to file
  std::ofstream p(engine_name, std::ios::binary);
  if (!p) {
    std::cerr << "Could not open plan output file" << std::endl;
    assert(false);
  }

  p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

  // Close everything down
  engine->destroy();
  builder->destroy();
  config->destroy();
  serialized_engine->destroy();
}

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good()) {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
  }
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  char* serialized_engine = new char[size];
  assert(serialized_engine);
  file.read(serialized_engine, size);
  file.close();

  *runtime = createInferRuntime(gLogger);
  assert(*runtime);
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  assert(*engine);
  *context = (*engine)->createExecutionContext();
  assert(*context);
  delete[] serialized_engine;
}

int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);
    std::string wts_name = "";
    std::string engine_name = "../gn.engine";
    std::string img_dir = "../images";

    // std::cout << "111111111111" << std::endl;

    // Create a model using the API directly and serialize it to a file
    serialize_engine(kBatchSize, engine_name);
    // std::cout << "2222222222222" << std::endl;

    // Deserialize the engine from file
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);
    // context->setBindingDimensions(0, Dims3{kChannel, kInputH, kInputW});
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Init CUDA preprocessing
    cuda_preprocess_init(kMaxInputImageSize);

    // Prepare cpu and gpu buffers
    float* gpu_buffers[2];
    float* cpu_output_buffer = nullptr;

    // auto bindings_number = engine->getNbBindings();
    // for (int i = 0; i < bindings_number; i++)
    // {
    //     size_t size = 1;
    //     auto dimension = engine->getBindingDimensions(i);
    //     for (int j = 0; j < dimension.nbDims; ++j)
    //     {
    //         std::cout<<dimension.d[j]<<" ";
    //         size *= dimension.d[j];
    //     }
    //     std::cout<< "size : " << size << std::endl;
    // }

    prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);

    // CUDA_CHECK(cudaMemcpy(cpu_output_buffer, gpu_buffers[1], kOutputSize*sizeof(float), cudaMemcpyDeviceToHost);)

    // std::cout<<"addr1"<<cpu_output_buffer<<std::endl;
    // Read images from directory
    std::vector<std::string> file_names{"bus.jpg", "zidane.jpg"};
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // batch predict
    std::vector<cv::Mat> img_batch;
    cv::Mat img = cv::imread(img_dir + "/" + file_names[0]);
    img_batch.push_back(img);
    // // cv::Mat img = cv::Mat(kInputH, kInputW, CV_8UC3);
    // int sizeN[kChannel];
    // for(int i = 0; i < kChannel;i ++) sizeN[i] = kInputH * kInputW;
    // cv::Mat img(kChannel, sizeN, CV_8UC1, cv::Scalar(99)); 
    // int sp = img.dims;
    // int h = img.rows;
    // int w = img.elemSize();
    // std::cout << sp << " " << h << " " << w << std::endl;
    // // cv::Mat img(kChannel, kInputH, kInputW, CV_8UC1, cv::Scalar(0)); )
    // // cv::randu(img, cv::Scalar::all(0), cv::Scalar::all(255));
    //自定义数据类型
    // typedef cv::Vec<float, kChannel> Vec32f;
    //生成一个2x3x5的Mat，数据为double型
    // cv::Mat img = cv::Mat::zeros(kInputH, kInputW, CV_32FC(kChannel));                                             
    // std::cout << "channel = " << img.channels() << std::endl;
    // for (int i = 0; i < img.rows; i++)
    // {
    //     for (int j = 0; j < img.cols; j++)
    //     {
    //         for (int c = 0; c < img.channels(); c++)
    //         {
    //             //给M的每一个元素赋值                
    //             img.at<Vec32f>(i, j)[c] = c / 255.0f;              
    //         }
    //     }
    // }
    // std::cout << img << std::endl;

    // img_batch.push_back(img);

    // {
    //   std::ofstream cpp_input;
    //   cpp_input.open("./img.txt");

    //   for (int i = 0; i < img.rows; i++)
    //   {
    //       for (int j = 0; j < img.cols; j++)
    //       {
    //           for (int c = 0; c < img.channels(); c++)
    //           { 
    //               cpp_input << img.at<Vec32f>(i, j)[c] << ",";           
    //           }
    //       }
    //       cpp_input << "\n";
    //   }
    //   // for(int k = 0;k < kInputH;k ++)
    //   // {
    //   //     // 每一行图像的指针
    //   //     const uchar* inData = img.ptr<uchar>(k);
    //   //     for(int i = 0;i < kInputW;i ++)
    //   //     {
    //   //           for(int j = 0;j < kChannel;j += 1)
    //   //           {
    //   //             // std::cout << ("inData[%d] : ", j) << "   " << (int)inData[i+j] << std::endl;
    //   //             cpp_input << (int)inData[i] << ",";
    //   //           }
    //   //     }
    //   //     cpp_input << "\n";
    //   // }
    //   cpp_input.close();
    // }
    
    
    // Preprocess
    // cuda_preprocess(img.ptr(), img.cols, img.rows, gpu_buffers[0], kInputW, kInputH, stream);
    cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    // CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));

    std::cout << "^^^^^^^999     : " << sizeof(gpu_buffers) <<std::endl;

    std::cout << "&&&&&&&&&&&&&&&&&" <<std::endl;
    // Run inference
    auto start = std::chrono::system_clock::now();
    infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(gpu_buffers[0]));
    CUDA_CHECK(cudaFree(gpu_buffers[1]));
    delete[] cpu_output_buffer;
    cuda_preprocess_destroy();
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}