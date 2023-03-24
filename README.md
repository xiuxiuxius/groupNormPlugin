# groupNormPlugin
a demo for groupNormPlugin in TensorRT8.5

```bash
mkdir build

cd build

cmake ..

make

./test_groupNorm
```

目前报错信息
```
cuda check : an illegal memory access was encountered
CUDA error 700 at /data/limm/groupNormPlugin/test_groupNorm.cpp:89test_groupNorm: /data/limm/groupNormPlugin/test_groupNorm.cpp:89: void infer(nvinfer1::IExecutionContext&, CUstream_st*&, void**, float*, int): Assertion `0' failed.
已放弃 (核心已转储)
```