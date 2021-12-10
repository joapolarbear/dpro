
# 3rdparty version

## Frameworks
* [MXNet](https://github.com/joapolarbear/incubator-mxnet/tree/mlsys2022)
* [TensorFlow](https://github.com/joapolarbear/tensorflow/tree/mlsys2022)
* [BytePS](https://github.com/joapolarbear/byteps/tree/mlsys2022)
* [pslite](https://github.com/joapolarbear/ps-lite/tree/mlsys2022)
* [ZMQ](https://github.com/chenyu-jiang/libzmq/commit/5ed25589f000dc613e1a8575ba193eb78eb9b86e)
* [Horovod](https://github.com/joapolarbear/horovod/tree/mlsys2022)
* [NCCL](https://github.com/joapolarbear/nccl/tree/mlsys2022)


## Benchmarks
* [BERT]( https://github.com/joapolarbear/bert/tree/mlsys2022)
* [gluon-nlp](https://github.com/joapolarbear/gluon-nlp/tree/mlsys2022)

## Tools
* [spdlog](https://github.com/gabime/spdlog/commit/6aafa89d20eef25ec75462ffb7eedc328f135638)
* [nvprof2json](https://github.com/joapolarbear/nvprof2json): convert nvprof results to JSON format
* [catapult](https://github.com/joapolarbear/catapult): convert JSON files to a HTML in the format of chrome://tracing.


# Installation

## TensorFlow

You can installed our compiled version of TensorFlow if you are using python3.7
```
wget https://github.com/joapolarbear/tensorflow/releases/download/v2.4.1-dev.2.0.2/tensorflow-2.4.1-cp37-cp37m-linux_x86_64.whl && \
pip3 --no-cache-dir install --force-reinstall tensorflow-2.4.1-cp37-cp37m-linux_x86_64.whl && \
rm tensorflow-2.4.1-cp37-cp37m-linux_x86_64.whl
```

Or you can build our customized TensorFlow yourself using bazel. First, clone our customized TensorFlow
```
git clone --recurse-submodules -b r2.4_dev https://github.com/joapolarbear/tensorflow.git
cd tensorflow
```
Then, you need to config the building process, if you are using python3.7 and cuda11, you can also use our configuration file
```
cp tools/sample_config/cuda11.3_python3.7 .tf_configure.bazelrc
```
Install dependencies
```
pip3 install -U --user keras_applications --no-deps
pip3 install -U --user keras_preprocessing --no-deps
```
Pin default python to python3.7
```
ln -sf /usr/bin/python3 /usr/bin/python
```

Then, follow the commands below to build and install TensorFlow.
```
cd /root/tensorflow && bazel build -j 32 --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
ls -lh /tmp/tensorflow_pkg
pip3 --no-cache-dir install --force-reinstall /tmp/tensorflow_pkg/tensorflow-2.4.1-cp37-cp37m-linux_x86_64.whl
bazel clean && ln -sf /usr/bin/python2.7 /usr/bin/python && rm -rf /tmp/tensorflow_pkg/*
rm -rf tensorflow && rm -rf /var/lib/apt/lists/*
```

## MXNet
```
cd customized-mxnet
make clean_all && make -j16 USE_OPENCV=1 \
    USE_BLAS=openblas \
    USE_CUDNN=1 \
    USE_CUDA=1 \
    USE_CUDA_PATH=/usr/local/cuda \
    USE_MKLDNN=0 \
    USE_DIST_KVSTORE=1 \
    USE_NCCL=1 \
    USE_NCCL_PATH=/usr/local/nccl
cd python
python3 setup.py build
python3 setup.py install
python3 setup.py bdist_wheel
ln -sf /usr/local/cuda-10.2/targets/x86_64-linux/lib/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ln -sf /root/customized-mxnet/include $MX_PATH/include && echo $MX_PATH
```

## BytePS + pslite + ZMQ
```
cd $HOME && git clone https://github.com/gabime/spdlog.git
cd $HOME/spdlog && mkdir build && cd build && cmake .. && make -j && make install
cd $HOME && git clone --single-branch --branch byteprofile_rdma --recurse-submodules https://github.com/joapolarbear/byteps.git
cd $HOME/byteps/3rdparty/ps-lite && make -j USE_RDMA=1
cd $HOME/byteps/
BYTEPS_USE_RDMA=1 BYTEPS_WITHOUT_PYTORCH=1 BYTEPS_WITHOUT_MXNET=1 python3 setup.py install
BYTEPS_WITHOUT_PYTORCH=1 BYTEPS_WITHOUT_MXNET=1 python3 setup.py bdist_wheel
```

## Horovod + NCCL
Install OpenMPI first
```
cd $HOME
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz
rm -rf /usr/lib/x86_64-linux-gnu/openmpi
tar -xvf openmpi-4.0.3.tar.gz && cd openmpi-4.0.3
./configure --prefix="/usr"
make -j && make all install
```

Then install NCCL
```
cd $HOME && git clone --recurse-submodules -b byteprofile https://github.com/joapolarbear/nccl.git
rm -rf /usr/include/nccl.h
cd $HOME/nccl && make -j src.build && make pkg.txz.build
mkdir -p $HOME/nccl
tar -Jxf ./build/pkg/txz/nccl*.txz -C $HOME/nccl/ --strip-components 1
echo "$HOME/nccl/lib" >> /etc/ld.so.conf.d/nvidia.conf
ldconfig && ln -sf $HOME/nccl/include/* /usr/include/
```

And install Horovod
```
cd $HOME && git clone --recurse-submodules -b b_v0.21.0 https://github.com/joapolarbear/horovod
cd $HOME/horovod && python3 setup.py sdist
pip3 install cloudpickle psutil pyyaml cffi==1.4.0 pycparser
HOROVOD_NCCL_HOME=$HOME/nccl \
HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL \
HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 \
pip3 install --no-cache-dir dist/horovod*
cp -r $HOME/horovod/examples $HOME/horovod_examples 
```