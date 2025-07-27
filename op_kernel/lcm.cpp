#include "kernel_operator.h"
#include <type_traits>
#include <climits>

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BYTE_ALIGN = 32;

using namespace AscendC;

__aicore__ inline uint32_t ceil_div(uint32_t a, uint32_t b) {
    if (b == 0) return a;
    return (a + b - 1) / b;
}

struct WorkspaceConfig {
    bool enableDataCopyPad;
    bool enableVectorization;
    uint32_t maxTileSize;
    uint32_t coreCount;
};

template<typename T>
__aicore__ inline T safe_gcd_unsigned(T a, T b) {
    while (b != 0) {
        T temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

template<typename T>
__aicore__ inline T safe_gcd_direct(T a, T b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    
    if (a == 0) return b;
    if (b == 0) return a;
    
    while (b != 0) {
        T temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

__aicore__ inline uint8_t safe_lcm_uint8(uint8_t a, uint8_t b) {
    if (a == 0 || b == 0) return 0;
    
    uint32_t a32 = static_cast<uint32_t>(a);
    uint32_t b32 = static_cast<uint32_t>(b);
    
    uint32_t g = safe_gcd_unsigned<uint32_t>(a32, b32);
    if (g == 0) return 0;
    
    uint32_t result = (a32 / g) * b32;
    
    return static_cast<uint8_t>(result);
}

__aicore__ inline int8_t safe_lcm_int8(int8_t a, int8_t b) {
    if (a == 0 || b == 0) return 0;
    
    int32_t a32, b32;
    if (a == -128) {
        a32 = 128;
    } else {
        a32 = static_cast<int32_t>(a < 0 ? -a : a);
    }
    
    if (b == -128) {
        b32 = 128;
    } else {
        b32 = static_cast<int32_t>(b < 0 ? -b : b);
    }
    
    int32_t g = safe_gcd_direct<int32_t>(a32, b32);
    if (g == 0) return 0;
    
    int32_t result = (a32 / g) * b32;
    
    return static_cast<int8_t>(result);
}

__aicore__ inline int16_t safe_lcm_int16(int16_t a, int16_t b) {
    if (a == 0 || b == 0) return 0;
    
    int32_t a32, b32;
    if (a == -32768) {
        a32 = 32768;
    } else {
        a32 = static_cast<int32_t>(a < 0 ? -a : a);
    }
    
    if (b == -32768) {
        b32 = 32768;
    } else {
        b32 = static_cast<int32_t>(b < 0 ? -b : b);
    }
    
    int32_t g = safe_gcd_direct<int32_t>(a32, b32);
    if (g == 0) return 0;
    
    int32_t result = (a32 / g) * b32;
    
    return static_cast<int16_t>(result);
}

__aicore__ inline int32_t safe_lcm_int32(int32_t a, int32_t b) {
    if (a == 0 || b == 0) return 0;
    
    constexpr int32_t INT32_MIN_VAL = (-2147483647 - 1);
    int64_t a64, b64;
    
    if (a == INT32_MIN_VAL) {
        a64 = static_cast<int64_t>(2147483648LL);
    } else {
        a64 = static_cast<int64_t>(a < 0 ? -static_cast<int64_t>(a) : a);
    }
    
    if (b == INT32_MIN_VAL) {
        b64 = static_cast<int64_t>(2147483648LL);
    } else {
        b64 = static_cast<int64_t>(b < 0 ? -static_cast<int64_t>(b) : b);
    }
    
    int64_t g = safe_gcd_direct<int64_t>(a64, b64);
    if (g == 0) return 0;
    
    int64_t result = (a64 / g) * b64;
    
    return static_cast<int32_t>(result);
}

__aicore__ inline int64_t safe_lcm_int64(int64_t a, int64_t b) {
    if (a == 0 || b == 0) return 0;
    
    constexpr int64_t INT64_MIN_VAL = (-9223372036854775807LL - 1);
    
    uint64_t ua, ub;
    
    if (a == INT64_MIN_VAL) {
        ua = static_cast<uint64_t>(9223372036854775808ULL);
    } else {
        ua = static_cast<uint64_t>(a < 0 ? -a : a);
    }
    
    if (b == INT64_MIN_VAL) {
        ub = static_cast<uint64_t>(9223372036854775808ULL);
    } else {
        ub = static_cast<uint64_t>(b < 0 ? -b : b);
    }
    
    uint64_t g = safe_gcd_unsigned<uint64_t>(ua, ub);
    if (g == 0) return 0;
    
    uint64_t result = (ua / g) * ub;
    
    int64_t final_result = static_cast<int64_t>(result);
    return (final_result < 0) ? -final_result : final_result;
}

__aicore__ inline uint32_t getBroadcastIndex(uint32_t outputIndex, uint32_t inputSize, uint32_t totalSize) {
    if (inputSize == 0) return 0;
    if (inputSize == 1) return 0;
    if (inputSize >= totalSize) return outputIndex % inputSize;
    
    return outputIndex % inputSize;
}

__aicore__ inline uint32_t getBroadcastIndexMultiDim(uint32_t outputIndex, 
                                                    uint32_t inputSize, 
                                                    uint32_t totalSize, 
                                                    const uint32_t* outputShape, 
                                                    const uint32_t* inputShape, 
                                                    uint32_t dims) {
    if (inputSize == 0) return 0;
    if (inputSize == 1) return 0;
    if (dims == 0 || dims == 1) return outputIndex % inputSize;
    if (inputSize >= totalSize) return outputIndex % inputSize;
    if (outputIndex >= totalSize) return 0;
    
    uint32_t idx = 0;
    uint32_t stride = 1;
    uint32_t tmpOutputIdx = outputIndex;
    
    uint32_t safeDims = (dims > 4) ? 4 : dims;
    
    for (int d = safeDims - 1; d >= 0; --d) {
        if (outputShape[d] == 0 || inputShape[d] == 0) return 0;
        
        uint32_t coord = tmpOutputIdx % outputShape[d];
        
        uint32_t inputCoord;
        if (inputShape[d] == 1) {
            inputCoord = 0;
        } else if (inputShape[d] == outputShape[d]) {
            inputCoord = coord;
        } else {
            inputCoord = coord % inputShape[d];
        }
        
        if (inputCoord >= inputShape[d]) {
            inputCoord = inputShape[d] - 1;
        }
        
        if (idx > UINT32_MAX - inputCoord * stride) {
            return outputIndex % inputSize;
        }
        
        idx += inputCoord * stride;
        
        if (stride > UINT32_MAX / inputShape[d]) {
            return outputIndex % inputSize;
        }
        stride *= inputShape[d];
        
        if (outputShape[d] > 0) {
            tmpOutputIdx /= outputShape[d];
        }
    }
    
    return idx % inputSize;
}

__aicore__ inline uint32_t getTileLength(uint32_t dtypeSize, uint32_t totalSize) {
    uint32_t baseTile;
    switch (dtypeSize) {
        case 1: baseTile = 2048; break;
        case 2: baseTile = 1024; break;
        case 8: baseTile = 256;  break;
        default: baseTile = 512; break;
    }
    
    uint32_t adaptedTile = (baseTile > totalSize) ? totalSize : baseTile;
    
    adaptedTile = (adaptedTile == 0) ? 1 : adaptedTile;
    
    if (totalSize > 1 && dtypeSize > 0) {
        uint32_t tileBytes = adaptedTile * dtypeSize;
        uint32_t alignedBytes = ceil_div(tileBytes, BYTE_ALIGN) * BYTE_ALIGN;
        adaptedTile = alignedBytes / dtypeSize;
        
        if (adaptedTile > totalSize) {
            adaptedTile = totalSize;
        }
    }
    
    return adaptedTile;
}

template<typename DTYPE>
__aicore__ inline DTYPE computeLcmBySize(DTYPE a, DTYPE b, uint32_t dtypeSize) {
    DTYPE result;
    

    
    switch (dtypeSize) {
        case 1:
            if (std::is_unsigned<DTYPE>::value) {
                result = static_cast<DTYPE>(safe_lcm_uint8(static_cast<uint8_t>(a), static_cast<uint8_t>(b)));
            } else {
                result = static_cast<DTYPE>(safe_lcm_int8(static_cast<int8_t>(a), static_cast<int8_t>(b)));
            }
            break;
        case 2:
            result = static_cast<DTYPE>(safe_lcm_int16(static_cast<int16_t>(a), static_cast<int16_t>(b)));
            break;
        case 4:
            result = static_cast<DTYPE>(safe_lcm_int32(static_cast<int32_t>(a), static_cast<int32_t>(b)));
            break;
        case 8:
            result = static_cast<DTYPE>(safe_lcm_int64(static_cast<int64_t>(a), static_cast<int64_t>(b)));
            break;
        default:
            result = static_cast<DTYPE>(safe_lcm_int32(static_cast<int32_t>(a), static_cast<int32_t>(b)));
            break;
    }
    

    
    return result;
}

template<typename DTYPE>
class KernelLcm {
public:
    __aicore__ inline KernelLcm() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace,
                               uint32_t totalSize, uint32_t inputSize, uint32_t otherSize,
                               uint32_t dtypeSize, bool needBroadcastOther,
                               uint32_t shapeDims, const uint32_t* inputShape, 
                               const uint32_t* otherShape, const uint32_t* outputShape,
                               uint32_t usedCoreNum, uint32_t singleCoreSize) {
        

        
        if (totalSize == 0 || inputSize == 0 || otherSize == 0) {
            return;
        }
        
        if (dtypeSize == 0 || (dtypeSize != 1 && dtypeSize != 2 && dtypeSize != 4 && dtypeSize != 8)) {
            return;
        }
        
        if (totalSize != inputSize) {
            return;
        }
        
        if (shapeDims > 0) {
            uint32_t expectedTotalSize = 1;
            for (uint32_t d = 0; d < shapeDims && d < 4; d++) {
                if (inputShape[d] == 0) {
                    return;
                }
                expectedTotalSize *= inputShape[d];
                
                if (otherShape[d] != inputShape[d] && otherShape[d] != 1) {
                    return;
                }
            }
            
            if (expectedTotalSize != totalSize) {
                return;
            }
        }
        
        inputGm.SetGlobalBuffer((__gm__ DTYPE*)input, inputSize);
        otherGm.SetGlobalBuffer((__gm__ DTYPE*)other, otherSize);
        outputGm.SetGlobalBuffer((__gm__ DTYPE*)out, totalSize);
        
        this->useWorkspace = (workspace != nullptr);
        if (this->useWorkspace) {
            uint32_t maxTileLength = getTileLength(dtypeSize, totalSize);
            uint32_t workspaceElements = maxTileLength * 4;
            workspaceGm.SetGlobalBuffer((__gm__ DTYPE*)workspace, workspaceElements);
            
            this->workspaceConfig.enableDataCopyPad = true;
            this->workspaceConfig.enableVectorization = (totalSize > 256);
            this->workspaceConfig.maxTileSize = maxTileLength;
            this->workspaceConfig.coreCount = 1;
        } else {
            this->workspaceConfig.enableDataCopyPad = false;
            this->workspaceConfig.enableVectorization = false;
            this->workspaceConfig.maxTileSize = this->tileLength;
            this->workspaceConfig.coreCount = 1;
        }
        
        uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedCoreNum) {
            this->coreOffset = totalSize;
            this->coreSize = 0;
            this->totalSize = 0;
            return;
        }
        
        this->coreOffset = blockIdx * singleCoreSize;        
        if (blockIdx == usedCoreNum - 1) {
            this->coreSize = totalSize - this->coreOffset;
        } else {
            this->coreSize = singleCoreSize;
        }
        
        if (this->coreOffset >= totalSize) {
            this->coreSize = 0;
            this->totalSize = 0;
            return;
        }
        
        if (this->coreOffset + this->coreSize > totalSize) {
            this->coreSize = totalSize - this->coreOffset;
        }
        
        this->totalSize = totalSize;
        this->inputSize = inputSize;
        this->otherSize = otherSize;
        this->dtypeSize = dtypeSize;
        this->needBroadcastOther = needBroadcastOther;
        this->tileLength = getTileLength(dtypeSize, this->coreSize);
        
        pipe.InitBuffer(inputQueue, BUFFER_NUM, this->tileLength * sizeof(DTYPE));
        pipe.InitBuffer(otherQueue, BUFFER_NUM, this->tileLength * sizeof(DTYPE));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, this->tileLength * sizeof(DTYPE));
        
        this->shapeDims = (shapeDims > 4) ? 4 : (shapeDims == 0 ? 1 : shapeDims);
        for (uint32_t i = 0; i < 4; ++i) {
            if (i < this->shapeDims) {
                if (shapeDims == 0) {
                    this->inputShape[i] = 1;
                    this->otherShape[i] = 1;
                } else {
                    this->inputShape[i] = inputShape[i];
                    this->otherShape[i] = otherShape[i];
                }
            } else {
                this->inputShape[i] = 1;
                this->otherShape[i] = 1;
            }
        }
        

    }
    
    __aicore__ inline void Process() {
        if (this->coreSize == 0) {
            return;
        }
        
        uint32_t loopCount = (this->coreSize + tileLength - 1) / tileLength;
        
        for (uint32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress) {
        LocalTensor<DTYPE> inputLocal = inputQueue.AllocTensor<DTYPE>();
        LocalTensor<DTYPE> otherLocal = otherQueue.AllocTensor<DTYPE>();
        
        uint32_t localOffset = progress * tileLength;
        uint32_t globalOffset = this->coreOffset + localOffset;
        uint32_t length = (localOffset + tileLength > this->coreSize) ? (this->coreSize - localOffset) : tileLength;
        
        if (this->workspaceConfig.enableDataCopyPad && inputSize == totalSize && 
            length * sizeof(DTYPE) >= BYTE_ALIGN) {
            uint32_t elementBytes = length * sizeof(DTYPE);
            uint32_t alignedBytes = ceil_div(elementBytes, BYTE_ALIGN) * BYTE_ALIGN;
            uint8_t rpad = (alignedBytes - elementBytes) / sizeof(DTYPE);
            
            AscendC::DataCopyExtParams copyParams = {1, elementBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<DTYPE> padParams = {false, 0, rpad, 0};
            AscendC::DataCopyPad<DTYPE>(inputLocal, inputGm[globalOffset], copyParams, padParams);
        } else {
            for (uint32_t j = 0; j < length; j++) {
                uint32_t inputIdx = globalOffset + j;
                inputLocal.SetValue(j, inputGm.GetValue(inputIdx));
            }
        }
        
        if (this->workspaceConfig.enableDataCopyPad && !needBroadcastOther && 
            otherSize == totalSize && length * sizeof(DTYPE) >= BYTE_ALIGN) {
            uint32_t elementBytes = length * sizeof(DTYPE);
            uint32_t alignedBytes = ceil_div(elementBytes, BYTE_ALIGN) * BYTE_ALIGN;
            uint8_t rpad = (alignedBytes - elementBytes) / sizeof(DTYPE);
            
            AscendC::DataCopyExtParams copyParams = {1, elementBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<DTYPE> padParams = {false, 0, rpad, 0};
            AscendC::DataCopyPad<DTYPE>(otherLocal, otherGm[globalOffset], copyParams, padParams);
        } else {
            for (uint32_t j = 0; j < length; j++) {
                uint32_t outputIdx = globalOffset + j;
                uint32_t otherIdx;
                
                if (needBroadcastOther) {
                    otherIdx = getBroadcastIndexMultiDim(outputIdx, otherSize, totalSize,
                                                       this->inputShape, this->otherShape, shapeDims);
                    if (otherIdx >= otherSize) {
                        otherIdx = outputIdx % otherSize;
                    }
                } else {
                    otherIdx = outputIdx;
                }
                
                otherLocal.SetValue(j, otherGm.GetValue(otherIdx));
            }
        }
        
        inputQueue.EnQue(inputLocal);
        otherQueue.EnQue(otherLocal);
    }
    
    __aicore__ inline void Compute(uint32_t progress) {
        LocalTensor<DTYPE> inputLocal = inputQueue.DeQue<DTYPE>();
        LocalTensor<DTYPE> otherLocal = otherQueue.DeQue<DTYPE>();
        LocalTensor<DTYPE> outputLocal = outputQueue.AllocTensor<DTYPE>();
        
        uint32_t localOffset = progress * tileLength;
        uint32_t length = (localOffset + tileLength > this->coreSize) ? (this->coreSize - localOffset) : tileLength;
        
        #pragma unroll 8
        for (uint32_t i = 0; i < length; i++) {
            DTYPE a = inputLocal.GetValue(i);
            DTYPE b = otherLocal.GetValue(i);
            
                    DTYPE result;
        if (a == 0 || b == 0) {
            result = static_cast<DTYPE>(0);
        } else {
            result = computeLcmBySize<DTYPE>(a, b, dtypeSize);
            
            if (dtypeSize == 8) {
                int64_t result64 = static_cast<int64_t>(result);
                if (result64 < 0) {
                    result = static_cast<DTYPE>(-result64);
                }
            }
        }
            
            outputLocal.SetValue(i, result);
        }
        
        inputQueue.FreeTensor(inputLocal);
        otherQueue.FreeTensor(otherLocal);
        outputQueue.EnQue(outputLocal);
    }
    
    __aicore__ inline void CopyOut(uint32_t progress) {
        LocalTensor<DTYPE> outputLocal = outputQueue.DeQue<DTYPE>();
        
        uint32_t localOffset = progress * tileLength;
        uint32_t globalOffset = this->coreOffset + localOffset;
        uint32_t length = (localOffset + tileLength > this->coreSize) ? (this->coreSize - localOffset) : tileLength;

        if (this->workspaceConfig.enableDataCopyPad && totalSize == inputSize && 
            length * sizeof(DTYPE) >= BYTE_ALIGN) {
            uint32_t elementBytes = length * sizeof(DTYPE);
            AscendC::DataCopyExtParams copyParams = {1, elementBytes, 0, 0, 0};
            AscendC::DataCopyPad<DTYPE>(outputGm[globalOffset], outputLocal, copyParams);
        } else {
            for (uint32_t i = 0; i < length; i++) {
                outputGm.SetValue(globalOffset + i, outputLocal.GetValue(i));
            }
        }
        
        outputQueue.FreeTensor(outputLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue, otherQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    GlobalTensor<DTYPE> inputGm, otherGm, outputGm, workspaceGm;
    
    uint32_t totalSize;
    uint32_t inputSize;
    uint32_t otherSize;
    uint32_t dtypeSize;
    uint32_t tileLength;
    bool needBroadcastOther;
    bool useWorkspace;
    
    WorkspaceConfig workspaceConfig;
    
    uint32_t inputShape[4];
    uint32_t otherShape[4];
    uint32_t shapeDims;
    
    uint32_t coreOffset;
    uint32_t coreSize;
};

template<typename DTYPE>
__aicore__ inline void lcm_compute(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace,
                                  uint32_t totalSize, uint32_t inputSize, uint32_t otherSize,
                                  uint32_t dtypeSize, uint32_t needBroadcastOther,
                                  uint32_t shapeDims, const uint32_t* inputShape, 
                                  const uint32_t* otherShape, const uint32_t* outputShape,
                                  uint32_t usedCoreNum, uint32_t singleCoreSize) {
    KernelLcm<DTYPE> op;
    op.Init(input, other, out, workspace, totalSize, inputSize, otherSize, dtypeSize,
            needBroadcastOther != 0, shapeDims, inputShape, otherShape, outputShape,
            usedCoreNum, singleCoreSize);
    op.Process();
}

extern "C" __global__ __aicore__ void lcm(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    uint32_t totalSize = tiling_data.totalSize;
    uint32_t inputSize = tiling_data.inputSize;
    uint32_t otherSize = tiling_data.otherSize;
    uint32_t dtypeSize = tiling_data.dtypeSize;
    uint32_t needBroadcastOther = tiling_data.needBroadcastOther;
    
    uint32_t usedCoreNum = tiling_data.usedCoreNum;
    uint32_t singleCoreSize = tiling_data.singleCoreSize;
    
    uint32_t shapeDims = tiling_data.shapeDims;
    uint32_t inputShape[4], otherShape[4], outputShape[4];
    
    inputShape[0] = tiling_data.inputShape0;
    inputShape[1] = tiling_data.inputShape1;
    inputShape[2] = tiling_data.inputShape2;
    inputShape[3] = tiling_data.inputShape3;
    
    otherShape[0] = tiling_data.otherShape0;
    otherShape[1] = tiling_data.otherShape1;
    otherShape[2] = tiling_data.otherShape2;
    otherShape[3] = tiling_data.otherShape3;
    
    outputShape[0] = inputShape[0];
    outputShape[1] = inputShape[1];
    outputShape[2] = inputShape[2];
    outputShape[3] = inputShape[3];
    

    
    switch (dtypeSize) {
        case 1:
            lcm_compute<int8_t>(input, other, out, workspace, totalSize, inputSize, otherSize, 
                               dtypeSize, needBroadcastOther,
                               shapeDims, inputShape, otherShape, outputShape,
                               usedCoreNum, singleCoreSize);
            break;
        case 2:
            lcm_compute<int16_t>(input, other, out, workspace, totalSize, inputSize, otherSize, 
                                dtypeSize, needBroadcastOther,
                                shapeDims, inputShape, otherShape, outputShape,
                                usedCoreNum, singleCoreSize);
            break;
        case 4:
            lcm_compute<int32_t>(input, other, out, workspace, totalSize, inputSize, otherSize, 
                                dtypeSize, needBroadcastOther,
                                shapeDims, inputShape, otherShape, outputShape,
                                usedCoreNum, singleCoreSize);
            break;
        case 8:
            lcm_compute<int64_t>(input, other, out, workspace, totalSize, inputSize, otherSize, 
                                dtypeSize, needBroadcastOther,
                                shapeDims, inputShape, otherShape, outputShape,
                                usedCoreNum, singleCoreSize);
            break;
        default:
            lcm_compute<int32_t>(input, other, out, workspace, totalSize, inputSize, otherSize, 
                                dtypeSize, needBroadcastOther,
                                shapeDims, inputShape, otherShape, outputShape,
                                usedCoreNum, singleCoreSize);
            break;
    }
}

#ifndef ASCENDC_CPU_DEBUG
void lcm_do(uint32_t blockDim, void *l2ctrl, void *stream, 
            uint8_t *input, uint8_t *other, uint8_t *out,
            uint8_t *workspace, uint8_t *tiling)
{
    lcm<<<blockDim, l2ctrl, stream>>>(input, other, out, workspace, tiling);
}
#endif 