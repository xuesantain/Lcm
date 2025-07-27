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
    const uint32_t VECTOR_SIZE = 8; // 向量化计算的基本单位
    uint32_t baseTile;
    
    // 基于数据类型优化基础tile大小
    switch (dtypeSize) {
        case 1: baseTile = 4096; break; // int8可以处理更多元素
        case 2: baseTile = 2048; break; // int16
        case 4: baseTile = 1024; break; // int32
        case 8: baseTile = 512;  break; // int64
        default: baseTile = 1024; break;
    }
    
    // 确保tile大小不超过总大小
    uint32_t adaptedTile = (baseTile > totalSize) ? totalSize : baseTile;
    adaptedTile = (adaptedTile == 0) ? VECTOR_SIZE : adaptedTile;
    
    // 确保tile大小是向量大小的倍数，以支持向量化计算
    if (adaptedTile >= VECTOR_SIZE) {
        adaptedTile = (adaptedTile / VECTOR_SIZE) * VECTOR_SIZE;
    }
    
    // 如果调整后的tile太小，至少保证有一个向量的大小
    if (adaptedTile < VECTOR_SIZE && totalSize >= VECTOR_SIZE) {
        adaptedTile = VECTOR_SIZE;
    }
    
    // 内存对齐优化
    if (totalSize > 1 && dtypeSize > 0) {
        uint32_t tileBytes = adaptedTile * dtypeSize;
        uint32_t alignedBytes = ceil_div(tileBytes, BYTE_ALIGN) * BYTE_ALIGN;
        uint32_t alignedTile = alignedBytes / dtypeSize;
        
        // 确保对齐后的tile仍然是向量大小的倍数
        if (alignedTile >= VECTOR_SIZE) {
            alignedTile = (alignedTile / VECTOR_SIZE) * VECTOR_SIZE;
        }
        
        // 只有在不超过总大小的情况下才使用对齐后的大小
        if (alignedTile <= totalSize && alignedTile >= VECTOR_SIZE) {
            adaptedTile = alignedTile;
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

// 优化后的广播索引计算 - 预计算常用模式
__aicore__ inline void precomputeBroadcastInfo(uint32_t totalSize, uint32_t inputSize, uint32_t otherSize,
                                              const uint32_t* inputShape, const uint32_t* otherShape, 
                                              uint32_t dims, bool& canUseFastPath, uint32_t& stride) {
    canUseFastPath = false;
    stride = 1;
    
    // 检查是否为简单的广播模式
    if (otherSize == 1) {
        canUseFastPath = true;
        return;
    }
    
    if (otherSize == totalSize) {
        canUseFastPath = true;
        return;
    }
    
    // 检查末尾维度广播
    if (dims > 0 && dims <= 4) {
        bool isTrailingBroadcast = true;
        for (uint32_t i = 0; i < dims - 1; i++) {
            if (inputShape[i] != otherShape[i]) {
                isTrailingBroadcast = false;
                break;
            }
        }
        if (isTrailingBroadcast && otherShape[dims-1] == 1) {
            canUseFastPath = true;
            stride = inputShape[dims-1];
        }
    }
}

__aicore__ inline uint32_t getBroadcastIndexFast(uint32_t outputIndex, uint32_t inputSize, 
                                                uint32_t totalSize, uint32_t stride, bool isTrailing) {
    if (inputSize == 1) return 0;
    if (inputSize >= totalSize) return outputIndex;
    
    if (isTrailing) {
        return (outputIndex / stride) * (inputSize == 1 ? 0 : 1);
    }
    
    return outputIndex % inputSize;
}

// 声明 scalar_binary_gcd 函数，移动到使用之前
template<typename T>
__aicore__ inline T scalar_binary_gcd(T a, T b) {
    if (a == 0) return b > 0 ? b : -b;
    if (b == 0) return a > 0 ? a : -a;
    
    using UnsignedT = typename std::make_unsigned<T>::type;
    UnsignedT ua = static_cast<UnsignedT>(a > 0 ? a : -a);
    UnsignedT ub = static_cast<UnsignedT>(b > 0 ? b : -b);
    
    uint32_t shift = 0;
    while (((ua | ub) & 1) == 0) {
        ua >>= 1;
        ub >>= 1;
        shift++;
    }
    
    while ((ua & 1) == 0) ua >>= 1;
    
    do {
        while ((ub & 1) == 0) ub >>= 1;
        if (ua > ub) {
            UnsignedT temp = ua;
            ua = ub;
            ub = temp;
        }
        ub -= ua;
    } while (ub != 0);
    
    return static_cast<T>(ua << shift);
}

// 向量化的二进制GCD算法 - Stein算法
template<typename T>
__aicore__ inline void vectorized_binary_gcd(const LocalTensor<T>& a_vec, const LocalTensor<T>& b_vec, 
                                            LocalTensor<T>& result_vec, uint32_t length) {
    // 为小数据类型使用向量化路径
    if (sizeof(T) <= 4 && length >= 8) {
        // 批量处理8个元素
        uint32_t simd_length = (length / 8) * 8;
        
        #pragma unroll 4
        for (uint32_t i = 0; i < simd_length; i += 8) {
            // 向量化二进制GCD计算
            for (uint32_t j = 0; j < 8; ++j) {
                T a = a_vec.GetValue(i + j);
                T b = b_vec.GetValue(i + j);
                
                if (a == 0) {
                    result_vec.SetValue(i + j, b > 0 ? b : -b);
                    continue;
                }
                if (b == 0) {
                    result_vec.SetValue(i + j, a > 0 ? a : -b);
                    continue;
                }
                
                // 转为无符号数进行计算
                using UnsignedT = typename std::make_unsigned<T>::type;
                UnsignedT ua = static_cast<UnsignedT>(a > 0 ? a : -a);
                UnsignedT ub = static_cast<UnsignedT>(b > 0 ? b : -b);
                
                // 二进制GCD - 减少除法运算
                uint32_t shift = 0;
                while (((ua | ub) & 1) == 0) {
                    ua >>= 1;
                    ub >>= 1;
                    shift++;
                }
                
                while ((ua & 1) == 0) ua >>= 1;
                
                do {
                    while ((ub & 1) == 0) ub >>= 1;
                    if (ua > ub) {
                        UnsignedT temp = ua;
                        ua = ub;
                        ub = temp;
                    }
                    ub -= ua;
                } while (ub != 0);
                
                result_vec.SetValue(i + j, static_cast<T>(ua << shift));
            }
        }
        
        // 处理剩余元素
        for (uint32_t i = simd_length; i < length; ++i) {
            T a = a_vec.GetValue(i);
            T b = b_vec.GetValue(i);
            result_vec.SetValue(i, scalar_binary_gcd(a, b));
        }
    } else {
        // 标量fallback
        for (uint32_t i = 0; i < length; ++i) {
            T a = a_vec.GetValue(i);
            T b = b_vec.GetValue(i);
            result_vec.SetValue(i, scalar_binary_gcd(a, b));
        }
    }
}

// 向量化LCM计算
template<typename T>
__aicore__ inline void vectorized_lcm_compute(const LocalTensor<T>& a_vec, const LocalTensor<T>& b_vec,
                                             LocalTensor<T>& result_vec, uint32_t length) {
    // 创建临时tensor用于存储GCD结果
    LocalTensor<T> gcd_vec = result_vec; // 复用result_vec的内存
    
    // 首先计算向量化GCD
    vectorized_binary_gcd(a_vec, b_vec, gcd_vec, length);
    
    // 然后计算LCM = |a * b| / gcd(a, b)
    if (sizeof(T) <= 4 && length >= 8) {
        uint32_t simd_length = (length / 8) * 8;
        
        #pragma unroll 4
        for (uint32_t i = 0; i < simd_length; i += 8) {
            for (uint32_t j = 0; j < 8; ++j) {
                T a = a_vec.GetValue(i + j);
                T b = b_vec.GetValue(i + j);
                T gcd_val = gcd_vec.GetValue(i + j);
                
                if (a == 0 || b == 0 || gcd_val == 0) {
                    result_vec.SetValue(i + j, static_cast<T>(0));
                    continue;
                }
                
                // 使用更安全的LCM计算：(a/gcd) * b
                using UnsignedT = typename std::make_unsigned<T>::type;
                UnsignedT ua = static_cast<UnsignedT>(a > 0 ? a : -a);
                UnsignedT ub = static_cast<UnsignedT>(b > 0 ? b : -b);
                UnsignedT ugcd = static_cast<UnsignedT>(gcd_val > 0 ? gcd_val : -gcd_val);
                
                UnsignedT lcm_result = (ua / ugcd) * ub;
                result_vec.SetValue(i + j, static_cast<T>(lcm_result));
            }
        }
        
        // 处理剩余元素
        for (uint32_t i = simd_length; i < length; ++i) {
            T a = a_vec.GetValue(i);
            T b = b_vec.GetValue(i);
            T gcd_val = gcd_vec.GetValue(i);
            
            if (a == 0 || b == 0 || gcd_val == 0) {
                result_vec.SetValue(i, static_cast<T>(0));
                continue;
            }
            
            using UnsignedT = typename std::make_unsigned<T>::type;
            UnsignedT ua = static_cast<UnsignedT>(a > 0 ? a : -a);
            UnsignedT ub = static_cast<UnsignedT>(b > 0 ? b : -b);
            UnsignedT ugcd = static_cast<UnsignedT>(gcd_val > 0 ? gcd_val : -gcd_val);
            
            UnsignedT lcm_result = (ua / ugcd) * ub;
            result_vec.SetValue(i, static_cast<T>(lcm_result));
        }
    } else {
        // 64位数据类型或小长度的标量处理
        for (uint32_t i = 0; i < length; ++i) {
            T a = a_vec.GetValue(i);
            T b = b_vec.GetValue(i);
            T gcd_val = gcd_vec.GetValue(i);
            
            if (a == 0 || b == 0 || gcd_val == 0) {
                result_vec.SetValue(i, static_cast<T>(0));
                continue;
            }
            
            // 对于64位数据，仍然使用标量计算但优化算法
            result_vec.SetValue(i, computeLcmBySize<T>(a, b, sizeof(T)));
        }
    }
}

// 安全的 Duplicate 函数，处理不支持的类型
template<typename T>
__aicore__ inline void SafeDuplicate(LocalTensor<T>& tensor, T value, uint32_t length) {
    // 检查类型是否被 Duplicate API 支持
    // 支持的类型: half, bfloat16_t, int16_t, uint16_t, int32_t, uint32_t, float
    if (std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value || 
        std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value || 
        std::is_same<T, float>::value) {
        // 对于支持的类型，使用 Duplicate API
        Duplicate(tensor, value, length);
    } else {
        // 对于不支持的类型(int8_t, int64_t等)，使用循环手动填充
        for (uint32_t i = 0; i < length; i++) {
            tensor.SetValue(i, value);
        }
    }
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
            CopyInOptimized(i);
            ComputeVectorized(i);
            CopyOutOptimized(i);
        }
    }

private:
    __aicore__ inline void CopyInOptimized(uint32_t progress) {
        LocalTensor<DTYPE> inputLocal = inputQueue.AllocTensor<DTYPE>();
        LocalTensor<DTYPE> otherLocal = otherQueue.AllocTensor<DTYPE>();
        
        uint32_t localOffset = progress * tileLength;
        uint32_t globalOffset = this->coreOffset + localOffset;
        uint32_t length = (localOffset + tileLength > this->coreSize) ? (this->coreSize - localOffset) : tileLength;
        
        // 预计算广播信息
        bool canUseFastBroadcast = false;
        uint32_t broadcastStride = 1;
        precomputeBroadcastInfo(totalSize, otherSize, totalSize, 
                               this->inputShape, this->otherShape, shapeDims,
                               canUseFastBroadcast, broadcastStride);
        
        // 优化的input复制 - 连续内存访问
        if (inputSize == totalSize && length >= 32) {
            // 使用向量化复制
            uint32_t vectorLength = (length / 32) * 32;
            if (vectorLength > 0) {
                DataCopy(inputLocal, inputGm[globalOffset], vectorLength);
            }
            // 处理剩余元素
            for (uint32_t j = vectorLength; j < length; j++) {
                inputLocal.SetValue(j, inputGm.GetValue(globalOffset + j));
            }
        } else {
            // fallback到逐元素复制
            for (uint32_t j = 0; j < length; j++) {
                uint32_t inputIdx = globalOffset + j;
                inputLocal.SetValue(j, inputGm.GetValue(inputIdx));
            }
        }
        
        // 优化的other复制
        if (!needBroadcastOther && otherSize == totalSize && length >= 32) {
            // 直接向量化复制
            uint32_t vectorLength = (length / 32) * 32;
            if (vectorLength > 0) {
                DataCopy(otherLocal, otherGm[globalOffset], vectorLength);
            }
            for (uint32_t j = vectorLength; j < length; j++) {
                otherLocal.SetValue(j, otherGm.GetValue(globalOffset + j));
            }
        } else if (needBroadcastOther && canUseFastBroadcast) {
            // 快速广播路径
            if (otherSize == 1) {
                // 标量广播 - 使用安全的 Duplicate
                DTYPE broadcastValue = otherGm.GetValue(0);
                SafeDuplicate(otherLocal, broadcastValue, length);
            } else {
                // 使用快速索引计算
                #pragma unroll 8
                for (uint32_t j = 0; j < length; j++) {
                    uint32_t outputIdx = globalOffset + j;
                    uint32_t otherIdx = getBroadcastIndexFast(outputIdx, otherSize, totalSize, 
                                                            broadcastStride, true);
                    otherLocal.SetValue(j, otherGm.GetValue(otherIdx));
                }
            }
        } else {
            // 复杂广播的fallback路径
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
    
    __aicore__ inline void ComputeVectorized(uint32_t progress) {
        LocalTensor<DTYPE> inputLocal = inputQueue.DeQue<DTYPE>();
        LocalTensor<DTYPE> otherLocal = otherQueue.DeQue<DTYPE>();
        LocalTensor<DTYPE> outputLocal = outputQueue.AllocTensor<DTYPE>();
        
        uint32_t localOffset = progress * tileLength;
        uint32_t length = (localOffset + tileLength > this->coreSize) ? (this->coreSize - localOffset) : tileLength;
        
        // 检查是否可以使用向量化路径
        bool canVectorize = (length >= 8) && (sizeof(DTYPE) <= 4);
        
        if (canVectorize) {
            // 使用向量化LCM计算
            vectorized_lcm_compute<DTYPE>(inputLocal, otherLocal, outputLocal, length);
        } else {
            // 优化的标量计算路径
            #pragma unroll 8
            for (uint32_t i = 0; i < length; i++) {
                DTYPE a = inputLocal.GetValue(i);
                DTYPE b = otherLocal.GetValue(i);
                
                DTYPE result;
                if (a == 0 || b == 0) {
                    result = static_cast<DTYPE>(0);
                } else {
                    // 使用优化的二进制GCD算法
                    DTYPE gcd_val = scalar_binary_gcd<DTYPE>(a, b);
                    if (gcd_val != 0) {
                        using UnsignedT = typename std::make_unsigned<DTYPE>::type;
                        UnsignedT ua = static_cast<UnsignedT>(a > 0 ? a : -a);
                        UnsignedT ub = static_cast<UnsignedT>(b > 0 ? b : -b);
                        UnsignedT ugcd = static_cast<UnsignedT>(gcd_val > 0 ? gcd_val : -gcd_val);
                        
                        result = static_cast<DTYPE>((ua / ugcd) * ub);
                    } else {
                        result = static_cast<DTYPE>(0);
                    }
                }
                
                outputLocal.SetValue(i, result);
            }
        }
        
        inputQueue.FreeTensor(inputLocal);
        otherQueue.FreeTensor(otherLocal);
        outputQueue.EnQue(outputLocal);
    }
    
    __aicore__ inline void CopyOutOptimized(uint32_t progress) {
        LocalTensor<DTYPE> outputLocal = outputQueue.DeQue<DTYPE>();
        
        uint32_t localOffset = progress * tileLength;
        uint32_t globalOffset = this->coreOffset + localOffset;
        uint32_t length = (localOffset + tileLength > this->coreSize) ? (this->coreSize - localOffset) : tileLength;

        // 优化输出复制 - 向量化写入
        if (totalSize == inputSize && length >= 32) {
            uint32_t vectorLength = (length / 32) * 32;
            if (vectorLength > 0) {
                DataCopy(outputGm[globalOffset], outputLocal, vectorLength);
            }
            // 处理剩余元素
            for (uint32_t i = vectorLength; i < length; i++) {
                outputGm.SetValue(globalOffset + i, outputLocal.GetValue(i));
            }
        } else {
            // fallback路径
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