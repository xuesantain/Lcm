#include "lcm_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>

namespace optiling {
const uint32_t TILE_NUM = 8;

static uint32_t GetDataTypeSize(ge::DataType dataType) {
    switch (dataType) {
        case ge::DT_INT8: return 1;
        case ge::DT_INT16: return 2;
        case ge::DT_INT32: return 4;
        case ge::DT_INT64: return 8;
        default: return 4;
    }
}

static uint32_t CeilDiv(uint32_t a, uint32_t b) {
    if (b == 0) return a;
    return (a + b - 1) / b;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;
    
    auto inputShape = context->GetInputShape(0)->GetOriginShape();
    auto otherShape = context->GetInputShape(1)->GetOriginShape();
    
    ge::DataType inputDataType = context->GetInputDesc(0)->GetDataType();
    uint32_t dtypeSize = GetDataTypeSize(inputDataType);
    
    uint32_t inputSize = inputShape.GetShapeSize();
    uint32_t otherSize = otherShape.GetShapeSize();
    
    int inputRank = inputShape.GetDimNum();
    int otherRank = otherShape.GetDimNum();
    
    for (int i = 0; i < inputRank; ++i) {
        int64_t inputDim = inputShape.GetDim(i);
        
        int otherIdx = i - (inputRank - otherRank);
        int64_t otherDim = 1;
        
        if (otherIdx >= 0 && otherIdx < otherRank) {
            otherDim = otherShape.GetDim(otherIdx);
        }
        
        if (otherDim != 1 && otherDim != inputDim) {
            return ge::GRAPH_FAILED;
        }
    }
    
    uint32_t totalSize = inputSize;
    
    uint32_t needBroadcastInput = 0;
    uint32_t needBroadcastOther = (otherSize != totalSize) ? 1 : 0;
    
    // 优化的核心分配策略
    const uint32_t MAX_CORE_NUM = 32;
    const uint32_t VECTOR_SIZE = 8; // 向量化计算的基本单位
    
    // 基于数据大小和类型动态调整最小处理单元
    uint32_t minElementsPerCore;
    if (dtypeSize <= 2) {
        minElementsPerCore = 1024; // 小数据类型可以处理更多元素
    } else if (dtypeSize == 4) {
        minElementsPerCore = 512;
    } else {
        minElementsPerCore = 256; // 64位数据类型
    }
    
    // 确保每个核心处理的元素数量是向量大小的倍数
    minElementsPerCore = ((minElementsPerCore + VECTOR_SIZE - 1) / VECTOR_SIZE) * VECTOR_SIZE;
    
    // 计算理想的核心数量
    uint32_t idealCoreNum = CeilDiv(totalSize, minElementsPerCore);
    uint32_t usedCoreNum = std::min(MAX_CORE_NUM, idealCoreNum);
    usedCoreNum = std::max(1u, usedCoreNum);
    
    // 优化负载均衡
    uint32_t elementsPerCore = totalSize / usedCoreNum;
    uint32_t remainder = totalSize % usedCoreNum;
    
    // 确保每个核心处理的元素数量是向量大小的倍数
    if (elementsPerCore % VECTOR_SIZE != 0) {
        elementsPerCore = ((elementsPerCore + VECTOR_SIZE - 1) / VECTOR_SIZE) * VECTOR_SIZE;
        // 重新计算核心数量
        usedCoreNum = CeilDiv(totalSize, elementsPerCore);
        usedCoreNum = std::min(MAX_CORE_NUM, usedCoreNum);
        usedCoreNum = std::max(1u, usedCoreNum);
    }
    
    uint32_t singleCoreSize = elementsPerCore;
    
    context->SetBlockDim(usedCoreNum);
    tiling.set_totalSize(totalSize);
    tiling.set_inputSize(inputSize);
    tiling.set_otherSize(otherSize);
    tiling.set_dtypeSize(dtypeSize);
    tiling.set_needBroadcastInput(needBroadcastInput);
    tiling.set_needBroadcastOther(needBroadcastOther);
    
    // 优化tile大小计算
    uint32_t optimalTileNum = TILE_NUM;
    
    // 基于数据特征调整tile数量
    if (needBroadcastOther) {
        // 广播操作需要更小的tile以减少内存压力
        optimalTileNum = std::min(TILE_NUM, 4u);
    } else if (totalSize > 1024 * 1024) {
        // 大数据集使用更多tile进行流水线优化
        optimalTileNum = std::min(TILE_NUM * 2, 16u);
    }
    
    tiling.set_tileNum(optimalTileNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_singleCoreSize(singleCoreSize);
    
    uint32_t shapeDims = std::min(inputRank, 4);
    tiling.set_shapeDims(shapeDims);
    
    uint32_t inputShapes[4] = {1, 1, 1, 1};
    for (int i = 0; i < inputRank && i < 4; i++) {
        inputShapes[i] = static_cast<uint32_t>(inputShape.GetDim(i));
    }
    tiling.set_inputShape0(inputShapes[0]);
    tiling.set_inputShape1(inputShapes[1]);
    tiling.set_inputShape2(inputShapes[2]);
    tiling.set_inputShape3(inputShapes[3]);
    
    uint32_t otherShapes[4] = {1, 1, 1, 1};
    for (int i = 0; i < otherRank && i < 4; i++) {
        otherShapes[i] = static_cast<uint32_t>(otherShape.GetDim(i));
    }
    tiling.set_otherShape0(otherShapes[0]);
    tiling.set_otherShape1(otherShapes[1]);
    tiling.set_otherShape2(otherShapes[2]);
    tiling.set_otherShape3(otherShapes[3]);
    
    tiling.set_outputShape0(inputShapes[0]);
    tiling.set_outputShape1(inputShapes[1]);
    tiling.set_outputShape2(inputShapes[2]);
    tiling.set_outputShape3(inputShapes[3]);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    // 计算工作空间大小 - 为向量化计算预留更多空间
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    uint32_t maxTileSize = 2048; // 基于dtypeSize动态调整
    if (dtypeSize <= 2) {
        maxTileSize = 4096;
    } else if (dtypeSize == 8) {
        maxTileSize = 1024;
    }
    currentWorkspace[0] = maxTileSize * dtypeSize * 4; // 为临时计算预留4倍空间
    
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *input_shape = context->GetInputShape(0);
    const gert::Shape *other_shape = context->GetInputShape(1);
    gert::Shape *output_shape = context->GetOutputShape(0);
    
    int input_rank = input_shape->GetDimNum();
    int other_rank = other_shape->GetDimNum();
    
    for (int i = 0; i < input_rank; ++i) {
        int64_t input_dim = input_shape->GetDim(i);
        
        int other_idx = i - (input_rank - other_rank);
        int64_t other_dim = 1;
        
        if (other_idx >= 0 && other_idx < other_rank) {
            other_dim = other_shape->GetDim(other_idx);
        }
        
        if (other_dim != 1 && other_dim != input_dim) {
            return GRAPH_FAILED;
        }
    }
    
    output_shape->SetDimNum(input_rank);
    for (int i = 0; i < input_rank; ++i) {
        output_shape->SetDim(i, input_shape->GetDim(i));
    }
    
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge


namespace ops {
class Lcm : public OpDef {
public:
    explicit Lcm(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("other")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Lcm);
}
