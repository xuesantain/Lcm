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
    
    const uint32_t MAX_CORE_NUM = 32;
    const uint32_t MIN_ELEMENTS_PER_CORE = 32;
    
    uint32_t usedCoreNum = std::min(MAX_CORE_NUM, CeilDiv(totalSize, MIN_ELEMENTS_PER_CORE));
    usedCoreNum = std::max(1u, usedCoreNum);
    
    uint32_t singleCoreSize = totalSize / usedCoreNum;
    
    context->SetBlockDim(usedCoreNum);
    tiling.set_totalSize(totalSize);
    tiling.set_inputSize(inputSize);
    tiling.set_otherSize(otherSize);
    tiling.set_dtypeSize(dtypeSize);
    tiling.set_needBroadcastInput(needBroadcastInput);
    tiling.set_needBroadcastOther(needBroadcastOther);
    tiling.set_tileNum(TILE_NUM);
    
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
    
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    
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
