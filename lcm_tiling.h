#ifndef LCM_TILING_H
#define LCM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalSize);
TILING_DATA_FIELD_DEF(uint32_t, inputSize);
TILING_DATA_FIELD_DEF(uint32_t, otherSize);
TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
TILING_DATA_FIELD_DEF(uint32_t, needBroadcastInput);
TILING_DATA_FIELD_DEF(uint32_t, needBroadcastOther);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);

TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreSize);

TILING_DATA_FIELD_DEF(uint32_t, shapeDims);
TILING_DATA_FIELD_DEF(uint32_t, inputShape0);
TILING_DATA_FIELD_DEF(uint32_t, inputShape1);
TILING_DATA_FIELD_DEF(uint32_t, inputShape2);
TILING_DATA_FIELD_DEF(uint32_t, inputShape3);
TILING_DATA_FIELD_DEF(uint32_t, otherShape0);
TILING_DATA_FIELD_DEF(uint32_t, otherShape1);
TILING_DATA_FIELD_DEF(uint32_t, otherShape2);
TILING_DATA_FIELD_DEF(uint32_t, otherShape3);
TILING_DATA_FIELD_DEF(uint32_t, outputShape0);
TILING_DATA_FIELD_DEF(uint32_t, outputShape1);
TILING_DATA_FIELD_DEF(uint32_t, outputShape2);
TILING_DATA_FIELD_DEF(uint32_t, outputShape3);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Lcm, TilingData)
} // namespace optiling
#endif // LCM_TILING_H 