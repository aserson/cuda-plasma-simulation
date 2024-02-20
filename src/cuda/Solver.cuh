#pragma once

#include "../Configs.h"
#include "Buffers.cuh"
#include "Helper.cuh"

namespace mhd {
class Solver : public Helper {
private:
    void calcJacobian(const GpuComplexBuffer2D& leftField,
                      const GpuComplexBuffer2D& rightField);

public:
    Solver(const mhd::Configs& configs);

    void calcKineticRigthPart();
    void calcMagneticRightPart();

    void timeSchemeKin(double weight = 1.0);
    void timeSchemeMag(double weight = 1.0);
};
}  // namespace mhd