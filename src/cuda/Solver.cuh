#pragma once

#include "Buffers.cuh"
#include "Helper.cuh"

namespace mhd {
class Solver : public mhd::Helper {
private:
    void calcJacobian(const GpuComplexBuffer2D& leftField,
                      const GpuComplexBuffer2D& rightField);

public:
    void calcKineticRigthPart();
    void calcMagneticRightPart();

    void timeSchemeKin(double weight = 1.0);
    void timeSchemeMag(double weight = 1.0);
};
}  // namespace mhd