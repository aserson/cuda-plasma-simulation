#include "Solver.cuh"

#include "SolverKernels.cuh"

namespace mhd {

void Solver::calcJacobian(const GpuComplexBuffer2D& leftField,
                          const GpuComplexBuffer2D& rightField) {
    _caller.call(DealaliasingDiffByX_kernel, leftField.data(),
                 ComplexBuffer().data(), _configs._gridLength,
                 _configs._dealWN);
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());

    _caller.call(DealaliasingDiffByY_kernel, rightField.data(),
                 ComplexBuffer().data(), _configs._gridLength,
                 _configs._dealWN);
    _transformator.inverse(ComplexBuffer(), DoubleBufferB());

    _caller.callFull(JacobianFirstPart_kernel, DoubleBufferA().data(),
                     DoubleBufferB().data(), DoubleBufferC().data(),
                     _configs._gridLength, _configs._lambda);

    _caller.call(DealaliasingDiffByY_kernel, leftField.data(),
                 ComplexBuffer().data(), _configs._gridLength,
                 _configs._dealWN);
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());

    _caller.call(DealaliasingDiffByX_kernel, rightField.data(),
                 ComplexBuffer().data(), _configs._gridLength,
                 _configs._dealWN);
    _transformator.inverse(ComplexBuffer(), DoubleBufferB());

    _caller.callFull(JacobianSecondPart_kernel, DoubleBufferA().data(),
                     DoubleBufferB().data(), DoubleBufferC().data(),
                     _configs._gridLength, _configs._lambda);

    _transformator.forward(DoubleBufferC(), ComplexBuffer());
    _caller.call(Dealaliasing_kernel, ComplexBuffer().data(),
                 _configs._gridLength, _configs._dealWN);
}

Solver::Solver(const mhd::Configs& configs) : Helper(configs) {}

void Solver::calcKineticRigthPart() {
    calcJacobian(Stream(), Vorticity());
    _caller.call(FirstRigthPart_kernel, Vorticity().data(),
                 ComplexBuffer().data(), RightPart().data(),
                 _configs._gridLength, _configs._nu);

    calcJacobian(Potential(), Current());
    _caller.call(SecondRigthPart_kernel, ComplexBuffer().data(),
                 RightPart().data(), _configs._gridLength);
}

void Solver::calcMagneticRightPart() {
    calcJacobian(Stream(), Potential());
    _caller.call(ThirdRigthPart_kernel, Potential().data(),
                 ComplexBuffer().data(), RightPart().data(),
                 _configs._gridLength, _configs._eta);
}

void Solver::timeSchemeKin(double weight) {
    _caller.call(TimeScheme_kernel, Vorticity().data(), OldVorticity().data(),
                 RightPart().data(), Vorticity().length(), _currents.timeStep,
                 weight);
}

void Solver::timeSchemeMag(double weight) {
    _caller.call(TimeScheme_kernel, Potential().data(), OldPotential().data(),
                 RightPart().data(), Potential().length(), _currents.timeStep,
                 weight);
}

};  // namespace mhd