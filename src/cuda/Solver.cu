#include "Solver.cuh"

#include "KernelCaller.cuh"
#include "SolverKernels.cuh"

namespace mhd {

void Solver::calcJacobian(const GpuComplexBuffer2D& leftField,
                          const GpuComplexBuffer2D& rightField) {
    CallKernel(DealaliasingDiffByX_kernel, leftField.data(),
               _complexBuffer.data());
    _transformator.inverse(_complexBuffer, _doubleBufferA);

    CallKernel(DealaliasingDiffByY_kernel, rightField.data(),
               _complexBuffer.data());
    _transformator.inverse(_complexBuffer, _doubleBufferB);

    CallKernelFull(JacobianFirstPart_kernel, _doubleBufferA.data(),
                   _doubleBufferB.data(), _doubleBufferC.data());

    CallKernel(DealaliasingDiffByY_kernel, leftField.data(),
               _complexBuffer.data());
    _transformator.inverse(_complexBuffer, _doubleBufferA);

    CallKernel(DealaliasingDiffByX_kernel, rightField.data(),
               _complexBuffer.data());
    _transformator.inverse(_complexBuffer, _doubleBufferB);

    CallKernelFull(JacobianSecondPart_kernel, _doubleBufferA.data(),
                   _doubleBufferB.data(), _doubleBufferC.data());

    _transformator.forward(_doubleBufferC, _complexBuffer);
    CallKernel(Dealaliasing_kernel, _complexBuffer.data());
}

void Solver::calcKineticRigthPart() {
    calcJacobian(_stream, _vorticity);
    CallKernel(FirstRigthPart_kernel, _vorticity.data(), _complexBuffer.data(),
               _rightPart.data());

    calcJacobian(_potential, _current);
    CallKernel(SecondRigthPart_kernel, _complexBuffer.data(),
               _rightPart.data());
}

void Solver::calcMagneticRightPart() {
    calcJacobian(_stream, _potential);
    CallKernel(ThirdRigthPart_kernel, _potential.data(), _complexBuffer.data(),
               _rightPart.data());
}

void Solver::timeSchemeKin(double weight) {
    CallKernel(TimeScheme_kernel, _vorticity.data(), _oldVorticity.data(),
               _rightPart.data(), _vorticity.length(), _params.timeStep,
               weight);
}

void Solver::timeSchemeMag(double weight) {
    CallKernel(TimeScheme_kernel, _potential.data(), _oldPotential.data(),
               _rightPart.data(), _potential.length(), _params.timeStep,
               weight);
}

};  // namespace mhd