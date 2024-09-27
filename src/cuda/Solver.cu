#include "cuda/Solver.cuh"

#include "cuda/SolverKernels.cuh"

namespace mhd {

void Solver::calcJacobian(cudaStream_t& stream,
                          const GpuComplexBuffer2D& leftField,
                          const GpuComplexBuffer2D& rightField) {
    _caller.call(stream, DealaliasingDiffByX_kernel, leftField.data(),
                 complexBuffer().data(), _configs._gridLength,
                 _configs._dealWN);
    _transformator.inverse(stream, complexBuffer(), doubleBufferA());

    _caller.call(stream, DealaliasingDiffByY_kernel, rightField.data(),
                 complexBuffer().data(), _configs._gridLength,
                 _configs._dealWN);
    _transformator.inverse(stream, complexBuffer(), doubleBufferB());

    _caller.callFull(stream, JacobianFirstPart_kernel, doubleBufferA().data(),
                     doubleBufferB().data(), doubleBufferC().data(),
                     _configs._gridLength, _configs._lambda);

    _caller.call(stream, DealaliasingDiffByY_kernel, leftField.data(),
                 complexBuffer().data(), _configs._gridLength,
                 _configs._dealWN);
    _transformator.inverse(stream, complexBuffer(), doubleBufferA());

    _caller.call(stream, DealaliasingDiffByX_kernel, rightField.data(),
                 complexBuffer().data(), _configs._gridLength,
                 _configs._dealWN);
    _transformator.inverse(stream, complexBuffer(), doubleBufferB());

    _caller.callFull(stream, JacobianSecondPart_kernel, doubleBufferA().data(),
                     doubleBufferB().data(), doubleBufferC().data(),
                     _configs._gridLength, _configs._lambda);

    _transformator.forward(stream, doubleBufferC(), complexBuffer());
    _caller.call(stream, Dealaliasing_kernel, complexBuffer().data(),
                 _configs._gridLength, _configs._dealWN);
}

Solver::Solver(const mhd::Configs& configs) : Helper(configs) {}

void Solver::calcKineticRigthPart() {
    calcJacobian(_stream1, stream(), vorticity());
    _caller.call(_stream1, FirstRigthPart_kernel, vorticity().data(),
                 complexBuffer().data(), rightPart().data(),
                 _configs._gridLength, _configs._nu);

    calcJacobian(_stream1, potential(), current());
    _caller.call(_stream1, SecondRigthPart_kernel, complexBuffer().data(),
                 rightPart().data(), _configs._gridLength);
}

void Solver::calcMagneticRightPart() {
    calcJacobian(_stream1, stream(), potential());
    _caller.call(_stream1, ThirdRigthPart_kernel, potential().data(),
                 complexBuffer().data(), rightPart().data(),
                 _configs._gridLength, _configs._eta);
}

void Solver::timeSchemeKin(double weight) {
    _caller.call(_stream1, TimeScheme_kernel, vorticity().data(),
                 oldVorticity().data(), rightPart().data(),
                 vorticity().length(), _currents.timeStep, weight);
}

void Solver::timeSchemeMag(double weight) {
    _caller.call(_stream1, TimeScheme_kernel, potential().data(),
                 oldPotential().data(), rightPart().data(),
                 potential().length(), _currents.timeStep, weight);
}

};  // namespace mhd