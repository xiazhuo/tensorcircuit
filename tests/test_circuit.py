import sys
import os
from functools import partial
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf

# see https://stackoverflow.com/questions/56307329/how-can-i-parametrize-tests-to-run-with-different-fixtures-in-pytest

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc
from .conftest import tfb


def test_wavefunction():
    qc = tc.Circuit(2)
    qc.apply_double_gate(
        tc.gates.Gate(np.arange(16).reshape(2, 2, 2, 2).astype(np.complex64)), 0, 1
    )
    assert np.real(qc.wavefunction()[0][2]) == 8
    qc = tc.Circuit(2)
    qc.apply_double_gate(
        tc.gates.Gate(np.arange(16).reshape(2, 2, 2, 2).astype(np.complex64)), 1, 0
    )
    qc.wavefunction()
    assert np.real(qc.wavefunction()[0][2]) == 4
    qc = tc.Circuit(2)
    qc.apply_single_gate(
        tc.gates.Gate(np.arange(4).reshape(2, 2).astype(np.complex64)), 0
    )
    qc.wavefunction()
    assert np.real(qc.wavefunction()[0][2]) == 2


def test_basics():
    c = tc.Circuit(2)
    c.x(0)
    assert np.allclose(c.amplitude("10"), np.array(1.0))
    c.CNOT(0, 1)
    assert np.allclose(c.amplitude("11"), np.array(1.0))


def test_measure():
    c = tc.Circuit(3)
    c.H(0)
    c.h(1)
    c.toffoli(0, 1, 2)
    assert c.measure(2)[0] in ["0", "1"]


def test_expectation():
    c = tc.Circuit(2)
    c.H(0)
    assert np.allclose(c.expectation((tc.gates.z(), [0])), 0, atol=1e-7)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_exp1(backend):
    @partial(tc.backend.jit, jit_compile=True)
    def sf():
        c = tc.Circuit(2)
        xx = np.array(
            [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.complex64
        )
        c.exp1(0, 1, unitary=xx, theta=tc.num_to_tensor(0.2))
        s = c.state()
        return s

    @tc.backend.jit
    def s1f():
        c = tc.Circuit(2)
        xx = np.array(
            [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.complex64
        )
        c.exp(0, 1, unitary=xx, theta=tc.num_to_tensor(0.2))
        s1 = c.state()
        return s1

    s = sf()
    s1 = s1f()
    assert np.allclose(s, s1, atol=1e-4)


def test_complex128(highp):
    tc.set_backend("tensorflow")
    tc.set_dtype("complex128")
    c = tc.Circuit(2)
    c.H(1)
    c.rx(0, theta=tc.gates.num_to_tensor(1j))
    c.wavefunction()
    assert np.allclose(c.expectation((tc.gates.z(), [1])), 0)


def test_qcode():
    qcode = """
4
x 0
cnot 0 1
r 2 theta 1.0 alpha 1.57
"""
    c = tc.Circuit.from_qcode(qcode)
    assert c.measure(1)[0] == "1"
    assert c.to_qcode() == qcode[1:]


def universal_ad():
    @tc.backend.jit
    def forward(theta):
        c = tc.Circuit(2)
        c.R(0, theta=theta, alpha=0.5, phi=0.8)
        return tc.backend.real(c.expectation((tc.gates.z(), [0])))

    gg = tc.backend.grad(forward)
    vag = tc.backend.value_and_grad(forward)
    gg = tc.backend.jit(gg)
    vag = tc.backend.jit(vag)
    theta = tc.gates.num_to_tensor(1.0)
    grad1 = gg(theta)
    v2, grad2 = vag(theta)
    assert grad1 == grad2
    return v2, grad2


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_ad(backend):
    # this amazingly shows how to code once and run in very different AD-ML engines
    print(universal_ad())


def test_single_qubit():
    c = tc.Circuit(1)
    c.H(0)
    w = c.state()[0]
    assert np.allclose(w, np.array([1, 1]) / np.sqrt(2), atol=1e-4)


def test_expectation_between_two_states():
    zp = np.array([1.0, 0.0])
    zd = np.array([0.0, 1.0])
    assert tc.expectation((tc.gates.y(), [0]), ket=zp, bra=zd) == 1j

    c = tc.Circuit(3)
    c.H(0)
    c.ry(1, theta=tc.num_to_tensor(0.8))
    c.cnot(1, 2)

    state = c.wavefunction()
    x1z2 = [(tc.gates.x(), [0]), (tc.gates.z(), [1])]
    e1 = c.expectation(*x1z2)
    e2 = tc.expectation(*x1z2, ket=state, bra=state, normalization=True)
    assert np.allclose(e2, e1)

    c = tc.Circuit(3)
    c.H(0)
    c.ry(1, theta=tc.num_to_tensor(0.8 + 0.7j))
    c.cnot(1, 2)

    state = c.wavefunction()
    x1z2 = [(tc.gates.x(), [0]), (tc.gates.z(), [1])]
    e1 = c.expectation(*x1z2) / tc.backend.norm(state) ** 2
    e2 = tc.expectation(*x1z2, ket=state, normalization=True)
    assert np.allclose(e2, e1)

    c = tc.Circuit(2)
    c.X(1)
    s1 = c.state()
    c2 = tc.Circuit(2)
    c2.X(0)
    s2 = c2.state()
    c3 = tc.Circuit(2)
    c3.H(1)
    s3 = c3.state()
    x1x2 = [(tc.gates.x(), [0]), (tc.gates.x(), [1])]
    e = tc.expectation(*x1x2, ket=s1, bra=s2)
    assert np.allclose(e, 1.0)
    e2 = tc.expectation(*x1x2, ket=s3, bra=s2)
    assert np.allclose(e2, 1.0 / np.sqrt(2))


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_any_inputs_state(backend):
    c = tc.Circuit(2, inputs=tc.array_to_tensor(np.array([0.0, 0.0, 0.0, 1.0])))
    c.X(0)
    z0 = c.expectation((tc.gates.z(), [0]))
    assert z0 == 1.0
    c = tc.Circuit(2, inputs=tc.array_to_tensor(np.array([0.0, 0.0, 1.0, 0.0])))
    c.X(0)
    z0 = c.expectation((tc.gates.z(), [0]))
    assert z0 == 1.0
    c = tc.Circuit(2, inputs=tc.array_to_tensor(np.array([1.0, 0.0, 0.0, 0.0])))
    c.X(0)
    z0 = c.expectation((tc.gates.z(), [0]))
    assert z0 == -1.0
    c = tc.Circuit(
        2,
        inputs=tc.array_to_tensor(np.array([1 / np.sqrt(2), 0.0, 1 / np.sqrt(2), 0.0])),
    )
    c.X(0)
    z0 = c.expectation((tc.gates.z(), [0]))
    assert np.allclose(z0, 0.0, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb")])
def test_postselection(backend):
    c = tc.Circuit(3)
    c.H(1)
    c.H(2)
    c.mid_measurement(1, 1)
    c.mid_measurement(2, 1)
    s = c.wavefunction()[0]
    assert np.allclose(tc.backend.real(s[3]), 0.5)


def test_unitary():
    c = tc.Circuit(2, inputs=np.eye(4))
    c.X(0)
    c.Y(1)
    answer = np.kron(tc.gates.x().tensor, tc.gates.y().tensor)
    assert np.allclose(c.wavefunction().reshape([4, 4]), answer, atol=1e-4)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_dqas_type_circuit(backend):
    eye = tc.gates.i().tensor
    x = tc.gates.x().tensor
    y = tc.gates.y().tensor
    z = tc.gates.z().tensor

    def f(params, structures):
        paramsc = tc.backend.cast(params, dtype="complex64")
        structuresc = tc.backend.softmax(structures, axis=-1)
        structuresc = tc.backend.cast(structuresc, dtype="complex64")
        c = tc.Circuit(5)
        for i in range(5):
            c.H(i)
        for j in range(2):
            for i in range(4):
                c.cz(i, i + 1)
            for i in range(5):
                c.any(
                    i,
                    unitary=structuresc[i, j, 0]
                    * (
                        tc.backend.cos(paramsc[i, j, 0]) * eye
                        + tc.backend.sin(paramsc[i, j, 0]) * x
                    )
                    + structuresc[i, j, 1]
                    * (
                        tc.backend.cos(paramsc[i, j, 1]) * eye
                        + tc.backend.sin(paramsc[i, j, 1]) * y
                    )
                    + structuresc[i, j, 2]
                    * (
                        tc.backend.cos(paramsc[i, j, 2]) * eye
                        + tc.backend.sin(paramsc[i, j, 2]) * z
                    ),
                )
        return tc.backend.real(c.expectation([tc.gates.z(), (2,)]))

    structures = tc.array_to_tensor(
        np.random.normal(size=[16, 5, 2, 3]), dtype="float32"
    )
    params = tc.array_to_tensor(np.random.normal(size=[5, 2, 3]), dtype="float32")

    vf = tc.backend.vmap(f, vectorized_argnums=(1,))

    assert np.allclose(vf(params, structures).shape, [16])

    vvag = tc.backend.vvag(f, argnums=0, vectorized_argnums=1)

    vvag = tc.backend.jit(vvag)

    value, grad = vvag(params, structures)

    assert np.allclose(value.shape, [16])
    assert np.allclose(grad.shape, [5, 2, 3])


def test_circuit_add_demo():
    # to be refactored for better API
    c = tc.Circuit(2)
    c.x(0)
    c2 = tc.Circuit(2, mps_inputs=[c._nodes, c._front])
    c2.X(0)
    answer = np.array([1.0, 0, 0, 0])
    assert np.allclose(c2.wavefunction().reshape([-1]), answer, atol=1e-4)
    c3 = tc.Circuit(2)
    c3.X(0)
    c3.replace_mps_inputs([c._nodes, c._front])
    assert np.allclose(c3.wavefunction().reshape([-1]), answer, atol=1e-4)
