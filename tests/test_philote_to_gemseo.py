# Philote-Python
#
# Copyright 2026 IRT Saint Exupery
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# originally authored by Francois Gallard, IRT Saint Exupery.
from concurrent import futures
import unittest
import grpc
from gemseo.core.discipline.discipline import Discipline
from gemseo.mda.newton_raphson import MDANewtonRaphson
from numpy import array
from numpy import eye
from numpy import ndarray
from numpy.testing import assert_allclose
import philote_mdo.general as pmdo
from philote_mdo.examples import Paraboloid
from philote_mdo.examples import QuadradicImplicit
from philote_mdo.examples import Rosenbrock
from philote_mdo.gemseo import PhiloteDiscipline
from philote_mdo.gemseo import PhiloteImplicitDiscipline

PORT = "[::]:50051"
CHANNEL = "localhost:50051"


class DynamicShapeDiscipline(pmdo.ExplicitDiscipline):
    """A discipline whose input and output shapes are set by the client."""

    def setup(self):
        self.add_input("x", dynamic_shape=True)
        self.add_output("y", dynamic_shape=True)

    def compute(self, inputs, outputs):
        outputs["y"] = 2.0 * inputs["x"]


class DiscreteDiscipline(pmdo.ExplicitDiscipline):
    """A discipline with a discrete input and a discrete output.

    The discrete input ``factor`` scales the continuous output ``y``, so
    that both the outputs and the Jacobian depend on it, and the discrete
    output ``tags`` echoes a non-scalar discrete value back to the client.
    """

    def setup(self):
        self.add_input("x", shape=(2,))
        self.add_discrete_input("factor", default=2.0)
        self.add_output("y", shape=(2,))
        self.add_discrete_output("tags")

    def setup_partials(self):
        self.declare_partials("y", "x")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["y"] = discrete_inputs["factor"] * inputs["x"]
        discrete_outputs["tags"] = ["scaled", discrete_inputs["factor"]]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["y", "x"] = discrete_inputs["factor"] * eye(2)


class PhiloteToGEMSEOTests(unittest.TestCase):
    """
    Integration tests for PhiloteDiscipline, which wraps a remote Philote
    explicit discipline server as a GEMSEO discipline.
    """

    def test_paraboloid_compute(self):
        """
        Integration test for the Paraboloid compute function.
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        discipline = pmdo.ExplicitServer(discipline=Paraboloid())
        discipline.attach_to_server(server)
        server.add_insecure_port(PORT)
        server.start()

        paraboloid_disc = PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        out = paraboloid_disc.execute({"x": array([1.0]), "y": array([2.0])})

        server.stop(0)

        self.assertEqual(out["f_xy"][0], 39.0)

    def test_paraboloid_linearize(self):
        """
        Integration test for the Paraboloid linearization, checked against
        the analytic Jacobian at (x, y) = (1, 2).
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        discipline = pmdo.ExplicitServer(discipline=Paraboloid())
        discipline.attach_to_server(server)
        server.add_insecure_port(PORT)
        server.start()

        paraboloid_disc = PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        inputs = {"x": array([1.0]), "y": array([2.0])}
        paraboloid_disc.linearize(inputs, compute_all_jacobians=True)

        server.stop(0)

        self.assertEqual(paraboloid_disc.jac["f_xy"]["x"][0][0], -2.0)
        self.assertEqual(paraboloid_disc.jac["f_xy"]["y"][0][0], 13.0)

    def test_missing_channel_raises_value_error(self):
        """
        Constructing a PhiloteDiscipline without a channel raises ValueError.
        """
        with self.assertRaises(ValueError):
            PhiloteDiscipline(channel=None)

        with self.assertRaises(ValueError):
            PhiloteDiscipline(channel="")

    def test_rosenbrock_with_options(self):
        """
        Discipline options passed to the constructor are sent to the server
        and used to build a Rosenbrock discipline of the requested dimension.
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        discipline = pmdo.ExplicitServer(discipline=Rosenbrock())
        discipline.attach_to_server(server)
        server.add_insecure_port(PORT)
        server.start()

        rosenbrock_disc = PhiloteDiscipline(
            channel=grpc.insecure_channel(CHANNEL), dimension=3
        )

        out = rosenbrock_disc.execute({"x": array([1.0, 1.0, 1.0])})

        server.stop(0)

        self.assertEqual(out["f"][0], 0.0)

    def _serve(self, discipline):
        """
        Serves a Philote discipline for the duration of the test.
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        pmdo.ExplicitServer(discipline=discipline).attach_to_server(server)
        server.add_insecure_port(PORT)
        server.start()
        self.addCleanup(server.stop, 0)

    def test_name_defaults_to_server_reported_name(self):
        """
        Without an explicit name, the discipline takes the name reported by
        the server.
        """
        paraboloid = Paraboloid()
        paraboloid._name = "paraboloid"
        self._serve(paraboloid)

        disc = PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        self.assertEqual(disc.name, "paraboloid")

    def test_name_falls_back_to_class_name(self):
        """
        When the server reports no name, GEMSEO's class-name default is used.
        """
        self._serve(Paraboloid())

        disc = PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        self.assertEqual(disc.name, "PhiloteDiscipline")

    def test_explicit_name_overrides_server_name(self):
        """
        A name passed to the constructor takes precedence over the server's.
        """
        paraboloid = Paraboloid()
        paraboloid._name = "paraboloid"
        self._serve(paraboloid)

        disc = PhiloteDiscipline(
            channel=grpc.insecure_channel(CHANNEL), name="remote_paraboloid"
        )

        self.assertEqual(disc.name, "remote_paraboloid")

    def test_dynamic_shape_variables_raise(self):
        """
        Dynamic-shape server variables are rejected at construction, naming
        the offending variables.
        """
        self._serve(DynamicShapeDiscipline())

        with self.assertRaises(NotImplementedError) as ctx:
            PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        self.assertIn("x (dynamic shape)", str(ctx.exception))
        self.assertIn("y (dynamic shape)", str(ctx.exception))

    def test_discrete_variables_in_grammars(self):
        """
        Discrete server variables are added to the grammars next to the
        continuous ones, and bound to no type, since they may carry any
        JSON-compatible value.
        """
        self._serve(DiscreteDiscipline())

        disc = PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        self.assertEqual({n: disc.input_grammar[n] for n in disc.input_grammar},
                         {"x": ndarray, "factor": None})
        self.assertEqual({n: disc.output_grammar[n] for n in disc.output_grammar},
                         {"y": ndarray, "tags": None})

    def test_discrete_variables_compute(self):
        """
        Discrete inputs are sent to the server and discrete outputs are
        returned in the local data, next to the continuous ones.
        """
        self._serve(DiscreteDiscipline())

        disc = PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        out = disc.execute({"x": array([1.0, 2.0]), "factor": 3.0})

        assert_allclose(out["y"], array([3.0, 6.0]))
        self.assertEqual(out["tags"], ["scaled", 3.0])

    def test_discrete_input_changes_output(self):
        """
        A different discrete input value yields a different output, which
        shows the value is actually sent rather than left at the
        server-side default.
        """
        self._serve(DiscreteDiscipline())

        disc = PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        out = disc.execute({"x": array([1.0, 2.0]), "factor": 10.0})

        assert_allclose(out["y"], array([10.0, 20.0]))

    def test_discrete_variables_linearize(self):
        """
        Discrete variables are excluded from the full Jacobian, and the
        discrete inputs are sent along with the continuous ones so that the
        server can use them to compute the partials.
        """
        self._serve(DiscreteDiscipline())

        disc = PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        jac = disc.linearize(
            {"x": array([1.0, 2.0]), "factor": 3.0}, compute_all_jacobians=True
        )

        self.assertEqual(list(jac), ["y"])
        self.assertEqual(list(jac["y"]), ["x"])
        assert_allclose(jac["y"]["x"], 3.0 * eye(2))


class DiscreteQuadratic(QuadradicImplicit):
    """A quadratic implicit discipline whose leading coefficient is discrete.

    The discrete input ``scale`` multiplies the whole residual, so that both
    the residuals and their partial derivatives depend on a value that
    travels on the discrete side of the wire protocol.
    """

    def setup(self):
        super().setup()
        self.add_discrete_input("scale", default=1.0)

    def compute_residuals(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        super().compute_residuals(inputs, outputs, residuals)
        residuals["x"] *= discrete_inputs["scale"]

    def residual_partials(
        self, inputs, outputs, partials, discrete_inputs=None, discrete_outputs=None
    ):
        super().residual_partials(inputs, outputs, partials)
        for key in partials:
            partials[key] *= discrete_inputs["scale"]


class Coupler(Discipline):
    """A GEMSEO discipline computing ``c = 6 - x``.

    Coupled with the quadratic implicit discipline, whose state variable is
    ``x`` and whose constant term is ``c``, it closes a cycle that a Newton
    MDA must solve, so that the state equation reads
    ``x**2 - 6 * x + 6 = 0``.
    """

    def __init__(self):
        super().__init__()
        self.input_grammar.update_from_names(["x"])
        self.output_grammar.update_from_names(["c"])
        self.default_input_data = {"x": array([1.0])}

    def _run(self, input_data):
        return {"c": 6.0 - input_data["x"]}

    def _compute_jacobian(self, input_names=(), output_names=()):
        self.jac = {"c": {"x": -eye(1)}}


class PhiloteImplicitToGEMSEOTests(unittest.TestCase):
    """
    Integration tests for PhiloteImplicitDiscipline, which wraps a remote
    Philote implicit discipline server as a GEMSEO discipline with residuals.
    """

    # The inputs of the quadratic discipline a*x**2 + b*x + c used below.
    INPUTS = {"a": array([1.0]), "b": array([-5.0]), "c": array([6.0])}

    def _serve(self, discipline=None):
        """
        Serves a Philote implicit discipline for the duration of the test.
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        pmdo.ImplicitServer(
            discipline=discipline or QuadradicImplicit()
        ).attach_to_server(server)
        server.add_insecure_port(PORT)
        server.start()
        self.addCleanup(server.stop, 0)

    @staticmethod
    def _create_discipline(**options):
        """
        Creates a PhiloteImplicitDiscipline connected to the served server.
        """
        return PhiloteImplicitDiscipline(
            channel=grpc.insecure_channel(CHANNEL), **options
        )

    def test_grammars_hold_states_and_residuals(self):
        """
        The state variables are both inputs and outputs, the residuals are
        outputs named after their state variable, and the mapping between
        them is declared to GEMSEO.
        """
        self._serve()

        disc = self._create_discipline()

        self.assertEqual(set(disc.input_grammar), {"a", "b", "c", "x"})
        self.assertEqual(set(disc.output_grammar), {"x", "x_residual"})
        self.assertEqual(disc.io.residual_to_state_variable, {"x_residual": "x"})
        self.assertFalse(disc.io.state_equations_are_solved)

    def test_residual_name_suffix(self):
        """
        The suffix used to name the residuals can be chosen.
        """
        self._serve()

        disc = self._create_discipline(residual_name_suffix="_res")

        self.assertEqual(set(disc.output_grammar), {"x", "x_res"})
        self.assertEqual(disc.io.residual_to_state_variable, {"x_res": "x"})

    def test_residual_name_collision_raises(self):
        """
        A suffix that makes a residual name collide with a server variable
        is rejected at construction.
        """
        self._serve()

        with self.assertRaises(ValueError) as ctx:
            self._create_discipline(residual_name_suffix="")

        self.assertIn("x", str(ctx.exception))

    def test_compute_residuals(self):
        """
        Executing the discipline echoes the state variable and returns the
        residual of the remote discipline at that state.
        """
        self._serve()

        disc = self._create_discipline()

        out = disc.execute({**self.INPUTS, "x": array([1.0])})

        # 1 * 1**2 - 5 * 1 + 6 = 2
        assert_allclose(out["x_residual"], array([2.0]))
        assert_allclose(out["x"], array([1.0]))

    def test_compute_residuals_with_discrete_input(self):
        """
        The discrete inputs are part of the input grammar and are sent to the
        server along with the continuous ones.
        """
        self._serve(DiscreteQuadratic())

        disc = self._create_discipline()

        self.assertEqual(set(disc.input_grammar), {"a", "b", "c", "x", "scale"})

        out = disc.execute({**self.INPUTS, "x": array([1.0]), "scale": 10.0})

        assert_allclose(out["x_residual"], array([20.0]))

    def test_solve_state_equations(self):
        """
        When the remote discipline solves its own state equations, executing
        the discipline returns the solution and a zero residual, and the MDAs
        are told not to iterate on the state variables.
        """
        self._serve()

        disc = self._create_discipline(solve_state_equations=True)

        self.assertTrue(disc.io.state_equations_are_solved)

        out = disc.execute({**self.INPUTS, "x": array([1.0])})

        # The larger root of x**2 - 5 * x + 6.
        assert_allclose(out["x"], array([3.0]))
        assert_allclose(out["x_residual"], array([0.0]), atol=1e-12)

    def test_linearize_residual_jacobian(self):
        """
        Linearizing the discipline returns the partial derivatives of the
        residual sent by the server, and the identity Jacobian of the state
        variable echoed from the input data.
        """
        self._serve()

        disc = self._create_discipline()

        x = 1.5
        jac = disc.linearize(
            {**self.INPUTS, "x": array([x])}, compute_all_jacobians=True
        )

        # R = a * x**2 + b * x + c
        assert_allclose(jac["x_residual"]["a"], array([[x**2]]))
        assert_allclose(jac["x_residual"]["b"], array([[x]]))
        assert_allclose(jac["x_residual"]["c"], array([[1.0]]))
        assert_allclose(jac["x_residual"]["x"], array([[2.0 * 1.0 * x - 5.0]]))
        # The state variable is echoed from the input data.
        assert_allclose(jac["x"]["x"], eye(1))
        assert_allclose(jac["x"]["a"], array([[0.0]]))

    def test_linearize_with_discrete_input(self):
        """
        The discrete inputs are sent along with the continuous ones so that
        the server can use them to compute the partials, and they are left
        out of the Jacobian.
        """
        self._serve(DiscreteQuadratic())

        disc = self._create_discipline()

        x = 1.5
        jac = disc.linearize(
            {**self.INPUTS, "x": array([x]), "scale": 10.0},
            compute_all_jacobians=True,
        )

        self.assertEqual(set(jac["x_residual"]), {"a", "b", "c", "x"})
        assert_allclose(jac["x_residual"]["a"], array([[10.0 * x**2]]))
        assert_allclose(jac["x_residual"]["x"], array([[10.0 * (2.0 * x - 5.0)]]))

    def _create_newton_mda(self):
        """
        Creates a Newton MDA coupling the remote quadratic discipline with an
        analytic discipline setting its constant term to ``c = 6 - x``.

        The state equation therefore reads ``x**2 - 6 * x + 6 = 0``, whose
        roots are ``3 +/- sqrt(3)``.
        """
        self._serve()

        quadratic = self._create_discipline()
        quadratic.default_input_data = {
            "a": array([1.0]),
            "b": array([-5.0]),
            "c": array([5.0]),
            "x": array([1.0]),
        }
        return MDANewtonRaphson([quadratic, Coupler()], tolerance=1e-13), quadratic

    def test_newton_mda_solves_the_state(self):
        """
        A GEMSEO Newton MDA drives the residual of the remote implicit
        discipline to zero and solves for its state variable.
        """
        mda, _ = self._create_newton_mda()

        out = mda.execute()

        solution = 3.0 - 3.0**0.5
        assert_allclose(out["x"], array([solution]), rtol=1e-10)
        assert_allclose(out["c"], array([6.0 - solution]), rtol=1e-10)
        assert_allclose(out["x_residual"], array([0.0]), atol=1e-10)

    @unittest.expectedFailure
    def test_newton_mda_solves_the_self_coupled_state(self):
        """
        A GEMSEO Newton MDA solves the state of a lone implicit discipline
        that does not solve its own residual, and its total derivatives
        match the implicit function theorem.

        Such a discipline is self-coupled through its state variable, which
        is both an input, the current guess, and an output. Two defects of
        GEMSEO 6.3.3 make this fail; the test is expected to fail until they
        are fixed upstream, and to pass again afterwards.

        1. `CouplingStructure.is_self_coupled` subtracts the state variables
           from the inputs that are also outputs, so a discipline whose only
           self-coupling is its state is reported as weakly coupled and
           `MDANewtonRaphson.__init__` rejects it. Neutralizing that
           subtraction is enough to make the solve below converge to x = 2.
        2. In `JacobianAssembly.total_derivatives`, `dfun_dy` is assembled
           over `couplings_and_res` while the solution of the coupled linear
           system is ordered by `couplings_and_states`. The state block of
           the partial derivatives of the functions is therefore looked up
           under the residual name, is never found, and the total
           derivatives across a state variable come out as zero.
        """
        self._serve()

        quadratic = self._create_discipline()
        quadratic.default_input_data = {**self.INPUTS, "x": array([1.0])}
        mda = MDANewtonRaphson([quadratic], tolerance=1e-13)

        out = mda.execute()

        # The smaller root of x**2 - 5 * x + 6, reached from x = 1.
        x = 2.0
        assert_allclose(out["x"], array([x]), rtol=1e-10)
        assert_allclose(out["x_residual"], array([0.0]), atol=1e-10)

        mda.add_differentiated_inputs(["a", "b", "c"])
        mda.add_differentiated_outputs(["x"])
        jac = mda.linearize()

        # dx/dv = -(dR/dv) / (dR/dx) with dR/dx = 2 * a * x + b.
        denominator = 2.0 * x - 5.0
        assert_allclose(jac["x"]["a"], array([[-(x**2) / denominator]]), rtol=1e-10)
        assert_allclose(jac["x"]["b"], array([[-x / denominator]]), rtol=1e-10)
        assert_allclose(jac["x"]["c"], array([[-1.0 / denominator]]), rtol=1e-10)

    def test_newton_mda_jacobian_at_the_solution(self):
        """
        The Jacobian that the Newton MDA consumes at each iteration, namely
        the partial derivatives of the residual with respect to the inputs
        and to the state variable, matches the analytic one at the solution.
        """
        mda, quadratic = self._create_newton_mda()

        out = mda.execute()
        solution = {
            "a": array([1.0]),
            "b": array([-5.0]),
            "c": out["c"],
            "x": out["x"],
        }

        x = 3.0 - 3.0**0.5
        jac = quadratic.linearize(solution, compute_all_jacobians=True)

        assert_allclose(jac["x_residual"]["a"], array([[x**2]]), rtol=1e-10)
        assert_allclose(jac["x_residual"]["b"], array([[x]]), rtol=1e-10)
        assert_allclose(jac["x_residual"]["c"], array([[1.0]]))
        # dR/dx = 2 * a * x + b, the diagonal term of the Newton matrix.
        assert_allclose(jac["x_residual"]["x"], array([[2.0 * x - 5.0]]), rtol=1e-10)
        assert_allclose(jac["x"]["x"], eye(1))

        # The whole Jacobian, state variable included, against finite differences.
        self.assertTrue(quadratic.check_jacobian(solution, threshold=1e-6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
