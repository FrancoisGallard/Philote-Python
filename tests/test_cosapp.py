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
"""Tests of the CoSApp wrappers."""

from concurrent import futures
import unittest

import grpc
from cosapp.base import Port
from cosapp.base import System
from cosapp.drivers import NonLinearSolver
from numpy import array
from numpy import ndarray
from numpy.testing import assert_allclose

import philote_mdo.general as pmdo
from philote_mdo.cosapp import CoSAppToPhiloteDiscipline
from philote_mdo.cosapp import PhiloteSystem
from philote_mdo.examples import Paraboloid
from philote_mdo.gemseo import PhiloteDiscipline


class XPort(Port):
    """A port with a scalar and a vector."""

    def setup(self):
        self.add_variable("x", 1.0, unit="m")
        self.add_variable("z", array([1.0, 2.0]))


class YPort(Port):
    """A port with a scalar."""

    def setup(self):
        self.add_variable("y", 0.0, unit="m**2")


class Quadratic(System):
    """A CoSApp system with ports, an inward, an outward and a string."""

    def setup(self):
        self.add_input(XPort, "p_in")
        self.add_output(YPort, "p_out")
        self.add_inward("a", 2.0)
        self.add_inward("label", "not numerical")
        self.add_outward("w", array([0.0, 0.0]))

    def compute(self):
        x = self.p_in.x
        z = self.p_in.z
        self.p_out.y = self.a * x**2 + z[0] * z[1]
        self.w = array([x * z[0], self.a * z[1] ** 2])


class InPlace(System):
    """A CoSApp system updating its output array in place."""

    def setup(self):
        self.add_inward("x", 1.0)
        self.add_outward("y", array([0.0, 0.0]))

    def compute(self):
        self.y[:] = [self.x, self.x**2]


class WeirdUnits(pmdo.ExplicitDiscipline):
    """A Philote discipline with names and units unknown to CoSApp."""

    def setup(self):
        self.add_input("sub.x", shape=(1,), units="furlong_per_fortnight")
        self.add_input("t", shape=(2, 2))
        self.add_output("1y", shape=(1,), units="m")

    def compute(self, inputs, outputs):
        outputs["1y"] = inputs["sub.x"] + inputs["t"].sum()


class Collision(pmdo.ExplicitDiscipline):
    """A Philote discipline whose names collide once sanitized."""

    def setup(self):
        self.add_input("a.b", shape=(1,))
        self.add_input("a_b", shape=(1,))
        self.add_output("c", shape=(1,))


class TestPhiloteSystem(unittest.TestCase):
    """Tests of a Philote discipline used as a CoSApp system."""

    def serve(self, discipline):
        """Serve a Philote discipline on a free port until the test ends.

        Args:
            discipline: The Philote discipline to be served.

        Returns:
            A gRPC channel to the server.
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        port = server.add_insecure_port("localhost:0")
        pmdo.ExplicitServer(discipline=discipline).attach_to_server(server)
        server.start()
        self.addCleanup(server.stop, 0)
        return grpc.insecure_channel(f"localhost:{port}")

    def test_with_ports(self):
        """Check the Philote variables gathered in ports."""
        system = PhiloteSystem("paraboloid", channel=self.serve(Paraboloid()))
        self.assertEqual(set(system.philote_in), {"x", "y"})
        self.assertEqual(set(system.philote_out), {"f_xy"})
        self.assertEqual(system.get_variable("philote_in.x").unit, "m")
        self.assertEqual(system.get_variable("philote_out.f_xy").unit, "m**2")
        self.assertEqual(
            system.variable_paths,
            {"x": "philote_in.x", "y": "philote_in.y", "f_xy": "philote_out.f_xy"},
        )

        system.philote_in.x = 1.0
        system.philote_in.y = 2.0
        system.run_once()
        self.assertIsInstance(system.philote_out.f_xy, float)
        self.assertAlmostEqual(system.philote_out.f_xy, 39.0)

    def test_custom_port_names(self):
        """Check the names of the ports can be customized."""
        system = PhiloteSystem(
            "paraboloid",
            channel=self.serve(Paraboloid()),
            input_port_name="p_in",
            output_port_name="p_out",
        )
        system.p_in.x = 1.0
        system.p_in.y = 2.0
        system.run_once()
        self.assertAlmostEqual(system.p_out.f_xy, 39.0)

    def test_without_ports(self):
        """Check the Philote variables declared as inwards and outwards."""
        system = PhiloteSystem(
            "paraboloid", channel=self.serve(Paraboloid()), use_ports=False
        )
        self.assertEqual(set(system.inwards), {"x", "y"})
        self.assertEqual(set(system.outwards), {"f_xy"})
        system.x = 1.0
        system.y = 2.0
        system.run_once()
        self.assertAlmostEqual(system.f_xy, 39.0)

    def test_options(self):
        """Check the discipline options are sent to the server."""

        class Scaled(pmdo.ExplicitDiscipline):
            def __init__(self):
                super().__init__()
                self.factor = 1.0

            def initialize(self):
                self.add_option("factor", "float")

            def set_options(self, options):
                self.factor = options["factor"]

            def setup(self):
                self.add_input("x", shape=(1,))
                self.add_output("y", shape=(1,))

            def compute(self, inputs, outputs):
                outputs["y"] = self.factor * inputs["x"]

        system = PhiloteSystem(
            "scaled", channel=self.serve(Scaled()), options={"factor": 3.0}
        )
        system.philote_in.x = 2.0
        system.run_once()
        self.assertAlmostEqual(system.philote_out.y, 6.0)

    def test_in_solver(self):
        """Check a Philote system can be solved by a CoSApp non-linear solver."""
        system = PhiloteSystem("paraboloid", channel=self.serve(Paraboloid()))
        solver = system.add_driver(NonLinearSolver("solver", tol=1e-10))
        solver.add_unknown("philote_in.x").add_equation("philote_out.f_xy == 39.0")
        system.philote_in.x = 0.0
        system.philote_in.y = 2.0
        system.run_drivers()
        self.assertAlmostEqual(system.philote_out.f_xy, 39.0)
        # (x-3)^2 + 2x + 36 - 3 = 39 <=> x^2 - 4x + 3 = 0 <=> x in {1, 3}.
        self.assertAlmostEqual(system.philote_in.x, 1.0)

    def test_sanitized_names(self):
        """Check the invalid CoSApp names and the unknown units."""
        with self.assertLogs("philote_mdo.cosapp.philote_to_cosapp", "WARNING") as logs:
            system = PhiloteSystem("weird", channel=self.serve(WeirdUnits()))
        self.assertEqual(
            logs.records[0].getMessage(),
            "The unit 'furlong_per_fortnight' of the Philote variable 'sub.x' is "
            "unknown to CoSApp; the variable is considered as dimensionless.",
        )
        self.assertEqual(
            system.variable_paths,
            {"sub.x": "philote_in.sub_x", "t": "philote_in.v_t", "1y": "philote_out.v_1y"},
        )
        self.assertEqual(system.get_variable("philote_in.sub_x").unit, "")

        system.philote_in.sub_x = 1.0
        system.philote_in.v_t = array([[1.0, 2.0], [3.0, 4.0]])
        system.run_once()
        self.assertAlmostEqual(system.philote_out.v_1y, 11.0)

    def test_name_collision(self):
        """Check that an error is raised when sanitized names collide."""
        with self.assertRaises(ValueError) as context:
            PhiloteSystem("collision", channel=self.serve(Collision()))
        self.assertEqual(
            str(context.exception),
            "Some Philote variables have the same CoSApp name: a.b -> a_b, a_b -> a_b.",
        )

    def test_no_channel(self):
        """Check that an error is raised without channel."""
        with self.assertRaises(ValueError) as context:
            PhiloteSystem("paraboloid", channel=None)
        self.assertEqual(
            str(context.exception),
            "No channel provided, the Philote client will not be able to connect.",
        )

    def test_cosapp_to_cosapp(self):
        """Check a CoSApp system used as another CoSApp system through Philote."""
        channel = self.serve(CoSAppToPhiloteDiscipline(Quadratic("quadratic")))
        system = PhiloteSystem("remote", channel=channel)
        self.assertEqual(
            system.variable_paths,
            {
                "a": "philote_in.a",
                "p_in.x": "philote_in.p_in_x",
                "p_in.z": "philote_in.p_in_z",
                "w": "philote_out.w",
                "p_out.y": "philote_out.p_out_y",
            },
        )
        self.assertEqual(system.get_variable("philote_in.p_in_x").unit, "m")
        system.philote_in.a = 3.0
        system.philote_in.p_in_x = 2.0
        system.philote_in.p_in_z = array([4.0, 5.0])
        system.run_once()
        self.assertAlmostEqual(system.philote_out.p_out_y, 32.0)
        self.assertIsInstance(system.philote_out.w, ndarray)
        assert_allclose(system.philote_out.w, [8.0, 75.0])

    def test_cosapp_to_gemseo(self):
        """Check a CoSApp system used as a GEMSEO discipline through Philote."""
        system = Quadratic("quadratic")
        discipline = PhiloteDiscipline(
            channel=self.serve(CoSAppToPhiloteDiscipline(system, fd_step=1e-7))
        )
        input_data = {
            "a": array([3.0]),
            "p_in.x": array([2.0]),
            "p_in.z": array([4.0, 5.0]),
        }
        output_data = discipline.execute(input_data)
        assert_allclose(output_data["p_out.y"], [32.0])
        assert_allclose(output_data["w"], [8.0, 75.0])
        # The CoSApp system is executed on the server side.
        self.assertAlmostEqual(system.p_out.y, 32.0)

        self.assertTrue(
            discipline.check_jacobian(input_data, step=1e-6, threshold=1e-4)
        )


class TestCoSAppToPhiloteDiscipline(unittest.TestCase):
    """Tests of a CoSApp system wrapped as a Philote discipline."""

    def test_names(self):
        """Check the Philote variables declared from a CoSApp system."""
        discipline = CoSAppToPhiloteDiscipline(Quadratic("quadratic"))
        self.assertEqual(discipline.input_names, ("a", "p_in.x", "p_in.z"))
        self.assertEqual(discipline.output_names, ("w", "p_out.y"))
        discipline.setup()
        discipline.setup_partials()
        self.assertEqual(
            {var.name: tuple(var.shape) for var in discipline._var_meta},
            {"a": (1,), "p_in.x": (1,), "p_in.z": (2,), "w": (2,), "p_out.y": (1,)},
        )
        units = {var.name: var.units for var in discipline._var_meta}
        self.assertEqual(units["p_in.x"], "m")
        self.assertEqual(units["p_out.y"], "m**2")
        self.assertEqual(len(discipline._partials_meta), 6)

    def test_custom_names(self):
        """Check that the exposed CoSApp variables can be chosen."""
        discipline = CoSAppToPhiloteDiscipline(
            Quadratic("quadratic"), input_names=["p_in.x"], output_names=["p_out.y"]
        )
        self.assertEqual(discipline.input_names, ("p_in.x",))
        self.assertEqual(discipline.output_names, ("p_out.y",))

    def test_without_jacobian(self):
        """Check that no partials are declared when the FD step is zero."""
        discipline = CoSAppToPhiloteDiscipline(Quadratic("quadratic"), fd_step=0.0)
        discipline.setup()
        discipline.setup_partials()
        self.assertFalse(discipline._partials_meta)

    def test_partials(self):
        """Check the finite difference Jacobian against the analytic one."""
        system = Quadratic("quadratic")
        discipline = CoSAppToPhiloteDiscipline(system, fd_step=1e-7)
        inputs = {
            "a": array([3.0]),
            "p_in.x": array([2.0]),
            "p_in.z": array([4.0, 5.0]),
        }
        partials = {}
        discipline.compute_partials(inputs, partials)
        expected = {
            ("p_out.y", "a"): [[4.0]],
            ("p_out.y", "p_in.x"): [[12.0]],
            ("p_out.y", "p_in.z"): [[5.0, 4.0]],
            ("w", "a"): [[0.0], [25.0]],
            ("w", "p_in.x"): [[4.0], [0.0]],
            ("w", "p_in.z"): [[2.0, 0.0], [0.0, 30.0]],
        }
        self.assertEqual(partials.keys(), expected.keys())
        for key, value in expected.items():
            assert_allclose(partials[key], value, rtol=1e-5, atol=1e-5)

        # The system is left in the state corresponding to the given inputs.
        self.assertAlmostEqual(system.p_in.x, 2.0)
        self.assertAlmostEqual(system.p_out.y, 32.0)

    def test_partials_in_place(self):
        """Check the Jacobian of a system updating its output array in place."""
        discipline = CoSAppToPhiloteDiscipline(InPlace("in_place"), fd_step=1e-7)
        partials = {}
        discipline.compute_partials({"x": array([3.0])}, partials)
        assert_allclose(partials["y", "x"], [[1.0], [6.0]], rtol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
