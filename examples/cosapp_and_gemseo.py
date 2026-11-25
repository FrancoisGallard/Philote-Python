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
"""Interoperate CoSApp with GEMSEO and Philote-MDO.

This example illustrates both directions of the CoSApp wrappers:

1. A CoSApp system computing the area and the perimeter of a rectangle is
   served through the Philote-MDO protocol with
   ``philote_mdo.cosapp.CoSAppToPhiloteDiscipline``, and a GEMSEO client
   maximizes its area under a perimeter constraint with a gradient-based
   optimizer.
2. The ``philote_mdo.examples.Paraboloid`` discipline is served through the
   Philote-MDO protocol, used in CoSApp as a
   ``philote_mdo.cosapp.PhiloteSystem`` and solved by a CoSApp non-linear
   solver.

This requires CoSApp and GEMSEO to be installed
(``pip install philote-mdo[cosapp,gemseo]``).
"""

from __future__ import annotations

from concurrent import futures

import grpc
from cosapp.base import Port
from cosapp.base import System
from cosapp.drivers import NonLinearSolver
from gemseo import create_design_space
from gemseo import create_scenario
from gemseo.settings.formulations import DisciplinaryOpt_Settings
from gemseo.settings.opt import SLSQP_Settings

from philote_mdo.cosapp import CoSAppToPhiloteDiscipline
from philote_mdo.cosapp import PhiloteSystem
from philote_mdo.examples import Paraboloid
from philote_mdo.gemseo import PhiloteDiscipline
from philote_mdo.general import ExplicitServer

HOST = "localhost"
"""The host of the Philote discipline servers."""


class GeometryPort(Port):
    """The dimensions of a rectangle."""

    def setup(self) -> None:
        """Declare the length and the width."""
        self.add_variable("length", 1.0, unit="m")
        self.add_variable("width", 1.0, unit="m")


class Rectangle(System):
    """The area and the perimeter of a rectangle."""

    def setup(self) -> None:
        """Declare the dimensions, the area and the perimeter."""
        self.add_input(GeometryPort, "geometry")
        self.add_outward("area", 1.0, unit="m**2")
        self.add_outward("perimeter", 4.0, unit="m")

    def compute(self) -> None:
        """Compute the area and the perimeter from the dimensions."""
        length = self.geometry.length
        width = self.geometry.width
        self.area = length * width
        self.perimeter = 2 * (length + width)


def create_server(port: int) -> grpc.Server:
    """Create a gRPC server to serve a Philote discipline.

    Args:
        port: The port of the server.

    Returns:
        The gRPC server, not started yet.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server.add_insecure_port(f"[::]:{port}")
    return server


def optimize_cosapp_system_with_gemseo(port: int) -> None:
    """Serve a CoSApp system and optimize it with GEMSEO.

    Args:
        port: The port of the server.
    """
    server = create_server(port)
    ExplicitServer(
        discipline=CoSAppToPhiloteDiscipline(Rectangle("rectangle"))
    ).attach_to_server(server)
    server.start()
    try:
        # The Philote variables are named after their CoSApp paths,
        # e.g. "geometry.length" and "area".
        rectangle = PhiloteDiscipline(channel=grpc.insecure_channel(f"{HOST}:{port}"))
        design_space = create_design_space()
        design_space.add_variable(
            "geometry.length", lower_bound=0.1, upper_bound=10.0, value=1.0
        )
        design_space.add_variable(
            "geometry.width", lower_bound=0.1, upper_bound=10.0, value=2.0
        )
        scenario = create_scenario(
            [rectangle],
            objective_name="area",
            design_space=design_space,
            maximize_objective=True,
            formulation_settings_model=DisciplinaryOpt_Settings(),
        )
        scenario.add_constraint("perimeter", constraint_type="ineq", value=10.0)
        # The Jacobian is approximated by finite differences on the server side.
        scenario.execute(algo_settings_model=SLSQP_Settings(max_iter=50))
        # The optimum is the square of side 2.5 m.
        print(scenario.optimization_result)  # noqa: T201
    finally:
        # Stop the server explicitly:
        # its worker threads would otherwise keep the process alive.
        server.stop(0)


def solve_philote_discipline_with_cosapp(port: int) -> None:
    """Serve a Philote discipline and solve it with CoSApp.

    Args:
        port: The port of the server.
    """
    server = create_server(port)
    ExplicitServer(discipline=Paraboloid()).attach_to_server(server)
    server.start()
    try:
        # The Philote inputs and outputs are gathered
        # in the ports "philote_in" and "philote_out".
        paraboloid = PhiloteSystem(
            "paraboloid", channel=grpc.insecure_channel(f"{HOST}:{port}")
        )
        solver = paraboloid.add_driver(NonLinearSolver("solver", tol=1e-10))
        solver.add_unknown("philote_in.x").add_equation("philote_out.f_xy == 39.0")
        paraboloid.philote_in.x = 0.0
        paraboloid.philote_in.y = 2.0
        paraboloid.run_drivers()
        # f_xy(x, 2) = 39 <=> x = 1 or x = 3; the solver converges to x = 1.
        print(f"x = {paraboloid.philote_in.x}")  # noqa: T201
    finally:
        server.stop(0)


if __name__ == "__main__":
    optimize_cosapp_system_with_gemseo(50052)
    solve_philote_discipline_with_cosapp(50053)
