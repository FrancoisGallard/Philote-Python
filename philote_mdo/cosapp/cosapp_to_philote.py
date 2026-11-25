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
"""A Philote explicit discipline wrapping a CoSApp system."""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING
from typing import Any

from cosapp.base import System
from cosapp.ports.port import ModeVarPort
from numpy import absolute
from numpy import array
from numpy import asarray
from numpy import maximum
from numpy import ndarray
from numpy import shape as get_shape
from numpy import zeros

from philote_mdo.general.explicit_discipline import ExplicitDiscipline

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable
    from collections.abc import Mapping

    from cosapp.ports.port import BasePort


class CoSAppToPhiloteDiscipline(ExplicitDiscipline):
    """A Philote-MDO explicit discipline wrapping a CoSApp system.

    This adapts a CoSApp ``System`` to the
    :class:`~philote_mdo.general.explicit_discipline.ExplicitDiscipline`
    interface, so that it can be exposed to remote Philote clients (e.g. an
    OpenMDAO component, a
    :class:`~philote_mdo.gemseo.philote_to_gemseo.PhiloteDiscipline` or a
    :class:`~philote_mdo.cosapp.philote_to_cosapp.PhiloteSystem`) through a
    :class:`~philote_mdo.general.explicit_server.ExplicitServer`, for
    example::

        server = ExplicitServer(discipline=CoSAppToPhiloteDiscipline(my_system))
        server.attach_to_server(grpc_server)

    The Philote variables are named after their CoSApp paths in the system,
    e.g. ``"p_in.x"`` for the variable ``x`` of the port ``p_in`` and ``"a"``
    for the inward or outward ``a``. By default, the inputs are the numerical
    variables of the input ports of the system, including its inwards, and
    the outputs are the numerical variables of its output ports, including
    its outwards. A scalar CoSApp variable is exposed as a Philote variable
    of shape ``(1,)``; an array keeps its shape. The CoSApp units are
    forwarded to the Philote clients.

    Executing the discipline sets the input values of the system and calls
    its ``run_drivers`` method, so that the drivers attached to the system
    (e.g. a ``NonLinearSolver``) are run.

    As CoSApp does not provide analytic derivatives, the Jacobian of every
    output with respect to every input is approximated with forward finite
    differences, using one execution of the system per input component in
    addition to the execution at the given input values; see
    :attr:`fd_step`.
    """

    system: System
    """The CoSApp system to be wrapped."""

    fd_step: float
    r"""The relative finite difference step to approximate the Jacobian.

    The input component :math:`x_j` is perturbed by
    :math:`h_j = \delta\max(1, |x_j|)` where :math:`\delta` is this relative
    step. If ``0``, the Jacobian is not available.
    """

    def __init__(
        self,
        system: System,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
        fd_step: float = 1e-6,
    ) -> None:
        """Initialize the discipline.

        Args:
            system: The CoSApp system to be wrapped.
            input_names: The CoSApp paths of the variables to be exposed as
                Philote inputs, e.g. ``"p_in.x"``. If empty, use all the
                numerical variables of the input ports of the system,
                including its inwards.
            output_names: The CoSApp paths of the variables to be exposed as
                Philote outputs, e.g. ``"p_out.y"``. If empty, use all the
                numerical variables of the output ports of the system,
                including its outwards.
            fd_step: The relative finite difference step to approximate the
                Jacobian. If ``0``, the Jacobian is not available.
        """
        super().__init__()
        self.system = system
        self.fd_step = fd_step
        self.__input_names = tuple(input_names) or self.__get_variable_names(
            system.inputs
        )
        self.__output_names = tuple(output_names) or self.__get_variable_names(
            system.outputs
        )
        self.__shapes = {
            name: get_shape(system[name])
            for name in (*self.__input_names, *self.__output_names)
        }

    @property
    def input_names(self) -> tuple[str, ...]:
        """The CoSApp paths of the variables exposed as Philote inputs."""
        return self.__input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        """The CoSApp paths of the variables exposed as Philote outputs."""
        return self.__output_names

    def setup(self) -> None:
        """Declare the Philote inputs and outputs from the CoSApp variables."""
        for name in self.__input_names:
            self.add_input(
                name, shape=self.__get_philote_shape(name), units=self.__get_unit(name)
            )
        for name in self.__output_names:
            self.add_output(
                name, shape=self.__get_philote_shape(name), units=self.__get_unit(name)
            )

    def setup_partials(self) -> None:
        """Declare the Jacobian of every output with respect to every input.

        No sparsity pattern is assumed. Nothing is declared when the finite
        difference step is ``0``.
        """
        if not self.fd_step:
            return

        for output_name in self.__output_names:
            for input_name in self.__input_names:
                self.declare_partials(output_name, input_name)

    def compute(
        self, inputs: Mapping[str, ndarray], outputs: dict[str, ndarray]
    ) -> None:
        """Execute the wrapped CoSApp system.

        Args:
            inputs: The input values, by Philote name.
            outputs: The dictionary to be filled with the output values
                computed by the wrapped CoSApp system, by Philote name.
        """
        outputs.update(self.__execute(inputs))

    def compute_partials(
        self, inputs: Mapping[str, ndarray], partials: dict[tuple[str, str], ndarray]
    ) -> None:
        """Approximate the Jacobian of the CoSApp system with finite differences.

        The system is executed last at the given input values, so that it is
        left in the state corresponding to these input values.

        Args:
            inputs: The input values, by Philote name.
            partials: The dictionary to be filled with the Jacobian matrices,
                by ``(output_name, input_name)`` pairs.
        """
        inputs = {name: asarray(value, dtype=float) for name, value in inputs.items()}
        perturbed_outputs = {}
        steps = {}
        for input_name in self.__input_names:
            value = inputs[input_name]
            steps[input_name] = self.fd_step * maximum(1.0, absolute(value.ravel()))
            for index, step in enumerate(steps[input_name]):
                perturbed_value = value.copy()
                perturbed_value.flat[index] += step
                perturbed_outputs[input_name, index] = self.__execute(
                    {**inputs, input_name: perturbed_value}
                )

        outputs = self.__execute(inputs)
        for output_name in self.__output_names:
            output_value = outputs[output_name].ravel()
            for input_name in self.__input_names:
                input_steps = steps[input_name]
                jacobian = zeros((output_value.size, input_steps.size))
                for index, step in enumerate(input_steps):
                    perturbed_output = perturbed_outputs[input_name, index]
                    jacobian[:, index] = (
                        perturbed_output[output_name].ravel() - output_value
                    ) / step
                partials[output_name, input_name] = jacobian

    def __execute(self, inputs: Mapping[str, ndarray]) -> dict[str, ndarray]:
        """Execute the CoSApp system.

        Args:
            inputs: The input values, by Philote name.

        Returns:
            The output values, by Philote name.
        """
        system = self.system
        for name in self.__input_names:
            shape = self.__shapes[name]
            value = asarray(inputs[name], dtype=float)
            if shape:
                system[name] = value.reshape(shape).copy()
            else:
                system[name] = float(value.ravel()[0])

        system.run_drivers()
        # Copy the output values,
        # as the system may update its output arrays in place.
        return {
            name: array(system[name], dtype=float).reshape(
                self.__get_philote_shape(name)
            )
            for name in self.__output_names
        }

    def __get_philote_shape(self, name: str) -> tuple[int, ...]:
        """Return the Philote shape of a CoSApp variable.

        Args:
            name: The CoSApp path of the variable.

        Returns:
            The Philote shape of the variable, ``(1,)`` for a scalar.
        """
        return self.__shapes[name] or (1,)

    def __get_unit(self, name: str) -> str:
        """Return the unit of a CoSApp variable.

        Args:
            name: The CoSApp path of the variable.

        Returns:
            The unit of the variable, empty if dimensionless.
        """
        return self.system.get_variable(name).unit or ""

    @staticmethod
    def __get_variable_names(ports: Mapping[str, BasePort]) -> tuple[str, ...]:
        """Return the CoSApp paths of the numerical variables of ports.

        The variables of the inwards and outwards are referred to by their
        names, the others by ``"port_name.variable_name"``. The mode variables
        are ignored.

        Args:
            ports: The ports, by name.

        Returns:
            The CoSApp paths of the numerical variables.
        """
        names = []
        for port_name, port in ports.items():
            if isinstance(port, ModeVarPort):
                continue
            prefix = (
                ""
                if port_name in {System.INWARDS, System.OUTWARDS}
                else f"{port_name}."
            )
            names.extend(
                f"{prefix}{name}" for name in port if _is_numerical(port[name])
            )
        return tuple(names)


def _is_numerical(value: Any) -> bool:
    """Check whether a value can be exchanged as a Philote continuous variable.

    Args:
        value: The value.

    Returns:
        Whether the value is a real number or an array of real numbers.
    """
    if isinstance(value, ndarray):
        return value.dtype.kind in "iuf" and value.size > 0
    return isinstance(value, Real) and not isinstance(value, bool)
