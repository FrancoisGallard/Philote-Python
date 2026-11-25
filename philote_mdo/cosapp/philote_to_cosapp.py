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
"""A CoSApp system wrapping a remote Philote explicit discipline server."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from typing import Any

from cosapp.base import Port
from cosapp.base import System
from cosapp.ports.units import is_valid_units
from cosapp.utils.naming import NameChecker
from numpy import asarray
from numpy import zeros

import philote_mdo.generated.data_pb2 as data
from philote_mdo.general.explicit_client import ExplicitClient

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable
    from collections.abc import Mapping

    from grpc import Channel
    from numpy import ndarray

logger = logging.getLogger(__name__)

_INVALID_CHARACTERS = re.compile(r"\W")
"""The characters not allowed in a CoSApp variable name."""


class PhiloteSystem(System):
    """A CoSApp system connected to a remote Philote explicit discipline.

    This system acts as a Philote-MDO client: at setup, it connects to a
    Philote explicit discipline server through a gRPC channel, triggers the
    remote ``Setup`` and requests the variable metadata. This metadata is
    used to declare the CoSApp variables, so that the resulting system can be
    used like any other CoSApp system, e.g. as a child of an assembly solved
    by a ``NonLinearSolver``.

    By default, the Philote inputs are gathered in an input ``Port`` named
    ``philote_in`` and the Philote outputs in an output port named
    ``philote_out``, so that they can be connected to other systems
    port-to-port or variable by variable, e.g.::

        system = PhiloteSystem("remote", channel=grpc.insecure_channel(url))
        system.philote_in.x = 1.0
        system.run_once()
        system.philote_out.f_xy

    With ``use_ports=False``, the Philote inputs and outputs are declared as
    inwards and outwards instead, e.g. ``system.x`` and ``system.f_xy``.

    A Philote variable of shape ``(1,)`` is exposed as a ``float``; any other
    variable is exposed as a NumPy array of its Philote shape. A Philote
    variable name that is not a valid CoSApp name, e.g. ``sub.x``, is
    sanitized, e.g. into ``sub_x``; see :attr:`variable_paths`.

    Calling :meth:`compute` sends the input values to the remote server and
    retrieves the output values. Only the continuous variables are
    supported.
    """

    variable_paths: dict[str, str] = None
    """The CoSApp paths of the variables, e.g. ``"philote_in.x"``, by Philote
    name."""

    _client: ExplicitClient = None
    """The Philote client connected to the remote discipline server."""

    _input_shapes: dict[str, tuple[int, ...]] = None
    """The shapes of the Philote inputs, by Philote name."""

    _output_shapes: dict[str, tuple[int, ...]] = None
    """The shapes of the Philote outputs, by Philote name."""

    def setup(
        self,
        channel: Channel,
        use_ports: bool = True,
        input_port_name: str = "philote_in",
        output_port_name: str = "philote_out",
        options: Mapping[str, Any] | None = None,
    ) -> None:
        """Connect the Philote client and declare the CoSApp variables.

        Args:
            channel: The gRPC channel to the Philote discipline server, e.g.
                created with ``grpc.insecure_channel("localhost:50051")``.
            use_ports: Whether to gather the Philote inputs and outputs in an
                input port and an output port respectively. Otherwise,
                declare them as inwards and outwards.
            input_port_name: The name of the input port when ``use_ports`` is
                ``True``.
            output_port_name: The name of the output port when ``use_ports``
                is ``True``.
            options: The discipline options to send to the server, if any.

        Raises:
            ValueError: When ``channel`` is empty or ``None``, or when two
                Philote variables have the same sanitized name.
        """
        if not channel:
            msg = "No channel provided, the Philote client will not be able to connect."
            raise ValueError(msg)

        client = ExplicitClient(channel=channel)
        client.send_stream_options()
        if options:
            client.send_options(options)
        client.run_setup()
        client.get_variable_definitions()
        self._client = client

        input_meta = [var for var in client._var_meta if var.type == data.kInput]
        output_meta = [var for var in client._var_meta if var.type == data.kOutput]
        self._input_shapes = {var.name: tuple(var.shape) for var in input_meta}
        self._output_shapes = {var.name: tuple(var.shape) for var in output_meta}
        input_names = self.__get_cosapp_names(input_meta)
        output_names = self.__get_cosapp_names(output_meta)
        if use_ports:
            self.add_input(
                self.__create_port_class("PhiloteInputPort", input_meta, input_names),
                input_port_name,
            )
            self.add_output(
                self.__create_port_class(
                    "PhiloteOutputPort", output_meta, output_names
                ),
                output_port_name,
            )
            self.variable_paths = {
                **{
                    var.name: f"{input_port_name}.{input_names[var.name]}"
                    for var in input_meta
                },
                **{
                    var.name: f"{output_port_name}.{output_names[var.name]}"
                    for var in output_meta
                },
            }
        else:
            for var in input_meta:
                self.add_inward(
                    input_names[var.name],
                    self.__get_default_value(var),
                    unit=self.__get_unit(var),
                )
            for var in output_meta:
                self.add_outward(
                    output_names[var.name],
                    self.__get_default_value(var),
                    unit=self.__get_unit(var),
                )
            self.variable_paths = {**input_names, **output_names}

    def compute(self) -> None:
        """Evaluate the remote Philote discipline.

        This sends the input values to the remote Philote discipline server
        through the ``ComputeFunction`` RPC and sets the resulting output
        values.
        """
        input_data = {
            name: asarray(self[self.variable_paths[name]], dtype=float).reshape(shape)
            for name, shape in self._input_shapes.items()
        }
        output_data = self._client.run_compute(input_data)
        for name, shape in self._output_shapes.items():
            value = asarray(output_data[name], dtype=float)
            if shape == (1,):
                value = float(value.ravel()[0])
            else:
                value = value.reshape(shape).copy()
            self[self.variable_paths[name]] = value

    @staticmethod
    def __get_cosapp_names(
        variables: Iterable[data.VariableMetaData],
    ) -> dict[str, str]:
        """Return valid CoSApp names for Philote variables.

        Args:
            variables: The metadata of the Philote variables.

        Returns:
            The CoSApp names, by Philote name.

        Raises:
            ValueError: When two Philote variables have the same CoSApp name.
        """
        reserved_names = NameChecker.reserved()
        names = {}
        for var in variables:
            name = _INVALID_CHARACTERS.sub("_", var.name)
            if not name[0].isalpha() or name in reserved_names:
                name = f"v_{name}"
            names[var.name] = name

        if len(set(names.values())) != len(names):
            msg = (
                "Some Philote variables have the same CoSApp name: "
                f"{', '.join(f'{k} -> {v}' for k, v in names.items())}."
            )
            raise ValueError(msg)

        return names

    @classmethod
    def __create_port_class(
        cls,
        class_name: str,
        variables: Iterable[data.VariableMetaData],
        names: Mapping[str, str],
    ) -> type[Port]:
        """Create a CoSApp port class declaring Philote variables.

        Args:
            class_name: The name of the port class.
            variables: The metadata of the Philote variables.
            names: The CoSApp names of the Philote variables, by Philote name.

        Returns:
            The port class.
        """
        definitions = [
            (names[var.name], cls.__get_default_value(var), cls.__get_unit(var))
            for var in variables
        ]

        def setup(port: Port) -> None:
            for name, value, unit in definitions:
                port.add_variable(name, value, unit=unit)

        return type(class_name, (Port,), {"setup": setup})

    @staticmethod
    def __get_default_value(variable: data.VariableMetaData) -> float | ndarray:
        """Return the default value of a Philote variable.

        Args:
            variable: The metadata of the Philote variable.

        Returns:
            The default value, ``0.`` for a variable of shape ``(1,)``.
        """
        shape = tuple(variable.shape)
        if shape == (1,):
            return 0.0
        return zeros(shape)

    @staticmethod
    def __get_unit(variable: data.VariableMetaData) -> str:
        """Return the CoSApp unit of a Philote variable.

        Args:
            variable: The metadata of the Philote variable.

        Returns:
            The unit of the Philote variable if CoSApp knows it, otherwise an
            empty string, i.e. dimensionless.
        """
        unit = variable.units
        if not unit or is_valid_units(unit):
            return unit

        logger.warning(
            "The unit %r of the Philote variable %r is unknown to CoSApp; "
            "the variable is considered as dimensionless.",
            unit,
            variable.name,
        )
        return ""
