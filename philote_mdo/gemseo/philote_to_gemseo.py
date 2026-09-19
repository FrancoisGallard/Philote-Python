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
"""GEMSEO disciplines wrapping remote Philote discipline servers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.core.discipline.discipline import Discipline
from gemseo.core.grammars.factory import GrammarType
from numpy import eye
from numpy import prod

import philote_mdo.general as pm
import philote_mdo.generated.data_pb2 as data

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable
    from collections.abc import Mapping


class BasePhiloteDiscipline(Discipline):
    """The base class of the GEMSEO disciplines connected to a Philote server.

    At construction, a subclass connects to a Philote discipline server
    through a gRPC channel, triggers the remote ``Setup`` and requests the
    input/output and partials metadata. This metadata is used to build the
    GEMSEO input and output grammars, so that the resulting discipline can
    be used like any other GEMSEO discipline.

    The discrete variables of the remote discipline are part of the
    grammars, next to the continuous ones, and are read from and written to
    the local data like any other variable. Since a Philote discrete
    variable may carry any JSON-compatible value, its grammar element is
    bound to no type. Discrete variables are never differentiated: they are
    excluded from the Jacobian, including when it is requested in full with
    ``compute_all_jacobians=True``.
    """

    default_grammar_type: ClassVar[GrammarType] = GrammarType.SIMPLE
    """The default type of grammar."""

    _CLIENT_CLASS: ClassVar[type] = pm.DisciplineClient
    """The class of the Philote client used to talk to the server."""

    def __init__(self, channel, name="", **options):
        """Initialize the discipline and connect its Philote client.

        Args:
            channel: The gRPC channel to the Philote discipline server,
                e.g. created with ``grpc.insecure_channel("localhost:50051")``.
            name: The name of the discipline.
                If empty, use the name reported by the server,
                or the class name if the server reports none.
            **options: The discipline options to send to the server,
                if any.

        Raises:
            ValueError: When ``channel`` is empty or ``None``.
            NotImplementedError: When the server declares dynamic-shape
                variables, which are not supported yet.
        """
        if not channel:
            msg = "No channel provided, the Philote client will not be able to connect."
            raise ValueError(msg)
        # The shape of each continuous input and output variable, indexed by
        # name; filled in by _initialize_grammars() and used by
        # _compute_jacobian() to reshape the flattened partial derivatives
        # sent by the server.
        self._shapes = {}
        # The names of the discrete input and output variables, filled in by
        # _initialize_grammars(); discrete inputs travel on their own side of
        # the wire protocol and so must be split off the GEMSEO input data.
        self._discrete_input_names = set()
        self._discrete_output_names = set()
        # generic Philote client
        self._client = self._CLIENT_CLASS(channel=channel)

        # call the init function of the remote discipline; an empty name
        # makes GEMSEO fall back to the class name
        self._client.get_discipline_info()
        super().__init__(name=name or self._client._name)

        self._client.send_stream_options()
        if options:
            self._client.send_options(options)

        # run setup
        self._client.run_setup()
        self._client.get_variable_definitions()
        self._client.get_partials_definitions()
        self._initialize_grammars()

    def _initialize_grammars(self) -> None:
        """Set up the GEMSEO discipline input and output grammars.

        Raises:
            NotImplementedError: When the subclass does not implement it.
        """
        raise NotImplementedError

    def _split_input_data(self, input_data: Mapping[str, Any]) -> tuple[dict, dict]:
        """Split the GEMSEO input data into continuous and discrete inputs.

        The GEMSEO input grammar holds the continuous and the discrete inputs
        side by side, while the Philote wire protocol carries them in two
        distinct kinds of message.

        Args:
            input_data: The input data, without namespace prefixes.

        Returns:
            The continuous input values and the discrete input values,
            both indexed by variable name.
        """
        inputs = {}
        discrete_inputs = {}
        for name, value in input_data.items():
            if name in self._discrete_input_names:
                discrete_inputs[name] = value
            else:
                inputs[name] = value
        return inputs, discrete_inputs

    def _get_differentiated_io(
        self,
        compute_all_jacobians: bool = False,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return the inputs and outputs used in the differentiation.

        The discrete variables are filtered out, as they cannot be
        differentiated. GEMSEO already does this when the differentiated
        variables are selected with :meth:`.add_differentiated_inputs` and
        :meth:`.add_differentiated_outputs`, but not when the whole Jacobian
        is requested.

        Args:
            compute_all_jacobians: Whether to compute the Jacobians of all the
                outputs with respect to all the inputs.

        Returns:
            The names of the differentiated inputs
            and the names of the differentiated outputs.
        """
        input_names, output_names = super()._get_differentiated_io(
            compute_all_jacobians
        )
        if not compute_all_jacobians:
            return input_names, output_names

        return (
            tuple(
                name for name in input_names if name not in self._discrete_input_names
            ),
            tuple(
                name
                for name in output_names
                if name not in self._discrete_output_names
            ),
        )

    def _check_dynamic_shapes(self) -> None:
        """Check that the server declares no dynamic-shape variable.

        Raises:
            NotImplementedError: When the server declares dynamic-shape
                variables, whose shapes would have to be sent to the server.
        """
        unsupported = [
            f"{var.name} (dynamic shape)"
            for var in self._client._var_meta
            if var.dynamic_shape
        ]
        if unsupported:
            msg = (
                f"{type(self).__name__} does not support these server variables yet: "
                + ", ".join(unsupported)
            )
            raise NotImplementedError(msg)

    def _read_discrete_variable_names(self) -> None:
        """Split the discrete server variables into input and output names."""
        for var in self._client._discrete_var_meta:
            if var.type == data.kDiscreteInput:
                self._discrete_input_names.add(var.name)

            if var.type == data.kDiscreteOutput:
                self._discrete_output_names.add(var.name)


class PhiloteDiscipline(BasePhiloteDiscipline):
    """A GEMSEO discipline connected to a remote Philote explicit discipline.

    This discipline acts as a Philote-MDO client: at construction, it
    connects to a Philote explicit discipline server through a gRPC
    channel, triggers the remote ``Setup`` and requests the input/output
    and partials metadata. This metadata is used to build the GEMSEO
    input and output grammars, so that the resulting discipline can be
    used like any other GEMSEO discipline, e.g. in an
    :class:`~gemseo.mda.base_mda.BaseMDA`.

    Executing the discipline (:meth:`.execute`) and linearizing it
    (:meth:`.linearize`) transparently call the remote server to compute
    the outputs and the Jacobian, respectively.

    The discrete variables of the remote discipline are part of the
    grammars, next to the continuous ones, and are read from and written to
    the local data like any other variable. Since a Philote discrete
    variable may carry any JSON-compatible value, its grammar element is
    bound to no type. Discrete variables are never differentiated: they are
    excluded from the Jacobian, including when it is requested in full with
    ``compute_all_jacobians=True``.
    """

    _CLIENT_CLASS: ClassVar[type] = pm.ExplicitClient
    """The class of the Philote client used to talk to the server."""

    def _run(self, input_data: dict) -> dict:
        """Compute the function evaluation.

        This sends the input values to the remote Philote discipline server
        through the ``ComputeFunction`` RPC and returns the resulting output
        values. Continuous and discrete inputs are sent separately, and the
        discrete outputs returned by the server, if any, are merged back into
        the output data.

        Args:
            input_data: The input data, without namespace prefixes.

        Returns:
            The output data computed by the remote discipline.
        """
        inputs, discrete_inputs = self._split_input_data(input_data)
        outputs = self._client.run_compute(inputs, discrete_inputs=discrete_inputs)
        # run_compute returns (outputs, discrete_outputs) when the server
        # sends back discrete output data, and a plain dictionary otherwise.
        if isinstance(outputs, tuple):
            outputs, discrete_outputs = outputs
            outputs.update(discrete_outputs)
        return outputs

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """Compute one Jacobian matrix per input-output pair.

        This sends the current input values to the remote Philote discipline
        server through the ``ComputeGradient`` RPC, reshapes the resulting
        flattened partial derivatives according to the input and output
        shapes obtained by :meth:`._initialize_grammars`, and stores them
        in :attr:`.jac` as a dictionary
        ``{output_name: {input_name: jacobian_matrix}}``.

        Note:
            The remote discipline always returns the Jacobian of every
            declared output with respect to every declared input, so
            ``input_names`` and ``output_names`` are currently not used to
            restrict the RPC request: the full Jacobian is always computed
            and stored.

        Args:
            input_names: The names of the inputs against which to differentiate the
                outputs. If empty, use all the inputs.
            output_names: The names of the outputs to be differentiated.
                If empty, use all the outputs.
        """
        inputs, discrete_inputs = self._split_input_data(self.get_input_data())
        jac_flat = self._client.run_compute_partials(
            inputs, discrete_inputs=discrete_inputs
        )
        for jac_key, jac_val in jac_flat.items():
            out_name, in_name = jac_key
            out_size = int(prod(self._shapes[out_name]))
            in_size = int(prod(self._shapes[in_name]))
            jac_loc = jac_val.reshape(out_size, in_size)
            if out_name not in self.jac:
                self.jac[out_name] = {in_name: jac_loc}
            else:
                self.jac[out_name][in_name] = jac_loc

    def _initialize_grammars(self):
        """Set up the GEMSEO discipline input and output grammars.

        This builds the input and output grammars, and populates
        :attr:`._shapes`, from the variable metadata already retrieved
        from the remote discipline server by
        :meth:`~philote_mdo.general.discipline_client.DisciplineClient.get_variable_definitions`.
        It does not perform any RPC call itself.

        Continuous variables are bound to :class:`~numpy.ndarray`, while
        discrete variables are bound to no type at all, since a Philote
        discrete variable may hold any JSON-compatible value.

        Raises:
            NotImplementedError: When the server declares dynamic-shape
                variables, whose shapes would have to be sent to the server.
        """
        self._check_dynamic_shapes()

        input_names = []
        output_names = []
        for var in self._client._var_meta:
            if var.type == data.kInput:
                input_names.append(var.name)
                self._shapes[var.name] = tuple(var.shape)

            if var.type == data.kOutput:
                output_names.append(var.name)
                self._shapes[var.name] = tuple(var.shape)

        self._read_discrete_variable_names()

        self.input_grammar.update_from_names(input_names)
        self.output_grammar.update_from_names(output_names)
        # A None type means that the grammar accepts any value for that name,
        # which is what a Philote discrete variable may carry.
        self.input_grammar.update_from_types(
            dict.fromkeys(self._discrete_input_names)
        )
        self.output_grammar.update_from_types(
            dict.fromkeys(self._discrete_output_names)
        )


class PhiloteImplicitDiscipline(BasePhiloteDiscipline):
    r"""A GEMSEO discipline connected to a remote Philote implicit discipline.

    A Philote implicit discipline defines its outputs :math:`y` implicitly,
    through the residual equations :math:`R(x, y) = 0` where :math:`x` are
    the inputs. GEMSEO calls :math:`y` the *state variables* and expects a
    discipline with residuals to:

    - take the state variables as **inputs**, as the current guess,
    - return both the state variables and the residuals as **outputs**,
    - map each residual name to its state variable name in
      :attr:`~gemseo.core.discipline.io.IO.residual_to_state_variable`.

    Since Philote gives a residual the very name of the output it belongs
    to, the residuals are renamed in the GEMSEO grammars by appending
    ``residual_name_suffix`` to the state variable name. The state
    variables themselves are echoed unchanged from the input data to the
    output data, so that GEMSEO can read their size and value from the
    local data.

    Declaring the residuals this way is what lets a GEMSEO MDA solve the
    state equations, e.g. with
    :class:`~gemseo.mda.newton_raphson.MDANewtonRaphson`, which drives the
    residuals to zero with the Newton step
    :math:`-\left[\partial R/\partial y\right]^{-1} R`.

    :meth:`.linearize` returns the Jacobians :math:`\partial R/\partial x`
    and :math:`\partial R/\partial y` of the residuals, obtained from the
    ``ComputeResidualGradients`` RPC, together with the identity Jacobian
    of the echoed state variables.

    When the remote discipline is able to solve its own state equations,
    pass ``solve_state_equations=True``: :meth:`._run` then calls the
    ``SolveResiduals`` RPC and returns the solved state variables, and
    :attr:`~gemseo.core.discipline.io.IO.state_equations_are_solved` tells
    the MDAs not to iterate on them.

    The discrete inputs of the remote discipline are part of the input
    grammar, next to the continuous ones, and are sent to the server along
    with every residual, solve and gradient request. Since a Philote
    discrete variable may carry any JSON-compatible value, its grammar
    element is bound to no type. The implicit Philote services never send
    discrete outputs back, so the discrete outputs of the remote
    discipline, if any, are left out of the output grammar.
    """

    _CLIENT_CLASS: ClassVar[type] = pm.ImplicitClient
    """The class of the Philote client used to talk to the server."""

    def __init__(
        self,
        channel,
        name="",
        residual_name_suffix="_residual",
        solve_state_equations=False,
        **options,
    ):
        """Initialize the discipline and connect its Philote client.

        Args:
            channel: The gRPC channel to the Philote discipline server,
                e.g. created with ``grpc.insecure_channel("localhost:50051")``.
            name: The name of the discipline.
                If empty, use the name reported by the server,
                or the class name if the server reports none.
            residual_name_suffix: The suffix appended to the name of a state
                variable to name its residual in the output grammar.
                Philote names a residual after the output it belongs to,
                while GEMSEO requires two distinct names.
            solve_state_equations: Whether the remote discipline solves its
                own state equations, in which case :meth:`._run` calls the
                ``SolveResiduals`` RPC and returns the solved state
                variables. Otherwise, the state variables are echoed from
                the input data and the residuals are left for an MDA to
                drive to zero.
            **options: The discipline options to send to the server,
                if any.

        Raises:
            ValueError: When ``channel`` is empty or ``None``,
                or when a residual name collides with a server variable name.
            NotImplementedError: When the server declares dynamic-shape
                variables, which are not supported yet.
        """
        self._residual_name_suffix = residual_name_suffix
        self._solve_state_equations = solve_state_equations
        # The name of the residual of each state variable, indexed by state
        # variable name; filled in by _initialize_grammars().
        self._state_to_residual = {}
        super().__init__(channel, name=name, **options)

    def _initialize_grammars(self):
        """Set up the GEMSEO discipline input and output grammars.

        This builds the input and output grammars, and populates
        :attr:`._shapes` and :attr:`._state_to_residual`, from the variable
        metadata already retrieved from the remote discipline server by
        :meth:`~philote_mdo.general.discipline_client.DisciplineClient.get_variable_definitions`.
        It does not perform any RPC call itself.

        The input grammar holds the Philote inputs and the state variables,
        while the output grammar holds the state variables and their
        residuals. Continuous variables are bound to :class:`~numpy.ndarray`,
        while discrete inputs are bound to no type at all, since a Philote
        discrete variable may hold any JSON-compatible value.

        Raises:
            ValueError: When a residual name collides with a server variable
                name.
            NotImplementedError: When the server declares dynamic-shape
                variables, whose shapes would have to be sent to the server.
        """
        self._check_dynamic_shapes()

        input_names = []
        state_names = []
        for var in self._client._var_meta:
            if var.type == data.kInput:
                input_names.append(var.name)
                self._shapes[var.name] = tuple(var.shape)

            if var.type == data.kOutput:
                state_names.append(var.name)
                self._shapes[var.name] = tuple(var.shape)

        self._read_discrete_variable_names()

        self._state_to_residual = {
            name: f"{name}{self._residual_name_suffix}" for name in state_names
        }
        variable_names = set(input_names).union(
            state_names, self._discrete_input_names, self._discrete_output_names
        )
        collisions = sorted(
            variable_names.intersection(self._state_to_residual.values())
        )
        if collisions:
            msg = (
                "The residual names built with the suffix "
                f"{self._residual_name_suffix!r} collide with the server "
                "variables: " + ", ".join(collisions)
            )
            raise ValueError(msg)

        self.input_grammar.update_from_names(input_names + state_names)
        self.output_grammar.update_from_names(
            state_names + list(self._state_to_residual.values())
        )
        # A None type means that the grammar accepts any value for that name,
        # which is what a Philote discrete variable may carry.
        self.input_grammar.update_from_types(
            dict.fromkeys(self._discrete_input_names)
        )
        self.io.residual_to_state_variable = {
            residual_name: state_name
            for state_name, residual_name in self._state_to_residual.items()
        }
        self.io.state_equations_are_solved = self._solve_state_equations

    def _split_implicit_input_data(
        self, input_data: Mapping[str, Any]
    ) -> tuple[dict, dict, dict]:
        """Split the GEMSEO input data into inputs, states and discrete inputs.

        The GEMSEO input grammar holds the Philote inputs, the state
        variables and the discrete inputs side by side, while the Philote
        wire protocol carries them as three distinct kinds of message.

        Args:
            input_data: The input data, without namespace prefixes.

        Returns:
            The continuous input values,
            the state variable values
            and the discrete input values,
            all indexed by variable name.
        """
        inputs = {}
        states = {}
        discrete_inputs = {}
        for name, value in input_data.items():
            if name in self._discrete_input_names:
                discrete_inputs[name] = value
            elif name in self._state_to_residual:
                states[name] = value
            else:
                inputs[name] = value
        return inputs, states, discrete_inputs

    def _run(self, input_data: dict) -> dict:
        """Compute the state variables and their residuals.

        When the remote discipline solves its own state equations, this
        first sends the input values through the ``SolveResiduals`` RPC and
        uses the solved state variables. Otherwise, the state variables are
        the ones read from the input data. In both cases, the residuals at
        these state variables are then computed through the
        ``ComputeResiduals`` RPC.

        Args:
            input_data: The input data, without namespace prefixes.

        Returns:
            The state variables and their residuals,
            the latter indexed by the residual names of the output grammar.
        """
        inputs, states, discrete_inputs = self._split_implicit_input_data(input_data)
        if self._solve_state_equations:
            states = self._client.run_solve_residuals(
                inputs, discrete_inputs=discrete_inputs
            )
            # run_solve_residuals returns (outputs, discrete_outputs) when the
            # server sends back discrete output data, and a plain dictionary
            # otherwise; the discrete outputs are not part of the grammars.
            if isinstance(states, tuple):
                states = states[0]

        residuals = self._client.run_compute_residuals(
            inputs, states, discrete_inputs=discrete_inputs
        )
        output_data = dict(states)
        for state_name, value in residuals.items():
            output_data[self._state_to_residual[state_name]] = value
        return output_data

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """Compute the Jacobians of the residuals and of the state variables.

        This sends the current input and state variable values to the remote
        Philote discipline server through the ``ComputeResidualGradients``
        RPC, reshapes the resulting flattened partial derivatives according
        to the shapes obtained by :meth:`._initialize_grammars`, and stores
        them in :attr:`.jac` under the residual names of the output grammar.

        Since a state variable is echoed from the input data to the output
        data, its Jacobian with respect to itself is the identity matrix and
        its Jacobian with respect to any other variable is zero.

        Args:
            input_names: The names of the inputs against which to differentiate the
                outputs. If empty, use all the inputs.
            output_names: The names of the outputs to be differentiated.
                If empty, use all the outputs.
        """
        # The remote discipline only sends the partial derivatives it
        # declares, so the requested Jacobian matrices are first filled with
        # zeros and then overwritten with the ones the server returns.
        self._init_jacobian(
            input_names, output_names, init_type=self.InitJacobianType.DENSE
        )
        for state_name in self._state_to_residual:
            if state_name in self.jac and state_name in self.jac[state_name]:
                self.jac[state_name][state_name] = eye(
                    int(prod(self._shapes[state_name]))
                )

        inputs, states, discrete_inputs = self._split_implicit_input_data(
            self.get_input_data()
        )
        jac_flat = self._client.run_residual_gradients(
            inputs, states, discrete_inputs=discrete_inputs
        )
        for (state_name, var_name), jac_val in jac_flat.items():
            residual_name = self._state_to_residual[state_name]
            res_size = int(prod(self._shapes[state_name]))
            var_size = int(prod(self._shapes[var_name]))
            self.jac[residual_name][var_name] = jac_val.reshape(res_size, var_size)
