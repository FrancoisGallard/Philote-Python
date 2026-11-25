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
"""Two-way interoperability between CoSApp and Philote-MDO.

This subpackage requires CoSApp (``pip install philote-mdo[cosapp]``):

* :class:`~philote_mdo.cosapp.philote_to_cosapp.PhiloteSystem` wraps a remote
  Philote explicit discipline server as a CoSApp ``System``.
* :class:`~philote_mdo.cosapp.cosapp_to_philote.CoSAppToPhiloteDiscipline`
  wraps a CoSApp ``System`` as a Philote-MDO explicit discipline, so that it
  can be served to remote Philote clients.
"""

from .cosapp_to_philote import CoSAppToPhiloteDiscipline
from .philote_to_cosapp import PhiloteSystem
