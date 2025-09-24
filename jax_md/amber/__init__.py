# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AMBER force field for JAX MD."""

from jax_md.amber.amber_energy import (
    amber_energy_fn,
    calculate_eem_charges_amber,
    prm_get_nonbond_pairs,
    prm_get_nonbond_terms,
    prm_get_nonbond14_pairs,
    distance,
    angle,
    torsion,
    torsion_single,
    bond_init,
    bond_get_energy,
    angle_init,
    angle_get_energy,
    torsion_init,
    torsion_get_energy,
    lj_init,
    lj_get_energy_nb,
    lj_get_energy_14,
    lj_get_energy,
    lj_get_energy_nbr,
    coul_init,
    coul_init_pme,
    coul_init_ewald,
    coul_get_energy_ewald,
    coul_get_energy_nb_ewald,
    coul_get_energy_nb,
    coul_get_energy_pme,
    coul_get_energy_nb_pme,
    coul_get_energy_14,
    coul_get_energy,
    bond_rest_get_energy,
    angle_rest_get_energy,
    torsion_rest_get_energy,
    rest_get_energy
)

from jax_md.amber.amber_forcefield import (
    BondRestraint,
    AngleRestraint,
    TorsionRestraint,
    AmberForceField,
    FFQForceField
)

from jax_md.amber.amber_helper import (
    GAFFTYPES,
    load_amber_ff,
    get_nonbond_pairs,
    angle as helper_angle,
    torsion as helper_torsion,
    torsion_v2,
    load_ffq_ff,
    move_dataclass
)

from jax_md.amber.amber_energy_v2 import (
    amber_energy,
    periodic_torsion,
    cmap_torsion,
    lennard_jones,
    lj_softcore,
    coulomb_recip,
    structure_factor,
    optimized_bspline_4,
    map_charges_to_grid,
    b,
    B,
    transform_gradients,
    calculate_eem_charges,
    linear_response
)