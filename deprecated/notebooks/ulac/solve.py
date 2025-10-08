import pickle
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import igl
import numpy as np
import pyvista as pv


# ==================================================================================================
def extract_uac_boundary_data(
    boundary_names: list[str], *uac_data: Any
) -> tuple[np.ndarray, np.ndarray]:
    uac_data_dict = {}
    for data in uac_data:
        uac_data_dict.update(asdict(data))

    boundary_indices = []
    boundary_values = []
    for boundary_name in boundary_names:
        boundary = uac_data_dict[boundary_name]
        boundary_indices.append(boundary["inds"])
        boundary_values.append(np.vstack([boundary["alpha"], boundary["beta"]]).T)

    boundary_indices = np.concatenate(boundary_indices)
    boundary_values = np.vstack(boundary_values)
    _, unique_mask = np.unique(boundary_indices, axis=0, return_index=True)
    boundary_indices = boundary_indices[unique_mask]
    boundary_values = boundary_values[unique_mask]

    return boundary_indices, boundary_values
