from functools import wraps
from pathlib import Path
from typing import Union, List
import os
import json
from multiprocessing import Pool
from pprint import pprint

import numpy as np
import openfoamparser_mai as Ofpp
import pyvista
from math import atan


def json_streaming_writer(filepath, data_func, data_args):
    """Write JSON data to a file using a generator to minimize memory usage."""
    data_gen = data_func(*data_args)
    try:
        with open(filepath, 'w') as file:
            print(f"writing {filepath}")
            file.write('[')
            for i, item in enumerate(data_gen):
                if i != 0:  # Add a comma before all but the first item
                    file.write(',')
                json.dump(item, file)
            file.write(']')
        print(f"Finished writing {filepath}")
    except Exception as e:
        print(f"Failed to write {filepath}: {str(e)}") 

def _face_center_position(points: list, mesh: Ofpp.FoamMesh) -> list:
    vertecis = [mesh.points[p] for p in points]
    vertecis = np.array(vertecis)
    return list(vertecis.mean(axis=0))

def pressure_field_on_surface(solver_path: Union[str, os.PathLike, Path],
                                 p: np.ndarray,
                                 surface_name: str = 'Surface') -> None:
    """Поле давлений на поверхности тела:
    'Nodes' - List[x: float, y: float, z:float], 
    'Faces' - List [List[int]], 
    'Elements' - List [Dict{Faces: List[int],
                            Pressure: float,
                            Velocity: List[float],
                            VelocityModule: float,
                            Position: List[float]}
                            ], 
    'Surfaces' - List[
                    Tuple[Surface_name: str, 
                    List[Dict{ParentElementID: int,
                              ParentFaceId: int,
                              Position: List[float]}]
                    ]

    Args:
        solver_path (Union[str, os.PathLike, Path]): Путь до папки с расчетом.
        p (np.ndarray): Поле давления.
        surface_name (str): Имя для поверхности.
    """
    
    # Step 0: parse mesh and scale vertices
    mesh_bin = Ofpp.FoamMesh(solver_path )

    # Step I: compute TFemFace_Surface
    domain_names = ["motorBike_0".encode('ascii')]
    surfaces = []

    for i, domain_name in enumerate(domain_names):
        bound_cells = list(mesh_bin.boundary_cells(domain_name))

        boundary_faces = []
        boundary_faces_cell_ids = []
        for bc_id in bound_cells:
            faces = mesh_bin.cell_faces[bc_id]
            for f in faces:
                if mesh_bin.is_face_on_boundary(f, domain_name):
                    boundary_faces.append(f)
                    boundary_faces_cell_ids.append(bc_id)

        f_b_set = set(zip(boundary_faces, boundary_faces_cell_ids))

        body_faces = []
        for f, b in f_b_set:
            try:
                position = _face_center_position(mesh_bin.faces[f], mesh_bin)
                d = {'ParentElementID': b,
                    'ParentFaceId': f,
                    'CentrePosition': {'X': position[0], 'Y': position[1], 'Z': position[2]},
                    'PressureValue': p[b]
                    }
                body_faces.append(d)
            except IndexError:
                print(f'Indexes for points: {f} is not valid!')

        surfaces.append({'Item1': surface_name,
                'Item2': body_faces}) 
        

        return surfaces
    
def get_flow(filename: Path):
    # filename = filename / Path("0.orig\\include\\initialConditions")
    # print(filename)
    with open(filename / Path("0.orig\\include\\initialConditions")) as f:
        a = [i for i in f if ("flowVelocity" in i)]
    s = a[0]
    s = [float(i) for i in s[s.index("(")+1:s.index(")")].split()]

    return ((s[0]**2 + s[1]**2)**0.5, atan(s[1]/s[0]))

def get_dataset(path: Path):

    i = path

    list_of_dirs = [j for j in os.listdir(i) if j.isdigit() and j != "0"]
    res_dir = list_of_dirs[0]
    
    p = Ofpp.parse_internal_field(i / Path(res_dir) / Path('p'))
    surfaces = pressure_field_on_surface(i , p)
    flow = get_flow(i)
    print(len(surfaces[0]["Item2"]))

    for s in surfaces[0]["Item2"]:    
        s["Velocity"] = flow[0]*0.003003
        s["Angle"] = flow[1]
        yield s

def get_datasets(path: Path):
    for i in path.iterdir():
        print(i, end=" - ")
        gen = get_dataset(i)
        for j in gen:
            yield j

def raw_streaming_writer(filepath, data_func, data_args):
    generator = data_func(*data_args)
    try:
        print(f"Started writing {filepath}")
        with open(filepath, "w+") as file:
            for i in generator:
                s = ' '.join(map(str, (i['CentrePosition']['X'],i['CentrePosition']['Y'],i['CentrePosition']['Z'],i['Angle'],i['Velocity'],i['PressureValue'])))
                file.write(s)
                file.write("\n")
        print(f"Finished writing {filepath}")
    except Exception as e:
        print(f"Failed to write {filepath}: {str(e)}")

def partial_streaming_writer(filepath, data_func, data_dirs):
    # try:
        print(f"Started writing {filepath}")
        for path in data_dirs:
            for calc in path.iterdir():
                # path.mkdir(parents=True, exist_ok=True)
                print(calc, end=" - ")
                gen = get_dataset(calc)
                with open(filepath / Path(calc.name), "w+") as file:
                    for i in gen:
                        s = ' '.join(map(str, (i['CentrePosition']['X'],i['CentrePosition']['Y'],i['CentrePosition']['Z'],i['Angle'],i['Velocity'],i['PressureValue'])))
                        file.write(s)
                        file.write("\n")
        print(f"Finished writing {filepath}")

