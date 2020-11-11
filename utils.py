from cspy.core.io.res import ResCrystalMapper, ResFileWriter, ResFileReader
from numpy import sqrt, square
from cspy.core.models.molecule import Molecule
from cspy.alpha.molecule import Molecule as Molecule2
from copy import deepcopy

def get_dmacrys_args(potential_file, multipole_file=None, axis_file=None):

    """
    Creates dictionaries of default arguments needed to run a DMACRYS calculation.

    Parameters
    ----------
    potential_file : string
        path to a .pots file
    multipole_file : string, optional
        path to a multipole file
    axis_file, string, optional
        path to an axis file


    Returns
    -------
    neighcrys_args : dict 
        arguments required to run NEIGHCRYS
    gaussian_args : dict
        arguments required to run Gaussian
    gdma_args : dict
        arguments required to run GDMA
    dmacrys_args : dict
        arguments required to run DMACRYS


    Notes
    -----
    If multipole_file=None, a Gaussian calculation will be
    attempted before a DMACRYS calculation. 
    """

    neighcrys_args = {'foreshorten_hydrogens': 'n', 'force_field_type': 'F', 
                      'limi': None, 'multipole_file':  multipole_file, 
                      'intermolecular_distance': 4.0, 'paste_coords_file': '', 
                      'limg': None, 'max_iterations': 0, 'neighcrys_input_mode': 'E', 
                      'dummy_atoms': False, 'paste_coords': 'n', 'step_size': 0.5, 
                      'standardise_bonds': 'n', 'vdw_cutoff': 35.0, 
                      'dont_check_anisotropy': False, 
                      'potential_file': potential_file,
                      'custom_atoms_anisotropy': 'F,Cl,Br,I', 
                      'axis_file': axis_file, 'pressure': '', 
                      'max_search': 3, 'lattice_factor': 1.0, 
                      'bondlength_file': 'bond_cutoffs',
                      'custom_labels': None, 'gdma_dist_tol': 1e-06, 
                      'wcal_ngcv': 'n', 'remove_symmetry_subgroup': ['0'], 
                      'raise_before_neighcrys_call': False}

    gaussian_args = {'nprocs': '1', 'functional': 'B3LYP', 'force_rerun': True, 
                     'gaussian_all_molecules': False, 'chelpg': False, 
                     'set_molecular_states': None, 'external_iteration_pcm': False, 
                     'gaussian_cleanup': True, 'memory': '1GB', 
                     'dielectric_constant': '3.0', 'basis_set': '6-311G**', 
                     'basis_file': '', 'opt': 'ModRedundant,', 
                     'polarizable_continuum': False, 'molecular_volume': False, 
                     'freq': False, 'iso': '', 'additional_args': ''}


    gdma_args = {'multipole_switch': 0.0, 'multipole_limit': 4}
        
    dmacrys_args = {'zip_unpack': 0, 'cutm': 20.0, 'clean_level': 1, 
                    'lennard_jones': False, 'lennard_jones_potential': None, 
                    'auto_neighbour_setting': False, 'remove_negative_eigens': False, 
                    'platon_unit_cell': False, 'raise_before_dmacrys_call': False, 
                    'new_ac': False, 'constant_volume': False, 'exact_prop': False,
                    'assisted_convergence': False, 'set_real_space': False, 
                    'check_z_value': False, 'dmacrys_timeout': 3600.0, 
                    'spline_off': False, 'zip_level': 0, 'seig1': False, 
                    'nbur': 20, 'setup_off': False, 'setup_on': False, 
                    'spline_on': False}


    return neighcrys_args, gaussian_args, gdma_args, dmacrys_args


def res_string_to_crystal(res_string):

    """
    Converts a res string to new a new crystal object.

    Parameters
    ----------
    res_string : string
        .res file as a string.

    Returns
    -------
    crystal : Crystal
        Crystal instance.
    """

    res_content = ResFileReader(file_content=res_string).read()
    crystal = ResCrystalMapper().map_to_object(res_content)
    crystal.asymmetric_unit = fragment_geometry(crystal.asymmetric_unit[0])
    return crystal

def crystal_to_res_string(crystal):

    """
    Converts a Crystal object to a res string.

    Parameters
    ----------
    crystal : Crystal
        Crystal instance.

    Returns
    -------
    res_string : string
        .res file as a string.

    """
    tmpcrys = deepcopy(crystal)
    file_content = ResCrystalMapper().map_to_content(tmpcrys)
    res_content = ResFileWriter(file_content=file_content,
                                output_location='').res_string_form
    return res_content


def fragment_geometry(unit):

    """
    Ensures the molecules in unit are not fragments.

    Parameters
    ----------
    unit : Crystal
        Crystal instance.

    Returns
    -------
    mols : list of Molecule instances
        A list of complete molecules in the unit cell.
    
    """
    num_atoms = len(unit.atoms)
    mol_id = [n for n in range(num_atoms)]
    for i_atom in range(num_atoms-1):
        for j_atom in range(1, num_atoms):
            atom_1 = unit.atoms[i_atom]
            atom_2 = unit.atoms[j_atom]
            distance = calc_dis(atom_1, atom_2)
            pair = [atom_1.element.symbol,
                    atom_2.element.symbol,
                    distance]
            if pair_close(pair):
                i_find = max(mol_id[i_atom], mol_id[j_atom])
                i_set = min(mol_id[i_atom], mol_id[j_atom])
                for i_mol in range(len(mol_id)):
                    if mol_id[i_mol] == i_find:
                       mol_id[i_mol] = i_set

    index = {}
    mol_index = 0
    mol_lists = []
    for i_mol in set(mol_id):
        index[i_mol] = mol_index
        mol_index += 1
        mol_lists.append([])

    for i_atom, i_mol in enumerate(mol_id):
        mol_lists[index[i_mol]].append(unit.atoms[i_atom])

    mols = []
    crystal = unit._crystal
    for mol_list in mol_lists:
        mols.append(Molecule(atoms=mol_list, crystal=crystal))

    return mols


def calc_dis(atom1, atom2):
    dis_x = atom1.xyz[0] - atom2.xyz[0]
    dis_y = atom1.xyz[1] - atom2.xyz[1]
    dis_z = atom1.xyz[2] - atom2.xyz[2]
    distance = square(dis_x) +\
               square(dis_y) +\
               square(dis_z)
    return sqrt(distance)


def pair_close(pair):
    default_bond_cutoff = Molecule2.bonds_dictionary
    label_1 = pair[0]
    label_2 = pair[1]
    dis = pair[2]
    for pair_ref, dis_ref in default_bond_cutoff.items():
        if (((label_1 == pair_ref[0]) and (label_2 == pair_ref[1])) or
            ((label_1 == pair_ref[1]) and (label_2 == pair_ref[0]))):
                if dis <= dis_ref:
                    return True
                return False
