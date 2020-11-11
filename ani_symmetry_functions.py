from cspy.alpha.crystal import CrystalStructure
from cspy.alpha.molecule import Molecule
import numpy as np
import copy
import math
import multiprocessing
from functools import partial
from contextlib import contextmanager
from code.packing_opt.dimer import Dimer

def sym_worker(dimer, rc, ra, etar, etaa, rsr_params, rsa_params, ts_params, zeta, include_intramolecular, separate_radial_by_element, separate_angular_by_element):
    symmetry_functions = get_symmetry_functions(dimer, rc, ra, etar, etaa, rsr_params, rsa_params, ts_params, zeta, include_intramolecular,
                                                separate_radial_by_element, separate_angular_by_element)
    return symmetry_functions


def sym_worker_unpack(args):
    return sym_worker(*args)


def get_symmetry_functions_array(dimers, rc, ra, etar, etaa, rsr_params, rsa_params, ts_params, zeta,
                                 include_intramolecular=True, separate_radial_by_element=True,
                                 separate_angular_by_element=True, num_workers=8):
    @contextmanager
    def poolcontext(*args, **kwargs):
        pool = multiprocessing.Pool(*args, **kwargs)
        yield pool
        pool.terminate()

    with poolcontext(processes=num_workers) as p:
        symmetry_functions_array = p.map(sym_worker_unpack,
                                         [[dimer, rc, ra, etar, etaa, rsr_params, rsa_params, ts_params, zeta, include_intramolecular,
                                           separate_radial_by_element, separate_angular_by_element] for dimer in dimers])
    return symmetry_functions_array


def get_symmetry_functions(dimer, rc, ra, etar, etaa, rsr_params, rsa_params, ts_params, zeta, include_intramolecular=True, separate_radial_by_element=False, separate_angular_by_element=False):

    """
    Based on descriptor described in:
    ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost
    J.S Smith, O. Uslayev and A. E. Roitberg (2017)

    dimer : dimer instance
    rc : float, radial_cutoff
    ra : float, radial_cutoff
    rsr_params : list, [radial rs values]
    rsa_params : list, [angular rs values]
    ts_params : list, [ts values]
    etar : float, radial eta
    etaa :  float, angular eta
    zeta : float

    Returns a list of length (len(rsr_params) + (len(rsa_params)*len(ts_params))) * number of atoms in a given molecule of dimer
    If by_element, symmetry functions are generated for each atom type in a given molcule, multiplying this number by the number of atom types 
    [atom1_radial_function1, atom1_radial_function2, atom1_angular_function1, atom1_angular_function2, atom2_radial_function1, etc.]
    Maintains ordering of res file
    """

    def calc_radial_function(mol_xyz, cm_atom_xyz, rc, r_s_params, etar, mol_labels=None, atom_types=None):

        """
        Radial function of ANI-1
        For a given atom returns a list of radial functions of length r_s_params

        mol_xyz : list of xyz of all atoms in a finite cluster larger than cutoff not in the central molecule
        cm_atom_xyz : xyz of a given atom in the central molecule
        r_c : radial cutoff
        r_s_params : list of hyperparameters controlling position of each gaussian
        eta : hyperparameter controlling width of each gaussian
        mol_labels : list of atom labels for each atom in mol_xyz
        atom_types : list of atom possible atom labels, if not None, symmetry functions are generated for each atom_type
        """
	
        def Gr(eta, r, r_s, r_cut):
            return np.exp(-eta*np.square(r-r_s)) * ((np.cos((np.pi*r)/r_cut)+1)*.5)


        distances = np.linalg.norm(mol_xyz-cm_atom_xyz, axis=1)
        sorted_distances = np.sort(distances)
        
        # Separated by element
        if mol_labels is not None and atom_types is not None:
            sorted_idxs = np.argsort(distances)
            sorted_mol_labels = mol_labels[sorted_idxs[::]]
            G1 = np.zeros((len(rsr_params),len(atom_types)))
            for count, r in enumerate(sorted_distances[1:np.argmax(sorted_distances>rc)]):
                for r_s_idx, r_s in enumerate(rsr_params):
                    for atom_type in range(len(atom_types)):
                        if atom_types[atom_type] == sorted_mol_labels[count+1]:
                            G1[r_s_idx][atom_type] += Gr(etar, r, r_s, rc)

            return G1

        # Not separated by element
        G1 = [0 for i in range(len(rsr_params))]
        max_idx = np.argmax(sorted_distances>rc)
        if max_idx != 0:
            sorted_distances = sorted_distances[1:max_idx]
        else:
            sorted_distances = sorted_distances[1:]
        for r in sorted_distances:
            for r_s_idx, r_s in enumerate(rsr_params):         
                G1[r_s_idx] += Gr(etar, r, r_s, rc)
        return G1

    def calc_angular_function(mol_xyz, cm_atom_xyz, ra, etaa, zeta, rsa_params, ts_params, mol_labels=None, atom_type_pairs=None):

        """
        Angular function of ANI-1
        For a given atom returns a list of angular functions of length rsa_params * ts_params

        mol_xyz : list of xyz of all atoms in a finite cluster larger than r_a not in the central molecule
        cm_atom_xyz : xyz of a given atom in the central molecule
        ra : angular cutoff function
        rs_params : list of hyperparameters controlling position of each gaussian
        ts_params: list of hyperparamters controlling the position of each gaussian
        etaa : hyperparameter controlling width of each gaussian
        zeta : hyperparameter controlling the width of each gaussian
        mol_labels : list of atom labels for each atom in mol_xyz
        atom_type_pairs : list of atom pairs for a given symmetry function, if not None, angular functions are returned for every atom pair
        (atom_type_pairs only include unique pairs, e.g. only one of ['H','C'] and ['C', 'H'])
        """
	
        def Ga(etaa, t, r_s, ra, t_s, zeta, r1, r2):
            return np.power((1+np.cos(t - t_s)), zeta) * np.exp(-etaa*(np.square(((r1 + r2)*.5)-r_s)))  \
	            * ((np.cos((np.pi*r1)/ra)+1)*.5) *((np.cos((np.pi*r2)/ra)+1)*.5)
	
        def three_points_angle(xyz_1, xyz_2, xyz_3):

            ba = xyz_2 - xyz_1
            ca = xyz_3 - xyz_1
            cosine_angle = np.clip(np.dot(ba, ca) / (np.linalg.norm(ba) * np.linalg.norm(ca)),-1,1)
            angle = np.arccos(cosine_angle)
            ref = np.array([1,1,1])
            normal = np.cross(ba, ca)
            if np.dot(normal, ref) < 0.0:
                angle = -angle
            return angle

        def is_atom_pair(l1, l2, pair):
            if [l1,l2] == pair:
                return True
            if [l2,l1] == pair:     
                return True
            return False

        distances = np.linalg.norm(mol_xyz-cm_atom_xyz, axis=1)
        sorted_distances = np.sort(distances)
        sorted_idxs = np.argsort(distances)
        max_idx = np.argmax(sorted_distances>ra)
        if max_idx != 0:
            important_distances = sorted_distances[1:max_idx]
            sorted_mol_xyz = mol_xyz[sorted_idxs[::]][1:max_idx]
        else:
            important_distances = sorted_distances[1:]
            sorted_mol_xyz = mol_xyz[sorted_idxs[::]][1:]
 
        # Separated by element
        if mol_labels is not None and atom_type_pairs is not None:
            G2 = np.zeros((len(rsa_params)*len(ts_params),len(atom_type_pairs)))
            if max_idx != 0:
               sorted_mol_labels = mol_labels[sorted_idxs[::]][1:np.argmax(sorted_distances>ra)]
            else:
               sorted_mol_labels = mol_labels[sorted_idxs[::]][1:]
            for j in range(len(important_distances) - 1):
                for k in range(j+1, len(important_distances)):
                    count=0
                    t = three_points_angle(cm_atom_xyz, sorted_mol_xyz[j], sorted_mol_xyz[k]) 
                    for r_s_idx in range(len(rsa_params)):
                        for t_s_idx in range(len(ts_params)):
                            for atom_type_pair in range(len(atom_type_pairs)):
                                if is_atom_pair(sorted_mol_labels[j], sorted_mol_labels[k], atom_type_pairs[atom_type_pair]):
                                    G2[count][atom_type_pair] += Ga(etaa, t, rsa_params[r_s_idx], ra, ts_params[t_s_idx], zeta, important_distances[j], important_distances[k])
                                    break
                            count+=1
            G2 = G2*np.power(2,1.-zeta)
            return G2

        # Not separated by element
        G2 = [0 for i in range(len(rsa_params) * len(ts_params))]
        for j in range(len(important_distances) - 1):
            for k in range(j+1, len(important_distances)):
                count=0
                t = three_points_angle(cm_atom_xyz, sorted_mol_xyz[j], sorted_mol_xyz[k])
                for r_s_idx in range(len(rsa_params)):
                    for t_s_idx in range(len(ts_params)):
                        G2[count] += Ga(etaa, t, rsa_params[r_s_idx], ra, ts_params[t_s_idx], zeta, important_distances[j], important_distances[k])
                        count += 1
        G2 = [np.power(2,1.-zeta)*G for G in G2]
        return G2

    xyz_data, labels = dimer.get_atom_data(separate_molecules=False)
    monomer_size = int(len(xyz_data)/2)
    # Get xyz data
    atom_types = list(set(labels))
    atom_type_pairs = []
    for i in range(len(atom_types)):
        for j in range(i,len(atom_types)):
            atom_type_pairs.append([atom_types[i],atom_types[j]])

    funcs = np.zeros((monomer_size,),np.dtype([('label', np.unicode_, 3),('radial', np.float64, (len(rsr_params),)),('angular', np.float64, (len(rsa_params)*len(ts_params)))]))
    if separate_radial_by_element and separate_angular_by_element:
        funcs = np.zeros((monomer_size,),np.dtype([('label', np.unicode_, 3),('radial', np.float64, (len(rsr_params),len(atom_types))),('angular', np.float64, \
                (len(rsa_params)*len(ts_params),len(atom_type_pairs)))]))
    elif separate_radial_by_element:
        funcs = np.zeros((monomer_size,),np.dtype([('label', np.unicode_, 3),('radial', np.float64, (len(rsr_params),len(atom_types))),('angular', np.float64, \
                (len(rsa_params)*len(ts_params),))]))
    elif separate_angular_by_element:
        funcs = np.zeros((monomer_size,),np.dtype([('label', np.unicode_, 3),('radial', np.float64, (len(rsr_params),)),('angular', np.float64, \
                (len(rsa_params)*len(ts_params),len(atom_type_pairs)))]))

    funcs[:]['label'] = labels[:monomer_size]
    #funcs[:]['label'] = labels
    mol_xyz = np.asarray(xyz_data)
    mol_labels = np.asarray(labels)

    for idx, cm_atom_xyz in enumerate(mol_xyz):
        if idx >= monomer_size:
            break
        if include_intramolecular:
            if separate_radial_by_element:
                funcs[idx]['radial'] = calc_radial_function(mol_xyz, cm_atom_xyz, rc, rsr_params, etar, mol_labels=mol_labels, atom_types=atom_types)
            else:
                funcs[idx]['radial'] = calc_radial_function(mol_xyz, cm_atom_xyz, rc, rsr_params, etar)
            if separate_angular_by_element:
                funcs[idx]['angular'] = calc_angular_function(mol_xyz, cm_atom_xyz, ra, etaa, zeta, rsa_params, ts_params, mol_labels=mol_labels, atom_type_pairs=atom_type_pairs)
            else:
                funcs[idx]['angular'] = calc_angular_function(mol_xyz, cm_atom_xyz, ra, etaa, zeta, rsa_params, ts_params)
        else:
            if separate_radial_by_element:
                funcs[idx]['radial'] = calc_radial_function(mol_xyz, cm_atom_xyz, rc, rsr_params, etar, mol_labels=mol_labels, atom_types=atom_types)
            else:
                funcs[idx]['radial'] = calc_radial_function(mol_xyz, cm_atom_xyz, rc, rsr_params, etar)
            if separate_angular_by_element:
                funcs[idx]['angular'] = calc_angular_function(mol_xyz, cm_atom_xyz, ra, etaa, zeta, rsa_params, ts_params, mol_labels=mol_labels, atom_type_pairs=atom_type_pairs)
            else:
                funcs[idx]['angular'] = calc_angular_function(mol_xyz, cm_atom_xyz, ra, etaa, zeta, rsa_params, ts_params)
    
    return flatten_symmetry_function(funcs, separate_radial_by_element, separate_angular_by_element)

def flatten_symmetry_function(symmetry_function, radial_separated_by_element=True, angular_separated_by_element=True):
    
    """
    Assumes the form sym_f[label]["radial"]["angular"]
    if separated_elements, radial arrays are assumed to be 2D (num_functions,num_atom_types),
    angular arrays are assumed to be 2D (num_functions, num_atom_pairs)
    """
    
    flattened_array = []
    for label in range(len(symmetry_function)):
        
        if radial_separated_by_element:
            for r in range(symmetry_function[label]['radial'].shape[0]):
                for atom_type in range(symmetry_function[label]['radial'].shape[1]):
                    flattened_array.append(symmetry_function[label]['radial'][:,atom_type][r])
        else:
            for r in range(symmetry_function[label]['radial'].shape[0]):
                flattened_array.append(symmetry_function[label]['radial'][r])
                
        if angular_separated_by_element:
            for a in range(symmetry_function[label]['angular'].shape[0]):
                for atom_type_pair in range(symmetry_function[label]['angular'].shape[1]):
                    flattened_array.append(symmetry_function[label]['angular'][:,atom_type_pair][a])
        else:
            for a in range(symmetry_function[label]['angular'].shape[0]):
                flattened_array.append(symmetry_function[label]['angular'][a])
    return flattened_array
 
