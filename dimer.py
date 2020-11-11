import cspy.alpha.crystal as crystal
import os


class Dimer:
    
    def __init__(self, name, molecules):
        
        """
        Class to store pairs of molecules 
        
        Parameters
        ----------
        name : string
            name with format d_ + crystal_name_ + mol_1_name_ + mol_2_name_
        molecules : list [molecule_instance_1, molecule_instance_2]

        """
    
        assert len(molecules) == 2, "molecules must contain two molecules."
        self.name = name
        self.molecules = molecules
        self.xyz, self.atomic_numbers = self.get_atom_data()
        self.energies = {}

        
    def __getitem__(self, i):
        return self.molecules[i]
        
    def get_atom_data(self, separate_molecules=True):

        """
        Extracts xyz and atomic numbers from atoms in self.molecules.
        
        Parameters
        ----------
        separate_molecules : boolean, optional
            If true returns [[mol1_xyz_data], [mol2_xyz_data]], [[mol1_atomic_numbers], [mol2_atomic_numbers]]
            Else returns [xyz_data], [atomic_numbers_data] (ordered mol1 then mol2)
        
        Returns
        -------
        xyz_data : list of tuples [(x1,y1,z1), (x2, y2, z2), ..]
        atomic_numbers : list of ints
        
        """
        
        xyz_data_1 = []
        atomic_numbers_1 = []
        xyz_data_2 = []
        atomic_numbers_2 = []

        for atom in self.molecules[0].atoms:
            xyz_data_1.append(atom.xyz)
            atomic_numbers_1.append(atom.atomic_number)
        for atom in self.molecules[1].atoms:
            xyz_data_2.append(atom.xyz)
            atomic_numbers_2.append(atom.atomic_number)
        if separate_molecules:
            return [xyz_data_1, xyz_data_2], [atomic_numbers_1, atomic_numbers_2]
        else:
            return xyz_data_1 + xyz_data_2, atomic_numbers_1 + atomic_numbers_2
        

    def calculate_fit_energy(self, multipole_file, running_dir, 
                             axis_file='/scratch/dm1m15/year_3/projects/packing_opt/maleic_hydrazide/mh_dimer.axis', 
                             potential_file='/home/dm1m15/cspy/potentials/fit.pots'):
        
        """
        Calls energy_minimize and takes the first iteration.
        
        Parameters
        ----------
        multipole_file : string
            Path to multipole file required for energy calculation.
        running_dir : string
            Path to running directory required for energy calculation.
        axis_file : string, optional
            Path to axis file required for energy calculation.
        potential_file : string, optional
            Path to potential file required for energy calculation.
            
        Updates
        -------
        self.energies["FIT"], float
            FIT+DMA Intermolecular energy calculated using DMACRYS
        """
        
        assert isinstance(multipole_file, str), "multipole_file must be a file path as a string."
        assert isinstance(potential_file, str), "potential_file must be a file path as a string."
        assert isinstance(running_dir, str), "running_dir must be a file path as a string."
        
        dimer_crystal = crystal.CrystalStructure()
        dimer_crystal.latt = "-1"
        box_length = 50.0
        dimer_crystal.lattice.set_lattice_parameters([50,50,50,90.0,90.0,90.0])
        for mol in self.molecules:
            dimer_crystal.unique_molecule_list.append(mol)
        dimer_crystal.update_sfac(retyped=False)
        dimer_crystal.name = self.name
        dimer_crystal.filename="dimer.res"
        
        neighcrys_args = {'foreshorten_hydrogens': 'n', 'force_field_type': 'F', 
                          'limi': None, 'multipole_file': multipole_file,
                          'intermolecular_distance': 4.0, 'paste_coords_file': '', 
                          'limg': None, 'max_iterations': 0, 'neighcrys_input_mode': 'E', 
                          'dummy_atoms': False, 'paste_coords': 'n', 'step_size': 0.5, 
                          'standardise_bonds': 'n', 'vdw_cutoff': 35.0, 'dont_check_anisotropy': False, 
                          'potential_file': potential_file, 
                          'custom_atoms_anisotropy': 'F,Cl,Br,I', 'axis_file': axis_file, 
                          'pressure': '', 'max_search': 3, 'lattice_factor': 1.0, 
                          'bondlength_file': 'bond_cutoffs', 'custom_labels': None, 
                          'gdma_dist_tol': 1e-06, 'wcal_ngcv': 'n', 
                          'remove_symmetry_subgroup': ['0'], 'raise_before_neighcrys_call': False}

        gaussian_args = {'nprocs': '1', 'functional': 'B3LYP', 
                         'force_rerun': True, 'gaussian_all_molecules': False, 
                         'chelpg': False, 'set_molecular_states': None, 
                         'external_iteration_pcm': False, 'gaussian_cleanup': True, 
                         'memory': '1GB', 'dielectric_constant': '3.0', 
                         'basis_set': '6-311G**', 'basis_file': '', 
                         'opt': 'ModRedundant,', 'polarizable_continuum': False, 
                         'molecular_volume': False, 'freq': False, 'iso': '', 
                         'additional_args': ''}

        gdma_args = {'multipole_switch': 0.0, 'multipole_limit': 4}
        
        dmacrys_args = {'zip_unpack': 1, 'cutm': 20.0, 'clean_level': 5, 
                        'lennard_jones': False, 'lennard_jones_potential': None, 
                        'auto_neighbour_setting': False, 'remove_negative_eigens': False, 
                        'platon_unit_cell': False, 'raise_before_dmacrys_call': False, 
                        'new_ac': False, 'constant_volume': False, 'exact_prop': False, 
                        'assisted_convergence': False, 'set_real_space': False, 
                        'check_z_value': False, 'dmacrys_timeout': 3600.0, 
                        'spline_off': False, 'zip_level': 1, 'seig1': False, 
                        'nbur': 20, 'setup_off': False, 'setup_on': False, 'spline_on': False}



        dimer_crystal.energy_minimize(run_dir=running_dir, 
                                      neighcrys_args=neighcrys_args, 
                                      gaussian_args=gaussian_args, 
                                      gdma_args=gdma_args, 
                                      dmacrys_args = dmacrys_args)
        self.energies["FIT"] = dimer_crystal.data_summary.initial_energy
        os.remove(self.name + ".p")
        
    def same_dimer(dimer_1, dimer_2, rmsd_threshold=0.1):
        
        """
        Checks if two dimers are the same by comparing interatomic distances.
        
        Parameters
        ----------
        dimer_1 : Dimer instance
        dimer_2 : Dimer instance
        rmsd_threshold : float
            minimum difference between two interatomic distances
       
        Returns
        -------
        same_dimer : boolean
            True if each equivalent interatomic distance is the same 
        """
        
        assert len(dimer_1.molecules[0].atoms) == len(dimer_2.molecules[0].atoms) \
                and len(dimer_1.molecules[1].atoms) == len(dimer_2.molecules[1].atoms),\
            "Dimers are assumed to have the same kinds of monomers, \
            and so have the same number of atoms"
            
        for count_i, i in enumerate(dimer_1.molecules[0].atoms):
            for count_j, j in enumerate(dimer_1.molecules[1].atoms):
                d1 = i.distance_to(j)
                d2 = dimer_2.molecules[0].atoms[count_i].distance_to(dimer_2.molecules[1].atoms[count_j])
                if abs(d1-d2) > rmsd_threshold:
                    return False
        return True
        
    def gen_res_file(self, output_dir):

        """
        Generates a .res file of the dimer in a 50 angstrom box.
        Saved to output_dir + /self.name + .res

        Parameters
        ----------
        output_dir : string
            Directory path.

        Output
        ------
        .res file

        """
        dimer_crystal = crystal.CrystalStructure()
        dimer_crystal.latt = "-1"
        box_length = 50.0
        dimer_crystal.lattice.set_lattice_parameters([50,50,50,90.0,90.0,90.0])
        for mol in self.molecules:
            dimer_crystal.unique_molecule_list.append(mol)
        dimer_crystal.update_sfac(retyped=False)
        dimer_crystal.write_res_to_file(output_dir + "/" + self.name + ".res")
        
    def gen_com_file(self, output_dir, method="MP2", basis_set="6-31+G*", nproc=2, mem=3, d3_correction=False):

        """
        Generates a .com input file for a single-point energy calculation in Gaussian.
        Saved as output_dir + /self.name + .com

        Parameters
        ----------
        output_dir : string
            Directory path.
        method : string, optional
            Energy model used to calculate the dimer energy in Gaussian.
        basis_set : string, optional
            The basis set used in the energy calculation.
        nproc : string, optional
            The number of processes requested for the energy calculation.
        mem : string, optional
            The amount of memory in Gb requested for the energy calculation.
        d3_correction : boolean, optional
            If true, the Grimme dispersion correction (D3) is included in the 
            calculation.

        Outputs
        -------
        .com file

        Notes
        -----
        All .com files are generated with a counterpoise correction.
        The d3_correction is only appropriate for density functional methods.

        """

        with open(output_dir + "/" + self.name + ".com", "w") as com_file:
            com_file.write("%NProcShared=" + str(_nproc) +"\n")
            com_file.write("%mem=" + str(_mem) + "GB\n")
            input_line = "# " + _method + "/" + _basis_set
            if _dispersion:
                input_line+= " EmpiricalDispersion=GD3"
            input_line += " counterpoise = 2\n\n"
            com_file.write(input_line)
            com_file.write('single point energy calculation for ' + self.name + '\n\n')
            com_file.write('0,1 0,1 0,1\n')
            for count, molecule in enumerate(self.molecules):
                for line in molecule.xyz_string_form():
                    line_mod = line[0:6].replace(" ", "") + '(fragment='
                    "".join(line_mod.split())
                    line_mod += str(count + 1) + ')' + line[7:]
                    com_file.write(line_mod + '\n')
            com_file.write('\n')
