from cspy.alpha.crystal import Molecule, CrystalStructure
from dimer import Dimer
import numpy as np
import copy
import os
import shutil
import logging
from math import ceil

class MbCrystal:
    
    def __init__(self, name, res_string):

        """    
        
        Class to store dimers of a crystal.
        
        Parameters
        ----------
        name : string
            name of crystal structure
        res_string : string
            .res file as a string

        """
    
        assert isinstance(res_string, str), "res_string must be a string."

        self.res_string = res_string
        self.name = name
        self.unique_dimers = None # [dimer]
        self.mapped_dimer_names = None  # {dimer_name : unique_dimer_name}
        self.converged_distance = None
        
        self.logger = self.setup_basic_logger()
        
    def setup_basic_logger(self):
        logging.basicConfig()
        logger = logging.getLogger('MbCrystal_logger')
        c_handler = logging.StreamHandler()
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        logging.basicConfig(level=logging.CRITICAL)
        logger.addHandler(c_handler)
        return logger
        
    
    def make_finite_cluster(self, cutoff, expansion_level=7, filename=None):

        """
        Creates a finite cluster out to cutoff, based on the unit cell of self.
        iffilename is None, returns crystal object of finite cluster
        else finite cluster is saved to filename as a res file.
        
        Parameters
        ----------
        cutoff : float
            Distance in angstroms to make a cluster from a reference molecule
        expansion_level : int, optional
            The number of periodic images to generate when making a cluster
            (3 is generally sufficient for small systems)
        filename : string, optional
            If filename != None, the cluster will be output as a .res file
            to filename. 
            
        Returns
        -------
        cluster : CrystalStructure
            Large unit cell CrystalStructure if filename=None
        res_file : .res file
            Large unit cell .res file if filename!=None
        
        Notes
        -----
        expansion_level is able to be set by the user as this is the bottleneck
        in creating a finite cluster. For cubic cells expansion_level=3 is 
        generally sufficient. Smaller values risk creating an uneven cluster, 
        especially with flat cells.
        filename is useful for debugging purposes but is not used otherwise.
        
        """
        
        if filename != None:
            assert isinstance(filename, str), "filename must be given as a string."
        assert isinstance(cutoff, float) or isinstance(cutoff, int), "cutoff must be a float or int."
        assert cutoff > 0, "cutoff must be positive."
        assert isinstance(expansion_level, int), "expansion_level must be an int"
        assert expansion_level > 1, "expansion level must be a positive value >= 1"
        

        def translate_mol(mol_1, translation, origin):

            """
            Takes a Molecule position and returns a new 
            Molecule at a translated position.
            
            Parameters
            ----------
            mol_1 : Molecule
                Molecule object to be translated.
            translation : tuple
                Translation to be applied to mol_1
            origin : tuple
                Centroid of the reference (central) Molecule
                
            Returns
            -------
            translated_mol : Molecule
                Molecule object in translated position.
                
            """

            translated_mol = Molecule()
            for atom_1 in mol_1.atoms:
                atom_1.xyz = atom_1.xyz + (np.add(np.ravel(translation[0, :]), origin))
                translated_mol.add_atom(atom_1)
            return translated_mol

        # make primitive cell
        tmpcrys = CrystalStructure()
        tmpcrys.init_from_res_string(self.res_string)
        tmpcrys.expand_to_primitive_unit_cell()

        # expand out periodic images to make sure the cell has full molecules
        mol = Molecule()
        for a in range(len(tmpcrys.unique_molecule_list)):
            for b in range(len(tmpcrys.unique_molecule_list[a].atoms)):
                mol.add_atom(tmpcrys.unique_molecule_list[a].atoms[b])
        del tmpcrys.unique_molecule_list[:]
        tmpcrys.unique_molecule_list = mol.fragment_geometry(lattice=tmpcrys.lattice, expansion_level=expansion_level)

        # get reference molecule in unit cell
        reference_mol = [list(tmpcrys.unique_molecule_list[0].centroid())]
        reference_mol_label = ["0"]
        
        # get all molecules in unit cell for comparison
        comparison_mols = []
        comparison_mol_labels = []
        for count, mol in enumerate(tmpcrys.unique_molecule_list):
            comparison_mol_labels.append(count)
            comparison_mols.append(list(mol.centroid()))

        # get all the molecule positions in periodic images out to expansion level
        translation = np.array(tmpcrys.lattice.make_all_translations(expansion_level))
        translation_matrix = np.tile(np.matrix(translation), (len(comparison_mols) * len(reference_mol), 1))

        # array of coordinates in reference_mol
        reference_atoms = np.repeat(reference_mol, len(comparison_mols) * len(translation), axis=0)

        # array of coordinates in comparison_mols
        reference_comparisons = np.tile(np.repeat(comparison_mols, len(translation), axis=0), (len(reference_mol), 1))

        # array of all labels
        labels = np.ravel(
            np.tile(np.repeat(comparison_mol_labels, len(translation), axis=0), (len(reference_mol_label), 1)))

        # calculate distance squared
        (x2, y2, z2) = np.split(np.square(np.subtract(reference_atoms,
                                                      np.add(reference_comparisons, translation_matrix))), 3, axis=1)
        # calculate euclidean distance
        distance = np.ravel(np.sqrt(np.add(x2, np.add(y2, z2))))

        # get indices of all molecules within cutoff
        indices = (list(np.ravel(np.where(np.subtract(distance, cutoff) <= 0.0))))
        # create new crystal structure to store finite cluster
        cluster = CrystalStructure()
        cluster.latt = "-1"
        cluster_length = 1000.0

        # place cluster in large box
        cluster.lattice.set_lattice_parameters([cluster_length, cluster_length, cluster_length, 90.0, 90.0, 90.0])

        # set origin
        origin = np.subtract([cluster_length / 2.0] * 3, list(tmpcrys.unique_molecule_list[0].centroid()))
        
        # create the cluster
        for i in range(len(indices)):
            mol = translate_mol(copy.deepcopy(tmpcrys.unique_molecule_list[labels[indices[i]]]),
                                translation_matrix[indices[i]], origin)
            cluster.unique_molecule_list.append(mol)
        cluster.update_sfac(retyped=False)
        if filename:
            cluster.write_res_to_file(filename)
        else:
            return cluster

    def get_molecule_list(self, cutoff):

        """
        Generates a list of Molecule objects from a finite cluster
        based on expanding self.res_string
        
        Parameters
        ----------
        cutoff : float
            Distance in angstroms to make a cluster from a reference molecule
            
        Returns
        -------
        central_molecule : Molecule
            Reference (central) molecule
        sorted_molecule_list : list of Molecule objects
            All other molecules in the cluster sorted by centroid-centroid
            euclidean distance to central_molecule
            
        """
        
        assert isinstance(cutoff, float) or isinstance(cutoff, int), "cutoff must be a float or int."
        assert cutoff > 0, "cutoff must be positive."

        cluster = self.make_finite_cluster(cutoff)

        # print("Getting molecule list..")
        # Reset origin and identify central molecule (origin is 500,500,500)
        min_distance = -1
        cm_idx = 0
        
        # Reset origin from the cluster unit cell to (0,0,0)
        # Identify central molecule in the cluster
        origin = [500, 500, 500]
        for count, molecule in enumerate(cluster.unique_molecule_list):
            distance = np.sqrt(np.square(molecule.centroid()[0] - 500)
                               + np.square(molecule.centroid()[1] - 500)
                               + np.square(molecule.centroid()[2] - 500))
            if distance < min_distance or min_distance == -1:
                min_distance = distance
                cm_idx = count
            for atom_idx in range(len(molecule.atoms)):
                cluster.unique_molecule_list[count].atoms[atom_idx].xyz[0] -= 500
                cluster.unique_molecule_list[count].atoms[atom_idx].xyz[1] -= 500
                cluster.unique_molecule_list[count].atoms[atom_idx].xyz[2] -= 500

        central_molecule = cluster.unique_molecule_list[cm_idx]
        molecule_list = cluster.unique_molecule_list[:]
        del molecule_list[cm_idx]

        # sort neighbours by distance from central_molecule
        neighbour_distances = []
        for mol in molecule_list:
            distance = np.sqrt(np.square(central_molecule.centroid()[0] - mol.centroid()[0]) + np.square(
                central_molecule.centroid()[1] - mol.centroid()[1]) + np.square(
                central_molecule.centroid()[2] - mol.centroid()[2]))
            neighbour_distances.append([mol, distance])

        neighbour_distances = sorted(neighbour_distances, key=lambda x: x[1])
        sorted_molecule_list = [i[0] for i in neighbour_distances]

        return central_molecule, sorted_molecule_list
    
    
    def get_all_dimers(self, central_molecule, sorted_molecule_list, only_involving_cm):
        
        """
        Identifies all dimers in sorted_molecule_list.
        
        Parameters
        ----------
        central_molecule : Molecule
            Molecule instance of the central molecule in a cluster
        sorted_molecule_list : list of Molecule instances
            All molecules in a cluster other than the central molecule
        only_involving_cm : boolean, optional
            if True, only dimers that have one monomer as the central molecule are retained
            else returns all pairs in the cluster.
        """
        
        assert isinstance(central_molecule, Molecule), "central_molecule must be a Molecule instance"
        assert isinstance(sorted_molecule_list, list), "sorted_molecule_list must be a list"
        
        # Retain all dimers involving the central molecule
        all_dimers = []
        for count, molecule in enumerate(sorted_molecule_list):
            all_dimers.append(Dimer("d_" + self.name + "_cm_" + str(count), [central_molecule, molecule]))

        # Retain all dimers not involving the central molecule
        if not only_involving_cm:
            for i in range(len(sorted_molecule_list) - 1):
                for j in range(i + 1, len(sorted_molecule_list)):
                    all_dimers.append(Dimer("d_" + self.name + "_" + str(i) + "_" + str(j),
                                            [sorted_molecule_list[i], sorted_molecule_list[j]]))
                    
        return all_dimers
    
    
    def get_unique_dimers(self, dimers, rmsd_threshold=0.1):
        
        """
        Identifies all unique dimers in a list of dimers by
        comparing intermolecular atom distances.
        
        Parameters
        ----------
        dimers : list of Dimer instances
        
        Returns
        -------
        unique_dimers : dictionary {dimer_name : dimer}
            all dimers considered to be unique
        mapped_dimer_names : dictionary {dimer_name : unique_dimer_name}
            all dimer names and their corresponding unique dimer name that they map to
        rmsd_threshold : float, optional
            minimum interatomic distance between equivalent atom pairs in two dimers
            for them to be considered the same.

        Notes
        -----
        Intermolecular atom distance comparisons assume both molecules are the same
        with the same atom ordering.
        mapped_dimer_names include all unique_dimers, which are just mapped to themselves.
        """
        
        # Identify unique dimers
        remaining_dimers = dimers[:]
        unique_dimers = {}
        mapped_dimers = {}
        while len(remaining_dimers) > 1:
            name = remaining_dimers[0].name
            unique_dimers[remaining_dimers[0].name] = remaining_dimers[0]
            mapped_dimers[remaining_dimers[0].name] = remaining_dimers[0].name
            del remaining_dimers[0]
            duplicate_dimer_idxs = []
            for count, dimer in enumerate(remaining_dimers):
                if Dimer.same_dimer(dimer, unique_dimers[name], rmsd_threshold):
                    mapped_dimers[dimer.name] = name
                    duplicate_dimer_idxs.append(count)
            duplicate_dimer_idxs = sorted(duplicate_dimer_idxs, reverse=True)
            for i in duplicate_dimer_idxs:
                del remaining_dimers[i]
        if len(remaining_dimers) == 1:
            unique_dimers[remaining_dimers[0].name] = remaining_dimers[0]
            mapped_dimers[remaining_dimers[0].name] = remaining_dimers[0].name
            del remaining_dimers[0]

        return unique_dimers, mapped_dimers
    
    
    def populate_dimers(self, cutoff, only_involving_cm=True,
                        rmsd_threshold=0.1):

        """
        Updates self.unique_dimers with all unique dimers in a finite cluster of size cutoff.
        
        
        Parameters
        ----------
        cutoff : float
            Distance in angstroms to make a cluster from a reference molecule.
        only_involving_cm : boolean, optional
            if True, only dimers that have one monomer as the central molecule are retained
            else returns all pairs in the cluster.
        rmsd_threshold : float, optional
            Threshold within which interatomic distances are considered the same
            
        Updates
        -------
        self.unique_dimers : list of Dimer objects
            All unique dimers in the cluster
        self.mapped_dimer_names : dict {string : string}
            All dimers with their name mapped to the corresponding name of the
            unique dimer
        
        Notes
        -----
        Dimer naming convention: d_crystalname_moleculenumber_moleculenumber,
        (moleculenumber = cm if central molecule).
        only_involving_cm is required to be false only when calling populate_trimers.
        rmsd_threshold is used to identify which dimers are unique.
        """
        
        assert isinstance(cutoff, float) or isinstance(cutoff, int), "cutoff must be a float or int."
        assert cutoff > 0, "cutoff must be positive."
        central_molecule, sorted_molecule_list = self.get_molecule_list(cutoff)

        all_dimers = self.get_all_dimers(central_molecule, 
                                         sorted_molecule_list, 
                                         only_involving_cm)
        
        unique_dimers, mapped_dimer_names = self.get_unique_dimers(all_dimers, rmsd_threshold)

        self.unique_dimers = [unique_dimers[i] for i in unique_dimers]
        self.mapped_dimer_names = mapped_dimer_names
        
    def populate_dimers_by_convergence(self, energy_threshold, multipole_file, num_rolling_avg=5,
                                       min_atom_contact_limit=8, max_atom_contact_limit=15,
                                       rmsd_threshold=2.0):
        """
        Identifies the distance where the sum of two-body energy FIT+DMA energies converges.
        
        Parameters
        ----------
        energy_threshold : float
            Difference in kJ/mol of the energy variance between two dimer shells where 
            the energy is considered converged
        multipole_file : string
            Path to multipole file required for energy calculation.
        num_rolling_int : int, optional
            The number of dimer shells to used to calculate the variance
        min_atom_contact_limit : float, optional
            The minimum distance to nearby dimers to start checking for convergence
        max_atom_contact_limit : float, optional
            The maximum distance to nearby dimers to check for convergence
        rmsd_threshold : float, optional
            Threshold within which sum of atom-atom distances are considered the same
            
        Updates
        -------
        self.converged_distance : float
            The distance at which the dimer sum energy converges
            
        Notes
        -----
        Convergence is assessed by checking the variance of the energy with each additional shell.
        If the variance falls below energy_threshold, and the additional shell is beyond 
        min_atom_contact_limit, the energy is assumed to be converged.
        max_atom_contact_limit is required for systems e.g. with a large dipole, that can have
        oscillating dimer sums. In these cases the energy is taken to be converged at 
        max_atom_contact_limit.
        
        """
        
        assert isinstance(energy_threshold, float) or isinstance(cutoff, int), \
        "energy_threshold must be a float or int."
        assert energy_threshold > 0, "energy_threshold must be positive."
        assert isinstance(multipole_file, str), "multipole_file must be a file path as a string."
        assert isinstance(min_atom_contact_limit, float) or isinstance(min_atom_contact_limit, int), \
        "min_atom_contact_limit must be a float or int."
        assert min_atom_contact_limit > 0, "min_atom_contact_limit must be positive."
        assert isinstance(max_atom_contact_limit, float) or isinstance(max_atom_contact_limit, int), \
        "max_atom_contact_limit must be a float or int."
        assert max_atom_contact_limit > 0, "max_atom_contact_limit must be positive."
        assert max_atom_contact_limit > min_atom_contact_limit, \
        "max_atom_contact_limit must be greater than min_atom_contact_limit"

        def get_nearest_atom_distance(dimer):
            
            """
            Identifies the smallest distance between two atoms of from
            different monomers of a dimer.
            
            Parameters
            ----------
            dimer : Dimer
                Dimer instance.
                
            Returns
            -------
            smallest_distance : float
                Euclidean distance between two nearest intermolecular atoms.
                
            """
            
            smallest_distance = -1
            for atom_i in dimer.molecules[0].atoms:
                for atom_j in dimer.molecules[1].atoms:
                    distance = atom_i.distance_to(atom_j)
                    if distance < smallest_distance or smallest_distance == -1:
                        smallest_distance = distance
            return smallest_distance

        def get_variance(energy_list):
            
            """
            Calculates the variance of a list of energies.
            
            Parameters
            ----------
            energy_list : list of floats
                Energies of each dimer shell
                
            Returns
            -------
            variance : float

            """
    
            return sum([np.square(i) for i in energy_list]) / float(len(energy_list)) - np.square(
                sum(energy_list) / float(len(energy_list)))

        # Expensive way of getting molecule list by atom_contact_limit by just making 
        # a large cluster by centroid-centroid distances first
        central_molecule, sorted_molecule_list = self.get_molecule_list(max_atom_contact_limit + 10)

        all_dimers = self.get_all_dimers(central_molecule, 
                                         sorted_molecule_list, 
                                         only_involving_cm=True)
        
        unique_dimers, mapped_dimer_names = self.get_unique_dimers(all_dimers)

        self.energy_sums = []
        self.variances = []
        self.averages = []
        energy_sum = 0
        distances = [i / 10. for i in range(int(ceil(max_atom_contact_limit)) * 10)]
        calculated_unique_dimer_energies = {}
        important_mapped_dimers = {}
        important_unique_dimers = {}
        remaining_dimers = dict(mapped_dimer_names)

        running_dir = "tmp_dimer_calculations"
        if os.path.isdir(running_dir):
            shutil.rmtree(running_dir)
        os.mkdir(running_dir)
        
        #Calculate dimers up to min_atom_contact_limit.."
        rolling_avg = ["-1" for i in range(num_rolling_avg)]
        for count, distance in enumerate(distances):
            found_new_dimers = False
            if len(remaining_dimers) == 0:
                break
            del_names = []
            for dimer in remaining_dimers:
                # For all dimers left in the set check if any have nearest atom-atom distances within distance
                if get_nearest_atom_distance(unique_dimers[mapped_dimer_names[dimer]]) <= distance:
                    found_new_dimers = True
                    del_names.append(dimer)
                    # If they do add to important_mapped_dimers
                    important_mapped_dimers[dimer] = remaining_dimers[dimer]
                    if mapped_dimer_names[dimer] in important_unique_dimers:
                        # If this dimer corresponds to a known unique dimer, just add that energy
                        energy_sum += important_unique_dimers[mapped_dimer_names[dimer]].energies["FIT"]
                    else:
                        # If not, calculate the energy and add the unique dimer to important_unique_dimers
                        unique_dimers[mapped_dimer_names[dimer]].calculate_fit_energy(multipole_file, running_dir)
                        os.rmdir(running_dir + "/" + mapped_dimer_names[dimer])
                        new_dimer_energy = unique_dimers[mapped_dimer_names[dimer]].energies["FIT"]
                        if new_dimer_energy is not None:
                            energy_sum += new_dimer_energy
                            important_unique_dimers[unique_dimers[mapped_dimer_names[dimer]]] = unique_dimers[
                                mapped_dimer_names[dimer]]
                        else:
                            self.logger.critical("Energy calculation for dimer {} \
                                  failed. Cannot calculate dimer sum. Terminating..".format(mapped_dimer_names[dimer]))
                            return
            rolling_avg[count % num_rolling_avg] = energy_sum
            if distance >= min_atom_contact_limit and found_new_dimers:
                if "-1" not in rolling_avg:
                    self.energy_sums.append(energy_sum)
                    self.variances.append(get_variance(rolling_avg))
                    self.averages.append(sum(rolling_avg) / float(len(rolling_avg)))
                    self.logger.info("current energy {}| nearest atom distance {}| \
                    variance {}".format(energy_sum, distance, get_variance(rolling_avg)))
                    # If the energy converges above min_atom_contact_limit, self.unique_dimers 
                    #and self.mapped_dimer_names become the current set
                    if get_variance(rolling_avg) <= energy_threshold:
                        self.mapped_dimer_names = important_mapped_dimers
                        self.unique_dimers = [important_unique_dimers[i] for i in important_unique_dimers]
                        self.logger.info("{} converged at {} with {} dimers and {} unique dimers"\
                                         .format(self.name, distance, len(self.mapped_dimer_names), 
                                                 len(self.unique_dimers)))
                        self.converged_distance = distance
                        shutil.rmtree(running_dir)
                        return

            for del_name in del_names:
                del remaining_dimers[del_name]
        self.logger.warning("{} did not converge".format(self.name))
        self.converged_distance = max_atom_contact_limit
        self.mapped_dimer_names = important_mapped_dimers
        self.unique_dimers = [important_unique_dimers[i] for i in important_unique_dimers]
        shutil.rmtree(running_dir)

    
