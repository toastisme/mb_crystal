import numpy as np
from cspy.alpha.crystal import CrystalStructure
from code.packing_opt.ani_symmetry_functions import get_symmetry_functions_array
import copy
import pickle
from code.packing_opt.dimer import Dimer
from code.packing_opt.mb_crystal import MbCrystal
from code.packing_opt.utils import get_dmacrys_args, res_string_to_crystal, crystal_to_res_string
from scipy import interpolate

class MbCrystalOptimiser(MbCrystal):
    
    
    def __init__(self, res_file, gp_model, symmetry_function_params, 
                 stepsizes, potential_file, multipole_file, axis_file=None):
        
        """
        Class to optimise res strings with machine learned, fragment-based models.

        Parameters
        ----------
        res_file : string
            path to .res file as a string
        gp_model : scikit_learn model
            trained machine learning model from ScikitLearn
        symmetry_function_params : dictionary 
            {'rc', 'ra', 'etaar', 'etaa', 'rsr_params', 
            'rsa_params', 'ts_params', 'zeta'}
        stepsizes : dictionary
            {'translation', 'rotation', 'unit_cell_lengths', 'lattice_angles'}
        potential_file : string
            Path to a .pots file required for DMACRYS calculation
        multipole_file : string
            Path to multipole file required for energy calculation.
        axis_file : string, optional
            Path to axis file required for energy calculation.
            
        """

        with open(res_file, "r") as g:
            res_string = g.read()
        name = res_file.split("/")[-1].split(".")[0]
        super(MbCrystalOptimiser, self).__init__(name, res_string)
        self.gp_model = gp_model
        self.symmetry_function_params = symmetry_function_params
        self.multipole_file = multipole_file
        self.axis_file = axis_file
        self.stepsizes = stepsizes
        self.potential_file = potential_file
        self.max_displacement = 0.0005
        self.lattice_type = None 
        self.get_cell_stress = None

        self.set_lattice_type(self.get_sg())
        
    def predict_dimer_energies(self, dimers):

        """
        Generates symmetry functions for the current set of dimers,
        and predicted energy differences.

        Parameters
        ----------
        dimers : list of Dimer instances

        Returns
        -------
        predicted_dimer_energies : list of floats

        """
        
        sym_fs = get_symmetry_functions_array(dimers, **self.symmetry_function_params) 
        
        return self.gp_model.predict(sym_fs)

    def get_sg(self):

        """
        Returns the space group extracted from self.name

        """
        return int(self.name[8:11]) 
       
    def set_lattice_type(self, sg):

        """
        Identifies the lattice type from the space group.
        The unit_cell_stress function that updates the unit cell
        during optimisation is then set based on this result.
     
        Parameters
        ----------
        sg : int
            spacegroup

        Updates
        -------
        lattice_type : string
        unit_cell_stress : function

        """

        assert isinstance(sg, int), "sg must be an int"
        assert sg > 0 and sg < 231, "sg must be between 1 and 230."


        sg_dict = {'1,2' :   'triclinic',

                   '3,15' :  'monoclinic',

                   '16,74' : 'orthorhombic',

                   '75,142' : 'tetragonal',

                   '143,167' : 'rhombohedral',

                   '168,194' : 'hexagonal',

                   '195,230' : 'cubic',
                  }

        stress_dict = {'triclinic' : self.get_triclinic_stress,
                       'monoclinic' : self.get_monoclinic_stress,
                       'orthorhombic' : self.get_orthorhombic_stress,
                       'tetragonal' : self.get_tetragonal_stress,
                       'hexagonal' : self.get_hexagonal_stress,
                       'cubic' : self.get_cubic_stress}

        for i in sg_dict:
            if sg >= int(i.split(",")[0]) and sg<= int(i.split(",")[1]):
                self.lattice_type = sg_dict[i]
                self.get_cell_stress = stress_dict[sg_dict[i]]
                return

        # This should be unreachable
        raise IndexError("Space group not found in sg_dict")

        
 
        
    def calculate_total_energy(self, res_string, return_data=False, buffer_size=0.5):

        """
        Calculates the energy of res_string using the fragment-based
        energy model.

        Parameters
        ----------
        res_string : string
            .res file as a string.
        return_data : list, optional
            if True, returns forces from DMACRYS.
        buffer_size : float, optional
            The distance (in angstroms) over which a spline is used to move from
            the fragment-based model to the DMACRYS model

        Returns
        -------
        total_energy : float
            lattice energy calculated using the fragment-based model.
       
        """


        def get_nearest_atom_distance(dimer):

            """
            Iterates over all interatomic distances and returns
            the smallest distance (in angstroms).

            Parameters
            ----------
            dimer : Dimer

            Returns
            -------
            smallest_distance : float
                The smallest interatomic distance in angstroms.
            """

            smallest_distance = -1
            for atom_i in dimer.molecules[0].atoms:
                for atom_j in dimer.molecules[1].atoms:
                    distance = atom_i.distance_to(atom_j)
                    if distance < smallest_distance or smallest_distance == -1:
                        smallest_distance = distance
            return smallest_distance
        
        self.res_string = res_string
        fit_energy, dma_data = self.calculate_fit_energy(return_data=True)
        self.populate_dimers(self.converged_distance)
        
        unique_dimer_energies = self.predict_dimer_energies(self.unique_dimers)
        unique_dimer_distances = [get_nearest_atom_distance(i) for i in self.unique_dimers]
        unique_dimer_dict = {i.name : [unique_dimer_energies[count], 
                             unique_dimer_distances[count]] for count, i  
                             in enumerate(self.unique_dimers)}

        all_dimer_data = [unique_dimer_dict[i] for i in self.mapped_dimer_names]
        all_dimer_data = sorted(all_dimer_data, key=lambda x:x[1])

        total_energy = fit_energy
        # Calculate the total energy up to the buffer zone
        buffer_start = all_dimer_data[-1][1] - buffer_size
        for count, dimer in enumerate(all_dimer_data):
            if dimer[1] > buffer_start:
                break
            total_energy += dimer[0]

        # Create spline
        spline_x = [buffer_start - i / 50. for i in range(50, 0, -10)] + [buffer_start] 
        spline_x += [all_dimer_data[-1][1]]
        spline_y = [1 for i in range(len(spline_x) - 1)] + [0]
        spline = interpolate.UnivariateSpline(spline_x, spline_y, s=0)
        
        # Add energies in the buffer zone
        for buffer_dimer in range(count, len(all_dimer_data)):
            total_energy += all_dimer_data[buffer_dimer][0] * spline(all_dimer_data[buffer_dimer][1])

        if return_data:
            return total_energy, dma_data
        return total_energy
    
    def calculate_fit_energy(self, res_string=None, return_data=False):

        """
        Calculates the FIT+DMA lattice energy of res_string.

        Parameters
        ----------
        res_string : string, optional
            .res file as a string. If None, self.res_string is used.
        return_data : boolean, optional
            If True, returns DMACRYS forces.
        
        Returns
        -------
        energy : float
            The calculated FIT+DMA lattice energy
        dma_data : list
            Forces from the calculation. Only returned if return_data is True.

        """
        
        from os import mkdir
        from shutil import rmtree
        from cspy.alpha.crystal import CrystalStructure
        
        tmpcrys = CrystalStructure()
        if res_string is None:
            tmpcrys.init_from_res_string(self.res_string)
        else:
            tmpcrys.init_from_res_string(res_string)
        tmpcrys.name = "tmpcrys"
        tmpcrys.filename = "tmpcrys.res"
        
        neighcrys_args, gaussian_args, gdma_args, dmacrys_args = \
            get_dmacrys_args(self.potential_file, self.multipole_file, self.axis_file)
        
        try:
            mkdir("tmp_dmacrys_calculations")
        except:
            rmtree("tmp_dmacrys_calculations")
            mkdir("tmp_dmacrys_calculations")
        running_dir = "tmp_dmacrys_calculations"
        
        tmpcrys.energy_minimize(run_dir=running_dir, neighcrys_args=neighcrys_args,
                                gaussian_args=gaussian_args, dmacrys_args=dmacrys_args)
        
        energy = tmpcrys.data_summary.initial_energy
        if return_data:
            dma_data = \
            MbCrystalOptimiser.get_dma_data("tmp_dmacrys_calculations/tmpcrys/tmpcrys.res.dmaout")
            rmtree("tmp_dmacrys_calculations")
            return energy, dma_data
        rmtree("tmp_dmacrys_calculations")
        return energy
    
    @staticmethod
    def get_dma_data(dmaout_file):
        """
        Extracts forces information from a dmaout_file, and the 
        local axis for asymmetric unit rotation.

        Parameters
        ----------
        dmaout_file : string
            Path to the dmaout_file.

        Returns
        -------
        dma_data : dictionary
             Unit cell strain, central forces on the asymmetric
             unit, central torque on the asymmetric unit, the centroid
             of the asymmetric unit, lattice vectors, and the asymmetric unit
             local axis.
        
       """ 
        
        with open(dmaout_file, "r") as g:
            lines = g.readlines()

        central_force_found = False
        central_torque_found = False
        dma_data = {} 
        for count, line in enumerate(lines):
            if line.startswith(" THE LATTICE VECTORS ARE AS FOLLOWS."):
                dma_data["lattice_vectors"] = []
                dma_data["lattice_vectors"].append([float(i) for i in lines[count+2].split()])
                dma_data["lattice_vectors"].append([float(i) for i in lines[count+3].split()])
                dma_data["lattice_vectors"].append([float(i) for i in lines[count+4].split()])
                dma_data["lattice_vectors"] = np.array(dma_data["lattice_vectors"])
            elif line.startswith("Local axis set for molecule:    1"):
                axis_x = [float(i) for i in  lines[count+1].split()[1:]]
                axis_y = [float(i) for i in  lines[count+2].split()[1:]]
                axis_z = [float(i) for i in  lines[count+3].split()[1:]]
                dma_data["local_axis"] = [axis_x, axis_y, axis_z]
            elif line.startswith("Molecule No.           Centred at               Total Mass"):
                dma_data["centroid"] =[float(i) for i in lines[count+4].split()[1:-1]]
            elif line.startswith("    Central force") and not central_force_found:
                dma_data["central_force"] = [i for i in line.split()[2:-1]]
                central_force_found = True
            elif line.startswith("   Central torque") and not central_torque_found:
                dma_data["central_torque"] = [i for i in lines[count+1].split()[4:-1]]
                central_torque_found = True
            elif line.startswith(" Units for strain derivatives"):
                strain = lines[count-2]
                strain =  [i for i in strain.split()[1:-1]]
                dma_data["strain"] = np.array([[strain[0], strain[5], strain[4]],
                                      [strain[5], strain[1], strain[3]], 
                                      [strain[4], strain[3], strain[2]]])
                break

        return dma_data
    
    def update_translation(self, crystal_obj):

        """
        Updates the asymmetric unit translational degrees of freedom
        using gradient descent, with the gradient calculated with
        the central difference.

        Parameters
        ----------
        crystal_obj : Crystal
            Crystal instance.

        Updates
        -------
        crystal_obj translational degrees of freedom.
 
        """
        
        from cspy.core.models import Vector3D
        from cspy.core.tasks.internal.molecule.transform \
        import translate
       
        if self.lattice_type == 'triclinic':
            return [0.0, 0.0, 0.0] 

        dR = self.stepsizes['translation']
        forces = []
        for count, i in enumerate([[1,0,0], [0,1,0], [0,0,1]]):

            trans_vec1 = Vector3D(dR/2. * i[0],
                                  dR/2. * i[1],
                                  dR/2. * i[2])

            translate(crystal_obj.asymmetric_unit[0], trans_vec1)
            e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
            
            trans_vec2 = Vector3D(-dR * i[0],
                                  -dR * i[1],
                                  -dR * i[2])
        
            translate(crystal_obj.asymmetric_unit[0], trans_vec2)
            e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
            dE = e1-e2
            forces.append(np.clip(dE, -self.max_displacement, self.max_displacement))
            translate(crystal_obj.asymmetric_unit[0], trans_vec1)
        
        update_vec = Vector3D(-forces[0], -forces[1], -forces[2])        
        translate(crystal_obj.asymmetric_unit[0], update_vec)
        return forces
        
    def update_rotation(self, crystal_obj, dma_data):

        """
        Updates the asymmetric unit rotational degrees of freedom
        using gradient descent, with the gradient calculated with
        the central difference.

        Parameters
        ----------
        crystal_obj : Crystal
            Crystal instance.

        Updates
        -------
        crystal_obj rotational degrees of freedom.

        """

        
        from cspy.core.tasks.internal.molecule.transform \
        import rotate, rotate_by_quaternion
        from cspy.core.models import Vector3D, Matrix3D
        from cspy.core.models.quaternion import Quaternion

        torques = []
        
        center = Vector3D(dma_data['centroid'][0], dma_data['centroid'][1], dma_data['centroid'][2])
        d_theta = self.stepsizes['rotation']
        cos_a_2 = np.cos(d_theta / 4.)
        cos_a_2_r = np.cos(-d_theta / 2.)
        sin_a_2 = np.sin(d_theta / 4.)
        sin_a_2_r = np.sin(-d_theta / 2.)
        for count, i in enumerate([[1,0,0], [0,1,0], [0,0,1]]):

            rotation_quaternion1 = Quaternion(cos_a_2, 
                                               sin_a_2* i[0],
                                               sin_a_2*i[1], 
                                               sin_a_2*i[2])
            rotate_by_quaternion(crystal_obj.asymmetric_unit[0],
                             rotation_quaternion1,
                             center)
            e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
            
            rotation_quaternion2 = Quaternion(cos_a_2_r, 
                                               sin_a_2_r*i[0], 
                                               sin_a_2_r*i[1], 
                                               sin_a_2_r*i[2])
            rotate_by_quaternion(crystal_obj.asymmetric_unit[0],
                             rotation_quaternion2,
                             center)

            e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
            
            rotate_by_quaternion(crystal_obj.asymmetric_unit[0],
                 rotation_quaternion1,
                 center)


            dE  = e1 - e2
            torques.append(np.clip(dE, -self.max_displacement, self.max_displacement))
            
        transformed_torques = MbCrystalOptimiser.coordinate_transform(torques, dma_data["local_axis"])
        torques = MbCrystalOptimiser.coordinate_transform([i[0] for i in transformed_torques], dma_data["local_axis"])
        magnitude = np.sqrt(torques[0]**2 + torques[1]**2 + torques[2]**2)
        torques_norm = [i/magnitude for i in torques]
        #torques = [i/d_theta for i in torques]
        
        rotation_quaternion = Quaternion(cos_a_2_r, 
                                           sin_a_2_r*torques_norm[0],
                                           sin_a_2_r*torques_norm[1], 
                                           sin_a_2_r*torques_norm[2])
        rotate_by_quaternion(crystal_obj.asymmetric_unit[0],
                         rotation_quaternion,
                         center)
        return torques

    def get_triclinic_stress(self, crystal_obj, dR, dT):

        """
        Obtains the forces on the unit cell treating it as triclinic.

        Parameters
        ----------
        crystal_obj : Crystal

        Returns
        -------
        stress : list of floats

        """
         
        stress = []

        crystal_obj.lattice.a += dR/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.a -= dR
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.a += dR/2.

        crystal_obj.lattice.b += dR/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.b -= dR
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.b += dR/2.

        crystal_obj.lattice.c += dR/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.c -= dR
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.c += dR/2.

        crystal_obj.lattice.alpha += dT/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.alpha -= dT
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.alpha += dT/2.

        crystal_obj.lattice.beta += dT/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.beta -= dT
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.beta += dT/2.

        crystal_obj.lattice.gamma += dT/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.gamma -= dT
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.gamma += dT/2.

        return stress

    def get_monoclinic_stress(self, crystal_obj, dR, dT):

        """
        Obtains the forces on the unit cell treating it as monoclinic.

        Parameters
        ----------
        crystal_obj : Crystal

        Returns
        -------
        stress : list of floats

        """

        stress = []

        crystal_obj.lattice.a += dR/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.a -= dR
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.a += dR/2.

        crystal_obj.lattice.b += dR/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.b -= dR
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.b += dR/2.

        crystal_obj.lattice.c += dR/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.c -= dR
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.c += dR/2.

        stress.append(0.0)

        crystal_obj.lattice.beta += dT/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.beta -= dT
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.beta += dT/2.

        stress.append(0.0)

        return stress

    def get_orthorhombic_stress(self, crystal_obj, dR, dT):

        """
        Obtains the forces on the unit cell treating it as orthorhombic.

        Parameters
        ----------
        crystal_obj : Crystal

        Returns
        -------
        stress : list of floats

        """

        stress = []

        crystal_obj.lattice.a += dR/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.a -= dR
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.a += dR/2.

        crystal_obj.lattice.b += dR/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.b -= dR
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.b += dR/2.

        crystal_obj.lattice.c += dR/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.c -= dR
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.c += dR/2.

        stress.append(0.0)
        stress.append(0.0)
        stress.append(0.0)

        return stress

    def get_tetragonal_stress(self, crystal_obj, dR, dT):

        """
        Obtains the forces on the unit cell treating it as tetragonal.

        Parameters
        ----------
        crystal_obj : Crystal

        Returns
        -------
        stress : list of floats

        """

        stress = []

        crystal_obj.lattice.a += dR/2.
        crystal_obj.lattice.b += dR/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.a -= dR
        crystal_obj.lattice.b -= dR
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.a += dR/2.
        crystal_obj.lattice.b += dR/2.

        crystal_obj.lattice.c += dR/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.c -= dR
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.c += dR/2.

        stress.append(0.0)
        stress.append(0.0)
        stress.append(0.0)

        return stress

    def get_rhombohedral_stress(self, crystal_obj, dR, dT):

        """
        Obtains the forces on the unit cell treating it as rhombohedral.

        Parameters
        ----------
        crystal_obj : Crystal

        Returns
        -------
        stress : list of floats

        """

        stress = []

        crystal_obj.lattice.a += dR/2.
        crystal_obj.lattice.b += dR/2.
        crystal_obj.lattice.c += dR/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.a -= dR
        crystal_obj.lattice.b -= dR
        crystal_obj.lattice.c -= dR
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.a += dR/2.
        crystal_obj.lattice.b += dR/2.
        crystal_obj.lattice.c += dR/2.

        crystal_obj.lattice.alpha += dT/2.
        crystal_obj.lattice.beta += dT/2.
        crystal_obj.lattice.gamma += dT/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.alpha -= dT
        crystal_obj.lattice.beta -= dT
        crystal_obj.lattice.gamma -= dT
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.alpha += dT/2.
        crystal_obj.lattice.beta += dT/2.
        crystal_obj.lattice.gamma += dT/2.

        return stress

    def get_hexagonal_stress(self, crystal_obj, dR, dT):

        """
        Obtains the forces on the unit cell treating it as hexagonal.

        Parameters
        ----------
        crystal_obj : Crystal

        Returns
        -------
        stress : list of floats

        """

        stress = []

        crystal_obj.lattice.a += dR/2.
        crystal_obj.lattice.b += dR/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.a -= dR
        crystal_obj.lattice.b -= dR
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.a += dR/2.
        crystal_obj.lattice.b += dR/2.

        crystal_obj.lattice.c += dR/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.c -= dR
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.c += dR/2.

        stress.append(0.0)
        stress.append(0.0)
        stress.append(0.0)

        return stress

    def get_cubic_stress(self, crystal_obj, dR, dT):

        """
        Obtains the forces on the unit cell treating it as rhombohedral.

        Parameters
        ----------
        crystal_obj : Crystal

        Returns
        -------
        stress : list of floats

        """

        stress = []

        crystal_obj.lattice.a += dR/2.
        crystal_obj.lattice.b += dR/2.
        crystal_obj.lattice.c += dR/2.
        e1 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        crystal_obj.lattice.a -= dR
        crystal_obj.lattice.b -= dR
        crystal_obj.lattice.c -= dR
        e2 = self.calculate_total_energy(crystal_to_res_string(crystal_obj))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        stress.append(np.clip(e1-e2, -self.max_displacement, self.max_displacement))
        crystal_obj.lattice.a += dR/2.
        crystal_obj.lattice.b += dR/2.
        crystal_obj.lattice.c += dR/2.

        stress.append(0.0)
        stress.append(0.0)
        stress.append(0.0)

        return stress

        
    def update_unit_cell(self, crystal_obj):

        """
        Updates the unit cell of crystal_obj using gradient descent
        with the gradient calculated with the central difference.

        Parameters
        ----------
        crystal_obj : Crystal
            Crystal instance.

        Updates
        -------
        crystal_obj lattice vectors.

        """
        

        dR = self.stepsizes['unit_cell_lengths']
        dT = self.stepsizes['unit_cell_angles']

        stress = self.get_cell_stress(crystal_obj, dR, dT)
        
        lvs = crystal_obj.lattice.lattice_vectors
        vecs = np.array(([lvs[0][0], lvs[0][1], lvs[0][2]],
                        [lvs[1][0], lvs[1][1], lvs[1][2]],
                        [lvs[2][0], lvs[2][1], lvs[2][2]]))
        
        stress_arr = np.array(([-stress[0], -stress[3], -stress[4]], 
                               [-stress[5], -stress[1], -stress[5]], 
                               [-stress[4], -stress[3], -stress[2]]))

        
        I = np.identity(3)
        vecs = np.matmul(I + stress_arr,vecs)
        
        crystal_obj.lattice.a = np.linalg.norm(vecs[0])
        crystal_obj.lattice.b = np.linalg.norm(vecs[1])
        crystal_obj.lattice.c = np.linalg.norm(vecs[2])
        
        crystal_obj.lattice.alpha = np.arccos(np.dot(vecs[1],vecs[2])\
                                /(np.linalg.norm(vecs[1])*np.linalg.norm(vecs[2])))\
                                * (180/np.pi)
        crystal_obj.lattice.beta = np.arccos(np.dot(vecs[2],vecs[0])\
                                /(np.linalg.norm(vecs[2])*np.linalg.norm(vecs[0])))\
                                * (180/np.pi)
        crystal_obj.lattice.gamma = np.arccos(np.dot(vecs[0],vecs[1])\
                                /(np.linalg.norm(vecs[0])*np.linalg.norm(vecs[1])))\
                                * (180/np.pi)    

        
        return stress
        
        
    def update_res_string(self, dma_data):

        """
        Updates the degrees of freedom of self.res_string
        using gradient descent, with the gradient calculated with
        the central difference.

        Parameters
        ----------
        crystal_obj : Crystal
            Crystal instance.

        Updates
        -------
        self.res_string degrees of freedom.

        """
     
        crystal_obj = res_string_to_crystal(self.res_string)
        forces = self.update_translation(crystal_obj)
        torques = self.update_rotation(crystal_obj, dma_data)
        stress = self.update_unit_cell(crystal_obj)
        self.res_string = crystal_to_res_string(crystal_obj)

        return [forces, torques, stress]
        
    @staticmethod
    def coordinate_transform(vec, axis_set):

        """
        Transforms vec to the axis frame axis_set.

        Parameters
        ----------
        vec : list of floats of len 3
        axis_set : 3 x 3 list of floats

        Returns
        -------
        vec transformed.

        """

        axis_set = np.array(axis_set)
        vec = np.array(([vec[0]], [vec[1]], [vec[2]]))

        return np.matmul(axis_set, vec)

    
