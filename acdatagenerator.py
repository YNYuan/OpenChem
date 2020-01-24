import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.neighbors import NearestNeighbors

class ACDataGenarator():
    def __init__(self, index_path, NNeighbors):
        catelog = pd.read_csv(index_path, sep=' ', header=None)
        catelog.columns = ['name', '?', 'affinity']
        self.catelog = catelog
        self.NNeighbors = NNeighbors

    def rawDataReader(self, raw_folder):
        ligand_file = pd.DataFrame(columns=['name', 'xcoord', 'xtype'])
        rec_file = pd.DataFrame(columns=['name', 'xcoord', 'xtype'])
        for i, name in enumerate(self.catelog['name']):
            ligand_path = raw_folder + name + "/" + name +"_ligand.sdf"    
            ligand_mol = Chem.SDMolSupplier(ligand_path)[0]          
            if ligand_mol is not None:
                ligand_X = ligand_mol.GetConformers()[0].GetPositions()
                ligand_atom = np.empty([ligand_mol.GetNumAtoms(), 1], dtype=np.int16)
                for j, atom in enumerate(ligand_mol.GetAtoms()):
                    ligand_atom[j,0] = atom.GetAtomicNum()
            else:
                ligand_X = None
                ligand_atom = None
            ligand_file.loc[i] = [name, ligand_X, ligand_atom]

        for i, name in enumerate(self.catelog['name']):
            rec_path = raw_folder + name + "/" + name +"_rec.pdb"
            rec_mol = Chem.MolFromPDBFile(rec_path)        
            if rec_mol is not None:
                rec_X = rec_mol.GetConformers()[0].GetPositions()
                rec_atom = np.empty([rec_mol.GetNumAtoms(), 1], dtype=np.int16)
                for j, atom in enumerate(rec_mol.GetAtoms()):
                    rec_atom[j,0] = atom.GetAtomicNum()
            else:
                rec_X = None
                rec_atom = None      
            rec_file.loc[i] = [name, rec_X, rec_atom]

        ligand_file = ligand_file[ligand_file['xcoord'].notna()]
        rec_file = rec_file[rec_file['xcoord'].notna()]
        whole_file = pd.merge(ligand_file, rec_file, on='name', how='inner')
        whole_file.columns = ['name', 'ligand_coord', 'ligand_type', 'rec_coord', 'rec_type']
        y = pd.merge(whole_file, self.catelog, on='name', how='left')[['name', 'affinity']]
        y.to_csv('affinity_data.csv', index=None)
        return whole_file

    def atomNumCount(self, coord):
        max_atom = 0
        min_atom = 100000
        for arr in coord:
            if arr is not None:
                max_atom = max(max_atom, arr.shape[0])
                min_atom = min(min_atom, arr.shape[0])
        return max_atom, min_atom

    def coordData(self, coord_df, type_df, data_type):
        max_atom, _ = self.atomNumCount(coord_df)
        coord_matrix = np.empty([coord_df.shape[0], max_atom, 3])
        type_matrix = np.empty([coord_df.shape[0], max_atom, 1])
        for i, sub_mat in enumerate(coord_df):
            coord_matrix[i, :sub_mat.shape[0], :] = sub_mat
        for i, sub_mat in enumerate(type_df):
            type_matrix[i, :sub_mat.shape[0], :] = sub_mat
        if data_type == "ligand":
            np.save("ligand_coord", coord_matrix)
            np.save("ligand_type", type_matrix)
        elif data_type == "rec":
            np.save("rec_coord", coord_matrix)
            np.save("rec_type", type_matrix)
        return coord_matrix, type_matrix

    def coordDataReduce(self, rec_coord, rec_type, ligand_coord, cutoff):
        new_rec_atom = 0
        rec_coord_new = np.zeros([rec_coord.shape[0], rec_coord.shape[1], 3])
        rec_type_new = np.zeros([rec_coord.shape[0], rec_coord.shape[1], 1])
        for sample_no in range(ligand_coord.shape[0]):    
            ligand_one = ligand_coord[sample_no]
            ligand_one = ligand_one[~np.all(ligand_one==0, axis=1)]
            rec_one = rec_coord[sample_no]
            rec_one = rec_one[~np.all(rec_one==0, axis=1)]
            rec_one_type = rec_type[sample_no]
            rec_index = []
            for atom_lig in ligand_one:
                for i, atom_rec in enumerate(rec_one):
                    dis = np.linalg.norm(atom_lig-atom_rec)
                    if dis < cutoff:
                        rec_index.append(i)
            rec_coord_new[sample_no, :np.unique(rec_index).shape[0], :] = rec_one[np.unique(rec_index)]
            rec_type_new[sample_no, :np.unique(rec_index).shape[0], :] = rec_one_type[np.unique(rec_index)]
            new_rec_atom = max(new_rec_atom, np.unique(rec_index).shape[0])
        print(new_rec_atom)
        rec_coord_new = rec_coord_new[:, :new_rec_atom, :]
        rec_type_new = rec_type_new[:, :new_rec_atom, :]
        np.save('rec_coord_new', rec_coord_new)
        np.save('rec_type_new', rec_type_new)
        return rec_coord_new, rec_type_new

    def coordDataComplex(self, ligand_coord, ligand_type, rec_coord, rec_type):
        temp_atom_shape = ligand_coord.shape[1] + rec_coord.shape[1]
        atom_shape = 0
        complex_coord = np.zeros([ligand_coord.shape[0], temp_atom_shape, 3])
        complex_type = np.zeros([ligand_coord.shape[0], temp_atom_shape, 1])
        for i in range(ligand_coord.shape[0]):
            ligand_samp = ligand_coord[i]
            ligand_samp = ligand_samp[~np.all(ligand_samp==0, axis=1)]
            rec_samp = rec_coord[i]
            rec_samp = rec_samp[~np.all(rec_samp==0, axis=1)]
            complex_samp = np.vstack([ligand_samp, rec_samp])
            atom_shape = max(atom_shape, complex_samp.shape[0])
            complex_coord[i, :complex_samp.shape[0], :] = complex_samp      
            ligand_type_samp = ligand_type[i]
            ligand_type_samp = ligand_type_samp[~np.all(ligand_type_samp==0, axis=1)]
            rec_type_samp = rec_type[i]
            rec_type_samp = rec_type_samp[~np.all(rec_type_samp==0, axis=1)]
            complex_type_samp = np.vstack([ligand_type_samp, rec_type_samp])
            complex_type[i, :complex_type_samp.shape[0], :] = complex_type_samp
        complex_coord = complex_coord[:, :atom_shape, :]
        complex_type = complex_type[:, :atom_shape, :]
        np.save('complex_coord', complex_coord)
        np.save('complex_type', complex_type)
        return complex_coord, complex_type

    def neighborData(self, coord_matrix, type_matrix):
        neighborNum = self.NNeighbors + 1
        Nbr_distance_matrix = np.zeros([coord_matrix.shape[0], coord_matrix.shape[1], neighborNum])
        Nbr_type_matrix = np.zeros([coord_matrix.shape[0], coord_matrix.shape[1], neighborNum])
        for i, coord in enumerate(coord_matrix):
            coord = coord[~np.all(coord==0, axis=1)]
            nbrs = NearestNeighbors(n_neighbors=min(neighborNum, coord.shape[0]), algorithm='kd_tree').fit(coord)
            distances, indices = nbrs.kneighbors(coord)
            Nbr_distance_matrix[i,:distances.shape[0],:distances.shape[1]] = distances
            type_index = type_matrix[i][type_matrix[i]!=0]
            loc_func = lambda t: type_index[t]
            sub_type_matrix = np.array([loc_func(xi) for xi in indices])
            Nbr_type_matrix[i, :sub_type_matrix.shape[0], :distances.shape[1]] = sub_type_matrix                             
        Nbr_distance_matrix = Nbr_distance_matrix[:,:,1:]
        Nbr_type_matrix = Nbr_type_matrix[:,:,1:]
        return Nbr_distance_matrix, Nbr_type_matrix


myGenerator = ACDataGenarator("refinedSet.txt", 5)
whole_file = myGenerator.rawDataReader("refined-set/")
ligand_coord, ligand_type = myGenerator.coordData(whole_file['ligand_coord'], whole_file['ligand_type'], 'ligand')
rec_coord, rec_type = myGenerator.coordData(whole_file['rec_coord'], whole_file['rec_type'], 'rec')
rec_coord_new, rec_type_new = myGenerator.coordDataReduce(rec_coord, rec_type, ligand_coord, 12)
complex_coord, complex_type = myGenerator.coordDataComplex(ligand_coord, ligand_type, rec_coord_new, rec_type_new)

ligand_distance_matrix, ligand_type_matrix = myGenerator.neighborData(ligand_coord, ligand_type)
np.save('ligand_distance_matrix', ligand_distance_matrix)
np.save('ligand_type_matrix', ligand_type_matrix)

rec_distance_matrix, rec_type_matrix = myGenerator.neighborData(rec_coord_new, rec_type_new)
np.save('rec_distance_matrix', rec_distance_matrix)
np.save('rec_type_matrix', rec_type_matrix)

complex_distance_matrix, complex_type_matrix = myGenerator.neighborData(complex_coord, complex_type)
np.save('complex_distance_matrix', complex_distance_matrix)
np.save('complex_type_matrix', complex_type_matrix)