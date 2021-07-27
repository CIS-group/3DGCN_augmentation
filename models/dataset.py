from keras.utils import to_categorical, Sequence
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem
import numpy as np
import pandas as pd
import random

def one_hot(x, allowable_set):
    # If x is not in allowed set, use last index
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))

def degree_rotation_matrix(axis, degree):
    theta = degree / 180 * np.pi
    if axis == "x":
        r = np.array([1, 0, 0, 0, np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta)]).reshape([3, 3])
    elif axis == "y":
        r = np.array([np.cos(theta), 0, np.sin(theta), 0, 1, 0, -np.sin(theta), 0, np.cos(theta)]).reshape([3, 3])
    elif axis == "z":
        r = np.array([np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta), 0, 0, 0, 1]).reshape([3, 3])
    else:
        raise ValueError("Unsupported axis for rotation: {}".format(axis))

    return r

class Dataset(object):
    def __init__(self, dataset, fold, iter, split, axis, degrees, exclude=[], batch=128):
        self.dataset = dataset
        self.path = "./data/{}.sdf".format(dataset)  # "../../data/{}.sdf".format(
        # dataset)
        self.task = None  # "binary"
        self.target_name = None  # "active"
        self.max_atoms = 0

        self.batch = batch
        self.outputs = 1
        self.axis = axis
        self.degrees = degrees
        self.exclude = exclude

        self.mols = []
        self.coords = []
        self.target = []
        self.x, self.c, self.y = {}, {}, {}

        self.use_atom_symbol = True
        self.use_degree = True
        self.use_hybridization = True
        self.use_implicit_valence = True
        self.use_partial_charge = False
        self.use_formal_charge = True
        self.use_ring_size = True
        self.use_hydrogen_bonding = True
        self.use_acid_base = True
        self.use_aromaticity = True
        self.use_chirality = True
        self.use_num_hydrogen = True

        # Load data
        if split == "":
            self.load_dataset(fold, iter)
        elif split == "stratified_sampling":
            self.load_dataset_stratified_sampling(fold, iter)
        elif split == "stratified_augmented_sampling":
            self.load_dataset_augmented_stratified_sampling(fold, iter)
        elif split == "stratified_random_degree_augmented_sampling":
            self.load_dataset_random_degree_sampling(fold, iter)
        elif split == "stratified_random_degree_axis_augmented_sampling":
            self.load_dataset_random_degree_axis_sampling(fold, iter)
        else:
            print('Select proper dataset splitting type.')

        # Calculate number of features
        mp = MPGenerator([], [], [], 1,
                         use_atom_symbol=self.use_atom_symbol,
                         use_degree=self.use_degree,
                         use_hybridization=self.use_hybridization,
                         use_implicit_valence=self.use_implicit_valence,
                         use_partial_charge=self.use_partial_charge,
                         use_formal_charge=self.use_formal_charge,
                         use_ring_size=self.use_ring_size,
                         use_hydrogen_bonding=self.use_hydrogen_bonding,
                         use_acid_base=self.use_acid_base,
                         use_aromaticity=self.use_aromaticity,
                         use_chirality=self.use_chirality,
                         use_num_hydrogen=self.use_num_hydrogen)
        self.num_features = mp.get_num_features()

        # Normalize
        if self.task == "regression":
            self.mean = np.mean(self.y["train"])
            self.std = np.std(self.y["train"])

            self.y["train"] = (self.y["train"] - self.mean) / self.std
            self.y["valid"] = self.y["train"]
            #self.y["valid"] = (self.y["valid"] - self.mean) / self.std
            self.y["test"] = (self.y["test"] - self.mean) / self.std
        else:
            self.mean = 0
            self.std = 1

    def load_dataset(self, fold, iter):
        # Dataset parameters
        if self.dataset == "bace_reg" or self.dataset == "delaney" or self.dataset == "freesolv":
            self.task = "regression"
            self.target_name = "target"
            self.loss = "mse"

        elif self.dataset == "bace_cla" or self.dataset == "hiv":
            self.task = "binary"
            self.target_name = "active"
            self.loss = "binary_crossentropy"

        elif self.dataset == "tox21":
            self.target_name = "NR-ER"

        # elif self.dataset == "tox21":  # Multitask tox21
        #     self.target_name = ["NR-Aromatase", "NR-AR", "NR-AR-LBD", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "NR-AhR",
        #                    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]

        else:
            pass

        # Load file
        x, c, y = [], [], []
        mols = Chem.SDMolSupplier(self.path)
        if len(self.exclude) == 0:
            for mol in mols:
                if mol is not None:
                    if mol.GetNumAtoms() > 200:
                        continue
                    # Multitask
                    if type(self.target_name) is list:
                        y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in self.target_name])
                        self.outputs = len(self.target_name)

                    # Singletask
                    elif self.target_name in mol.GetPropNames():
                        _y = float(mol.GetProp(self.target_name))
                        if _y == -1:
                            continue
                        else:
                            y.append(_y)

                    else:
                        continue

                    x.append(mol)
                    c.append(mol.GetConformer().GetPositions())
            assert len(x) == len(y)

        else:
            for mol in mols:
                if mol is not None:
                    if mol.GetProp("_Name") in self.exclude:
                        print("{} is excluded".format(mol.GetProp("_Name")))
                    else:
                        if mol.GetNumAtoms() > 200:
                            continue
                        # Multitask
                        if type(self.target_name) is list:
                            y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in self.target_name])
                            self.outputs = len(self.target_name)

                        # Singletask
                        elif self.target_name in mol.GetPropNames():
                            _y = float(mol.GetProp(self.target_name))
                            if _y == -1:
                                continue
                            else:
                                y.append(_y)

                        else:
                            continue

                        x.append(mol)
                        c.append(mol.GetConformer().GetPositions())
                assert len(x) == len(y)

        # Filter and update maximum number of atoms
        new_x, new_c, new_y = [], [], []
        if self.max_atoms > 0:
            for mol, coo, tar in zip(x, c, y):
                if mol.GetNumAtoms() <= self.max_atoms:
                    new_x.append(mol)
                    new_c.append(coo)
                    new_y.append(tar)

            x = new_x
            c = new_c
            y = new_y

        else:
            for mol, tar in zip(x, y):
                self.max_atoms = max(self.max_atoms, mol.GetNumAtoms())

        if self.task != "regression":
            self.mols, self.coords, self.target = np.array(x), np.array(c), np.array(y, dtype=int)
        else:
            self.mols, self.coords, self.target = np.array(x), np.array(c), np.array(y)

        # Shuffle data
        np.random.seed(100)  # 37
        idx = np.random.permutation(len(self.mols))
        self.mols, self.coords, self.target = self.mols[idx], self.coords[idx], self.target[idx]

        # Split data
        len_test = int(len(self.mols) / fold)

        self.x = {"train": np.concatenate((self.mols[:len_test*iter], self.mols[len_test*(iter + 1):])),
                  "test": self.mols[len_test*iter:len_test*(iter+1)]}
        self.c = {"train": np.concatenate((self.coords[:len_test*iter], self.coords[len_test*(iter + 1):])),
                  "test": self.coords[len_test*iter:len_test*(iter+1)]}
        self.y = {"train": np.concatenate((self.target[:len_test*iter], self.target[len_test*(iter + 1):])),
                  "test": self.target[len_test*iter:len_test*(iter+1)]}

    def load_dataset_stratified_sampling(self, fold, iter):
        # Dataset parameters
        if self.dataset == "bace_reg" or self.dataset == "delaney" or self.dataset == "freesolv":
            self.task = "regression"
            self.target_name = "target"
            self.loss = "mse"

        elif self.dataset == "bace_cla" or self.dataset == "hiv":
            self.task = "binary"
            self.target_name = "active"
            self.loss = "binary_crossentropy"

        elif self.dataset == "tox21":
            self.target_name = "NR-ER"

        # elif self.dataset == "tox21":  # Multitask tox21
        #     self.target_name = ["NR-Aromatase", "NR-AR", "NR-AR-LBD", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "NR-AhR",
        #                    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]

        else:
            pass

        # Load file
        x, c, y = [], [], []
        mols = Chem.SDMolSupplier(self.path)
        if len(self.exclude) == 0:
            for mol in mols:
                if mol is not None:
                    if mol.GetNumAtoms() > 200:
                        continue
                    # Multitask
                    if type(self.target_name) is list:
                        y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in self.target_name])
                        self.outputs = len(self.target_name)

                    # Singletask
                    elif self.target_name in mol.GetPropNames():
                        _y = float(mol.GetProp(self.target_name))
                        if _y == -1:
                            continue
                        else:
                            y.append(_y)

                    else:
                        continue

                    x.append(mol)
                    c.append(mol.GetConformer().GetPositions())
            assert len(x) == len(y)

        else:
            for mol in mols:
                if mol is not None:
                    if mol.GetProp("_Name") in self.exclude:
                        print("{} is excluded".format(mol.GetProp("_Name")))
                    else:
                        if mol.GetNumAtoms() > 200:
                            continue
                        # Multitask
                        if type(self.target_name) is list:
                            y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in self.target_name])
                            self.outputs = len(self.target_name)

                        # Singletask
                        elif self.target_name in mol.GetPropNames():
                            _y = float(mol.GetProp(self.target_name))
                            if _y == -1:
                                continue
                            else:
                                y.append(_y)

                        else:
                            continue

                        x.append(mol)
                        c.append(mol.GetConformer().GetPositions())
                assert len(x) == len(y)
        # Filter and update maximum number of atoms
        new_x, new_c, new_y = [], [], []
        if self.max_atoms > 0:
            for mol, coo, tar in zip(x, c, y):
                if mol.GetNumAtoms() <= self.max_atoms:
                    new_x.append(mol)
                    new_c.append(coo)
                    new_y.append(tar)

            x = new_x
            c = new_c
            y = new_y

        else:
            for mol, tar in zip(x, y):
                self.max_atoms = max(self.max_atoms, mol.GetNumAtoms())

        if self.task != "regression":
            self.mols, self.coords, self.target = np.array(x), np.array(c), np.array(y, dtype=int)
        else:
            self.mols, self.coords, self.target = np.array(x), np.array(c), np.array(y)

        # Shuffle data
        idx_inactive = np.squeeze(np.argwhere(self.target == 0))  # 825
        idx_active = np.squeeze(np.argwhere(self.target == 1))  # 653

        np.random.seed(100)  # 37
        np.random.shuffle(idx_inactive)
        np.random.seed(100)  # 37
        np.random.shuffle(idx_active)

        # Split data
        len_inactive = int(len(idx_inactive) / fold)
        len_active = int(len(idx_active) / fold)

        test_idx = np.append(idx_inactive[len_inactive*iter:len_inactive*(iter+1)], idx_active[len_active*iter:len_active*(iter+1)])
        train_idx = np.append(np.append(idx_inactive[:len_inactive*iter], idx_inactive[len_inactive*(iter+1):]),
                              np.append(idx_active[:len_active*iter], idx_active[len_active*(iter+1):]))

        test_idx = np.random.permutation(test_idx)
        train_idx = np.random.permutation(train_idx)

        self.x = {"train": self.mols[train_idx], "test": self.mols[test_idx]}
        self.c = {"train": self.coords[train_idx], "test": self.coords[test_idx]}
        self.y = {"train": self.target[train_idx], "test": self.target[test_idx]}

        print('Finish stratified splitting')

    def load_dataset_augmented_stratified_sampling(self, fold, iter):
        # Dataset parameters
        if self.dataset == "bace_reg" or self.dataset == "delaney" or self.dataset == "freesolv":
            self.task = "regression"
            self.target_name = "target"
            self.loss = "mse"

        elif self.dataset == "bace_cla" or self.dataset == "hiv":
            self.task = "binary"
            self.target_name = "active"
            self.loss = "binary_crossentropy"

        elif self.dataset == "tox21":
            self.target_name = "NR-ER"

        else:
            pass

        # Load file
        x, c, y = [], [], []
        mols = Chem.SDMolSupplier(self.path)
        if len(self.exclude) == 0:
            for mol in mols:
                if mol is not None:
                    if mol.GetNumAtoms() > 200:
                        continue
                    # Multitask
                    if type(self.target_name) is list:
                        y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in self.target_name])
                        self.outputs = len(self.target_name)

                    # Singletask
                    elif self.target_name in mol.GetPropNames():
                        _y = float(mol.GetProp(self.target_name))
                        if _y == -1:
                            continue
                        else:
                            y.append(_y)

                    else:
                        continue

                    x.append(mol)
                    c.append(mol.GetConformer().GetPositions())
            assert len(x) == len(y)

        else:
            for mol in mols:
                if mol is not None:
                    if mol.GetProp("_Name") in self.exclude:
                        print("{} is excluded".format(mol.GetProp("_Name")))
                    else:
                        if mol.GetNumAtoms() > 200:
                            continue
                        # Multitask
                        if type(self.target_name) is list:
                            y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in self.target_name])
                            self.outputs = len(self.target_name)

                        # Singletask
                        elif self.target_name in mol.GetPropNames():
                            _y = float(mol.GetProp(self.target_name))
                            if _y == -1:
                                continue
                            else:
                                y.append(_y)

                        else:
                            continue

                        x.append(mol)
                        c.append(mol.GetConformer().GetPositions())
                assert len(x) == len(y)

        # Filter and update maximum number of atoms
        new_x, new_c, new_y = [], [], []
        if self.max_atoms > 0:
            for mol, coo, tar in zip(x, c, y):
                if mol.GetNumAtoms() <= self.max_atoms:
                    new_x.append(mol)
                    new_c.append(coo)
                    new_y.append(tar)

            x = new_x
            c = new_c
            y = new_y

        else:
            for mol, tar in zip(x, y):
                self.max_atoms = max(self.max_atoms, mol.GetNumAtoms())

        if self.task != "regression":
            self.mols, self.coords, self.target = np.array(x), np.array(c), np.array(y, dtype=int)
        else:
            self.mols, self.coords, self.target = np.array(x), np.array(c), np.array(y)

        # Shuffle data
        idx_inactive = np.squeeze(np.argwhere(self.target == 0))  # 825
        idx_active = np.squeeze(np.argwhere(self.target == 1))  # 653
        idx_augmented_inactive = np.squeeze(np.argwhere(self.target == 1))  # 653

        np.random.seed(100)  # 37
        np.random.shuffle(idx_inactive)
        np.random.seed(100)  # 37
        np.random.shuffle(idx_active)
        np.random.seed(100) # 37
        np.random.shuffle(idx_augmented_inactive)

        # Split data
        len_inactive = int(len(idx_inactive) / fold)
        len_active = int(len(idx_active) / fold)
        len_augmented_inactive = int(len(idx_augmented_inactive) / fold)

        test_idx = np.append(idx_inactive[len_inactive*iter:len_inactive*(iter+1)], idx_active[len_active*iter:len_active*(iter+1)])
        train_idx = np.append(np.append(idx_inactive[:len_inactive*iter], idx_inactive[len_inactive*(iter+1):]),
                              np.append(idx_active[:len_active*iter], idx_active[len_active*(iter+1):]))
        # not using augmented data for test.
        augmented_idx = np.append(idx_augmented_inactive[:len_augmented_inactive * iter],
                                  idx_augmented_inactive[len_augmented_inactive * (iter + 1):])

        augmented_idx = np.random.permutation(augmented_idx)
        test_idx = np.random.permutation(test_idx)
        train_idx = np.random.permutation(train_idx)

        rotated_coords = []
        for degree in self.degrees:
            # for axis in ["x", "y", "z"]: # delete to select specific augmentation axis by self.axis  
            for coords in self.coords[augmented_idx]:                       
                rotation_matrix = degree_rotation_matrix(self.axis, degree) # self.axis
                new_coords = np.matmul(coords, rotation_matrix)
                rotated_coords.append(new_coords)
        rotated_coords = np.asarray(rotated_coords, dtype=object)

        x_train = np.append(self.mols[train_idx], self.mols[augmented_idx])
        for i in range(len(self.degrees) - 1):
            x_train = np.append(x_train, self.mols[augmented_idx])
        c_train = np.append(self.coords[train_idx], rotated_coords)
        y_train = np.append(self.target[train_idx], np.zeros(len(self.degrees) * len(augmented_idx), dtype=int))

        self.x = {"train": x_train, "test": self.mols[test_idx]}
        self.c = {"train": c_train, "test": self.coords[test_idx]}
        self.y = {"train": y_train, "test": self.target[test_idx]}

        print('Finish stratified augmented splitting')

    def load_dataset_random_degree_sampling(self, fold, iter):
        # Dataset parameters
        if self.dataset == "bace_reg" or self.dataset == "delaney" or self.dataset == "freesolv":
            self.task = "regression"
            self.target_name = "target"
            self.loss = "mse"

        elif self.dataset == "bace_cla" or self.dataset == "hiv":
            self.task = "binary"
            self.target_name = "active"
            self.loss = "binary_crossentropy"

        elif self.dataset == "tox21":
            self.target_name = "NR-ER"

        else:
            pass

        # Load file
        x, c, y = [], [], []
        mols = Chem.SDMolSupplier(self.path)
        if len(self.exclude) == 0:
            for mol in mols:
                if mol is not None:
                    if mol.GetNumAtoms() > 200:
                        continue
                    # Multitask
                    if type(self.target_name) is list:
                        y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in self.target_name])
                        self.outputs = len(self.target_name)

                    # Singletask
                    elif self.target_name in mol.GetPropNames():
                        _y = float(mol.GetProp(self.target_name))
                        if _y == -1:
                            continue
                        else:
                            y.append(_y)

                    else:
                        continue

                    x.append(mol)
                    c.append(mol.GetConformer().GetPositions())
            assert len(x) == len(y)

        else:
            for mol in mols:
                if mol is not None:
                    if mol.GetProp("_Name") in self.exclude:
                        print("{} is excluded".format(mol.GetProp("_Name")))
                    else:
                        if mol.GetNumAtoms() > 200:
                            continue
                        # Multitask
                        if type(self.target_name) is list:
                            y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in self.target_name])
                            self.outputs = len(self.target_name)

                        # Singletask
                        elif self.target_name in mol.GetPropNames():
                            _y = float(mol.GetProp(self.target_name))
                            if _y == -1:
                                continue
                            else:
                                y.append(_y)

                        else:
                            continue

                        x.append(mol)
                        c.append(mol.GetConformer().GetPositions())
                assert len(x) == len(y)

        # Filter and update maximum number of atoms
        new_x, new_c, new_y = [], [], []
        if self.max_atoms > 0:
            for mol, coo, tar in zip(x, c, y):
                if mol.GetNumAtoms() <= self.max_atoms:
                    new_x.append(mol)
                    new_c.append(coo)
                    new_y.append(tar)

            x = new_x
            c = new_c
            y = new_y

        else:
            for mol, tar in zip(x, y):
                self.max_atoms = max(self.max_atoms, mol.GetNumAtoms())

        if self.task != "regression":
            self.mols, self.coords, self.target = np.array(x), np.array(c), np.array(y, dtype=int)
        else:
            self.mols, self.coords, self.target = np.array(x), np.array(c), np.array(y)

        # Shuffle data
        idx_inactive = np.squeeze(np.argwhere(self.target == 0))  # 825
        idx_active = np.squeeze(np.argwhere(self.target == 1))  # 653
        idx_augmented_inactive = np.squeeze(np.argwhere(self.target == 1))  # 653

        np.random.seed(100)  # 37
        np.random.shuffle(idx_inactive)
        np.random.seed(100)  # 37
        np.random.shuffle(idx_active)
        np.random.seed(100) # 37
        np.random.shuffle(idx_augmented_inactive)

        # Split data
        len_inactive = int(len(idx_inactive) / fold)
        len_active = int(len(idx_active) / fold)
        len_augmented_inactive = int(len(idx_augmented_inactive) / fold)

        test_idx = np.append(idx_inactive[len_inactive*iter:len_inactive*(iter+1)], idx_active[len_active*iter:len_active*(iter+1)])
        train_idx = np.append(np.append(idx_inactive[:len_inactive*iter], idx_inactive[len_inactive*(iter+1):]),
                              np.append(idx_active[:len_active*iter], idx_active[len_active*(iter+1):]))
        # not using augmented data for test.
        augmented_idx = np.append(idx_augmented_inactive[:len_augmented_inactive * iter],
                                  idx_augmented_inactive[len_augmented_inactive * (iter + 1):])

        augmented_idx = np.random.permutation(augmented_idx)
        test_idx = np.random.permutation(test_idx)
        train_idx = np.random.permutation(train_idx)

        rotated_coords = []
        for i in range(3):  
            for coords in self.coords[augmented_idx]:
                degree = np.random.uniform(90, 270)
                rotation_matrix = degree_rotation_matrix(self.axis, degree)
                new_coords = np.matmul(coords, rotation_matrix)
                rotated_coords.append(new_coords)
        rotated_coords = np.asarray(rotated_coords, dtype=object)

        x_train = np.append(self.mols[train_idx], self.mols[augmented_idx])
        for i in range(2):
            x_train = np.append(x_train, self.mols[augmented_idx])
        c_train = np.append(self.coords[train_idx], rotated_coords)
        y_train = np.append(self.target[train_idx], np.zeros(3 * len(augmented_idx), dtype=int))

        self.x = {"train": x_train, "test": self.mols[test_idx]}
        self.c = {"train": c_train, "test": self.coords[test_idx]}
        self.y = {"train": y_train, "test": self.target[test_idx]}

        print('Finish stratified random degree augmented splitting')

    def load_dataset_random_degree_axis_sampling(self, fold, iter):
        # Dataset parameters
        if self.dataset == "bace_reg" or self.dataset == "delaney" or self.dataset == "freesolv":
            self.task = "regression"
            self.target_name = "target"
            self.loss = "mse"

        elif self.dataset == "bace_cla" or self.dataset == "hiv":
            self.task = "binary"
            self.target_name = "active"
            self.loss = "binary_crossentropy"

        elif self.dataset == "tox21":
            self.target_name = "NR-ER"

        else:
            pass

        # Load file
        x, c, y = [], [], []
        mols = Chem.SDMolSupplier(self.path)
        if len(self.exclude) == 0:
            for mol in mols:
                if mol is not None:
                    if mol.GetNumAtoms() > 200:
                        continue
                    # Multitask
                    if type(self.target_name) is list:
                        y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in self.target_name])
                        self.outputs = len(self.target_name)

                    # Singletask
                    elif self.target_name in mol.GetPropNames():
                        _y = float(mol.GetProp(self.target_name))
                        if _y == -1:
                            continue
                        else:
                            y.append(_y)

                    else:
                        continue

                    x.append(mol)
                    c.append(mol.GetConformer().GetPositions())
            assert len(x) == len(y)

        else:
            for mol in mols:
                if mol is not None:
                    if mol.GetProp("_Name") in self.exclude:
                        print("{} is excluded".format(mol.GetProp("_Name")))
                    else:
                        if mol.GetNumAtoms() > 200:
                            continue
                        # Multitask
                        if type(self.target_name) is list:
                            y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in self.target_name])
                            self.outputs = len(self.target_name)

                        # Singletask
                        elif self.target_name in mol.GetPropNames():
                            _y = float(mol.GetProp(self.target_name))
                            if _y == -1:
                                continue
                            else:
                                y.append(_y)

                        else:
                            continue

                        x.append(mol)
                        c.append(mol.GetConformer().GetPositions())
                assert len(x) == len(y)

        # Filter and update maximum number of atoms
        new_x, new_c, new_y = [], [], []
        if self.max_atoms > 0:
            for mol, coo, tar in zip(x, c, y):
                if mol.GetNumAtoms() <= self.max_atoms:
                    new_x.append(mol)
                    new_c.append(coo)
                    new_y.append(tar)

            x = new_x
            c = new_c
            y = new_y

        else:
            for mol, tar in zip(x, y):
                self.max_atoms = max(self.max_atoms, mol.GetNumAtoms())

        if self.task != "regression":
            self.mols, self.coords, self.target = np.array(x), np.array(c), np.array(y, dtype=int)
        else:
            self.mols, self.coords, self.target = np.array(x), np.array(c), np.array(y)

        # Shuffle data
        idx_inactive = np.squeeze(np.argwhere(self.target == 0))  # 825
        idx_active = np.squeeze(np.argwhere(self.target == 1))  # 653
        idx_augmented_inactive = np.squeeze(np.argwhere(self.target == 1))  # 653

        np.random.seed(100)  # 37
        np.random.shuffle(idx_inactive)
        np.random.seed(100)  # 37
        np.random.shuffle(idx_active)
        np.random.seed(100) # 37
        np.random.shuffle(idx_augmented_inactive)

        # Split data
        len_inactive = int(len(idx_inactive) / fold)
        len_active = int(len(idx_active) / fold)
        len_augmented_inactive = int(len(idx_augmented_inactive) / fold)

        test_idx = np.append(idx_inactive[len_inactive*iter:len_inactive*(iter+1)], idx_active[len_active*iter:len_active*(iter+1)])
        train_idx = np.append(np.append(idx_inactive[:len_inactive*iter], idx_inactive[len_inactive*(iter+1):]),
                              np.append(idx_active[:len_active*iter], idx_active[len_active*(iter+1):]))
        # not using augmented data for test.
        augmented_idx = np.append(idx_augmented_inactive[:len_augmented_inactive * iter],
                                  idx_augmented_inactive[len_augmented_inactive * (iter + 1):])

        augmented_idx = np.random.permutation(augmented_idx)
        test_idx = np.random.permutation(test_idx)
        train_idx = np.random.permutation(train_idx)

        rotated_coords = []
        for i in range(3):  
            for coords in self.coords[augmented_idx]:
                degree = np.random.uniform(90, 270)
                axis = random.choice(["x", "y", "z"])
                rotation_matrix = degree_rotation_matrix(axis, degree)
                new_coords = np.matmul(coords, rotation_matrix)
                rotated_coords.append(new_coords)
        rotated_coords = np.asarray(rotated_coords, dtype=object)

        x_train = np.append(self.mols[train_idx], self.mols[augmented_idx])
        for i in range(2):
            x_train = np.append(x_train, self.mols[augmented_idx])
        c_train = np.append(self.coords[train_idx], rotated_coords)
        y_train = np.append(self.target[train_idx], np.zeros(3 * len(augmented_idx), dtype=int))

        self.x = {"train": x_train, "test": self.mols[test_idx]}
        self.c = {"train": c_train, "test": self.coords[test_idx]}
        self.y = {"train": y_train, "test": self.target[test_idx]}

        print('Finish stratified random degree&axis augmented splitting')

    def save_dataset(self, path, pred=None, target="test", filename=None):
        mols = []
        for idx, (x, c, y) in enumerate(zip(self.x[target], self.c[target], self.y[target])):
            x.SetProp("true", str(y * self.std + self.mean))
            if pred is not None:
                x.SetProp("pred", str(pred[idx][0] * self.std + self.mean))
            mols.append(x)

        if filename is not None:
            w = Chem.SDWriter(path + filename + ".sdf")
        else:
            w = Chem.SDWriter(path + target + ".sdf")
        for mol in mols:
            if mol is not None:
                w.write(mol)

    def replace_dataset(self, path, subset="test", target_name="target"):
        x, c, y = [], [], []
        mols = Chem.SDMolSupplier(path)

        for mol in mols:
            if mol is not None:
                # Multitask
                if type(target_name) is list:
                    y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in target_name])
                    self.outputs = len(self.target_name)

                # Singletask

                elif target_name in mol.GetPropNames():
                    _y = float(mol.GetProp(target_name))

                    if _y == -1:
                        continue
                    else:
                        y.append(_y)

                else:
                    y.append(float(0))
                x.append(mol)
                c.append(mol.GetConformer().GetPositions())

        # Normalize
        x = np.array(x)
        c = np.array(c)
        y = (np.array(y) - self.mean) / self.std

        self.x[subset] = x
        self.c[subset] = c
        self.y[subset] = y.astype(int) if self.task != "regression" else y

    def set_features(self, use_atom_symbol=True, use_degree=True, use_hybridization=True, use_implicit_valence=True,
                     use_partial_charge=False, use_formal_charge=True, use_ring_size=True, use_hydrogen_bonding=True,
                     use_acid_base=True, use_aromaticity=True, use_chirality=True, use_num_hydrogen=True):

        self.use_atom_symbol = use_atom_symbol
        self.use_degree = use_degree
        self.use_hybridization = use_hybridization
        self.use_implicit_valence = use_implicit_valence
        self.use_partial_charge = use_partial_charge
        self.use_formal_charge = use_formal_charge
        self.use_ring_size = use_ring_size
        self.use_hydrogen_bonding = use_hydrogen_bonding
        self.use_acid_base = use_acid_base
        self.use_aromaticity = use_aromaticity
        self.use_chirality = use_chirality
        self.use_num_hydrogen = use_num_hydrogen

        # Calculate number of features
        mp = MPGenerator([], [], [], 1,
                         use_atom_symbol=self.use_atom_symbol,
                         use_degree=self.use_degree,
                         use_hybridization=self.use_hybridization,
                         use_implicit_valence=self.use_implicit_valence,
                         use_partial_charge=self.use_partial_charge,
                         use_formal_charge=self.use_formal_charge,
                         use_ring_size=self.use_ring_size,
                         use_hydrogen_bonding=self.use_hydrogen_bonding,
                         use_acid_base=self.use_acid_base,
                         use_aromaticity=self.use_aromaticity,
                         use_chirality=self.use_chirality,
                         use_num_hydrogen=self.use_num_hydrogen)
        self.num_features = mp.get_num_features()

    def generator(self, target, task=None):
        return MPGenerator(self.x[target], self.c[target], self.y[target], self.batch,
                           task=task if task is not None else self.task,
                           num_atoms=self.max_atoms,
                           use_atom_symbol=self.use_atom_symbol,
                           use_degree=self.use_degree,
                           use_hybridization=self.use_hybridization,
                           use_implicit_valence=self.use_implicit_valence,
                           use_partial_charge=self.use_partial_charge,
                           use_formal_charge=self.use_formal_charge,
                           use_ring_size=self.use_ring_size,
                           use_hydrogen_bonding=self.use_hydrogen_bonding,
                           use_acid_base=self.use_acid_base,
                           use_aromaticity=self.use_aromaticity,
                           use_chirality=self.use_chirality,
                           use_num_hydrogen=self.use_num_hydrogen)


class MPGenerator(Sequence):
    def __init__(self, x_set, c_set, y_set, batch, task="binary", num_atoms=0,
                 use_degree=True, use_hybridization=True, use_implicit_valence=True, use_partial_charge=False,
                 use_formal_charge=True, use_ring_size=True, use_hydrogen_bonding=True, use_acid_base=True,
                 use_aromaticity=True, use_chirality=True, use_num_hydrogen=True, use_atom_symbol=True):
        self.x, self.c, self.y = x_set, c_set, y_set

        self.batch = batch
        self.task = task
        self.num_atoms = num_atoms

        self.use_atom_symbol = use_atom_symbol
        self.use_degree = use_degree
        self.use_hybridization = use_hybridization
        self.use_implicit_valence = use_implicit_valence
        self.use_partial_charge = use_partial_charge
        self.use_formal_charge = use_formal_charge
        self.use_ring_size = use_ring_size
        self.use_hydrogen_bonding = use_hydrogen_bonding
        self.use_acid_base = use_acid_base
        self.use_aromaticity = use_aromaticity
        self.use_chirality = use_chirality
        self.use_num_hydrogen = use_num_hydrogen

        self.hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        self.hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
        self.acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        self.basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch:(idx + 1) * self.batch]
        batch_c = self.c[idx * self.batch:(idx + 1) * self.batch]
        batch_y = self.y[idx * self.batch:(idx + 1) * self.batch]

        if self.task == "category":
            return self.tensorize(batch_x, batch_c), to_categorical(batch_y)
        elif self.task == "binary":
            return self.tensorize(batch_x, batch_c), np.array(batch_y, dtype=int)
        elif self.task == "regression":
            return self.tensorize(batch_x, batch_c), np.array(batch_y, dtype=float)
        elif self.task == "input_only":
            return self.tensorize(batch_x, batch_c)

    def tensorize(self, batch_x, batch_c):
        atom_tensor = np.zeros((len(batch_x), self.num_atoms, self.get_num_features()))
        adjm_tensor = np.zeros((len(batch_x), self.num_atoms, self.num_atoms))
        posn_tensor = np.zeros((len(batch_x), self.num_atoms, self.num_atoms, 3))

        for mol_idx, mol in enumerate(batch_x):
            Chem.RemoveHs(mol)
            mol_atoms = mol.GetNumAtoms()

            # Atom features
            atom_tensor[mol_idx, :mol_atoms, :] = self.get_atom_features(mol)

            # Adjacency matrix
            adjms = np.array(rdmolops.GetAdjacencyMatrix(mol), dtype="float")

            # Normalize adjacency matrix by D^(-1/2) * A_hat * D^(-1/2), Kipf et al. 2016
            adjms += np.eye(mol_atoms)
            degree = np.array(adjms.sum(1))
            deg_inv_sqrt = np.power(degree, -0.5)
            deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
            deg_inv_sqrt = np.diag(deg_inv_sqrt)

            adjms = np.matmul(np.matmul(deg_inv_sqrt, adjms), deg_inv_sqrt)

            adjm_tensor[mol_idx, : mol_atoms, : mol_atoms] = adjms

            # Relative position matrix
            for atom_idx in range(mol_atoms):
                pos_c = batch_c[mol_idx][atom_idx]

                for neighbor_idx in range(mol_atoms):
                    pos_n = batch_c[mol_idx][neighbor_idx]

                    # Direction should be Neighbor -> Center
                    n_to_c = [pos_c[0] - pos_n[0], pos_c[1] - pos_n[1], pos_c[2] - pos_n[2]]
                    posn_tensor[mol_idx, atom_idx, neighbor_idx, :] = n_to_c

        return [atom_tensor, adjm_tensor, posn_tensor]

    def get_num_features(self):
        mol = Chem.MolFromSmiles("CC")
        return len(self.get_atom_features(mol)[0])

    def get_atom_features(self, mol):
        AllChem.ComputeGasteigerCharges(mol)
        Chem.AssignStereochemistry(mol)

        hydrogen_donor_match = sum(mol.GetSubstructMatches(self.hydrogen_donor), ())
        hydrogen_acceptor_match = sum(mol.GetSubstructMatches(self.hydrogen_acceptor), ())
        acidic_match = sum(mol.GetSubstructMatches(self.acidic), ())
        basic_match = sum(mol.GetSubstructMatches(self.basic), ())

        ring = mol.GetRingInfo()

        m = []
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)

            o = []
            o += one_hot(atom.GetSymbol(), ['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P',
                                            'I', 'Si', 'B', 'Na', 'Sn', 'Se', 'other']) if self.use_atom_symbol else []
            o += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) if self.use_degree else []
            o += one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                   Chem.rdchem.HybridizationType.SP2,
                                                   Chem.rdchem.HybridizationType.SP3,
                                                   Chem.rdchem.HybridizationType.SP3D,
                                                   Chem.rdchem.HybridizationType.SP3D2]) if self.use_hybridization else []
            o += one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) if self.use_implicit_valence else []
            o += one_hot(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3]) if self.use_degree else []
            # o += [atom.GetProp("_GasteigerCharge")] if self.use_partial_charge else [] # some molecules return NaN
            o += [atom.GetIsAromatic()] if self.use_aromaticity else []
            o += [ring.IsAtomInRingOfSize(atom_idx, 3),
                  ring.IsAtomInRingOfSize(atom_idx, 4),
                  ring.IsAtomInRingOfSize(atom_idx, 5),
                  ring.IsAtomInRingOfSize(atom_idx, 6),
                  ring.IsAtomInRingOfSize(atom_idx, 7),
                  ring.IsAtomInRingOfSize(atom_idx, 8)] if self.use_ring_size else []
            o += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) if self.use_num_hydrogen else []

            if self.use_chirality:
                try:
                    o += one_hot(atom.GetProp('_CIPCode'), ["R", "S"]) + [atom.HasProp("_ChiralityPossible")]
                except:
                    o += [False, False] + [atom.HasProp("_ChiralityPossible")]
            if self.use_hydrogen_bonding:
                o += [atom_idx in hydrogen_donor_match]
                o += [atom_idx in hydrogen_acceptor_match]
            if self.use_acid_base:
                o += [atom_idx in acidic_match]
                o += [atom_idx in basic_match]

            m.append(o)

        return np.array(m, dtype=float)

