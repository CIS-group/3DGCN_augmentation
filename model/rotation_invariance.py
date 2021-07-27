import numpy as np
import csv

import os
import sys
import argparse
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedFormatter, NullFormatter

# import modeuls from current directory
from trainer import Trainer
from callback import calculate_roc_pr, calculate_f1_acc
from dataset import MPGenerator

from rdkit import Chem


plt.rcParams["font.size"] = 16
plt.rcParams["axes.axisbelow"] = True


def load_model(trial_path, axis):
    with open(trial_path + "/hyper.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            hyper = dict(row)

    dataset = hyper["dataset"]
    model = hyper["model"]
    batch = int(hyper["batch"])
    units_conv = int(hyper["units_conv"])
    units_dense = int(hyper["units_dense"])
    num_layers = int(hyper["num_layers"])
    loss = hyper["loss"]
    pooling = hyper["pooling"]
    std = float(hyper["data_std"])
    mean = float(hyper["data_mean"])

    # Load model
    trainer = Trainer(dataset, split_type="stratified_augmented_sampling", axis=axis, degrees=[0])
    trainer.load_data(batch=batch, iter=1)
    trainer.data.std = std
    trainer.data.mean = mean
    trainer.load_model(
        model,
        units_conv=units_conv,
        units_dense=units_dense,
        num_layers=num_layers,
        loss=loss,
        pooling=pooling,
    )

    # Load best weight
    trainer.model.load_weights(trial_path + "/best_weight.hdf5")
    # print("Loaded Weights from {}".format(trial_path + "/best_weight.hdf5"))

    return trainer, hyper


def random_rotation_matrix():
    theta = np.random.rand() * 2 * np.pi
    r_x = np.array(
        [1, 0, 0, 0, np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta)]
    ).reshape([3, 3])
    theta = np.random.rand() * 2 * np.pi
    r_y = np.array(
        [np.cos(theta), 0, np.sin(theta), 0, 1, 0, -np.sin(theta), 0, np.cos(theta)]
    ).reshape([3, 3])
    theta = np.random.rand() * 2 * np.pi
    r_z = np.array(
        [np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta), 0, 0, 0, 1]
    ).reshape([3, 3])

    return np.matmul(np.matmul(r_x, r_y), r_z)


def degree_rotation_matrix(axis, degree):
    theta = degree / 180 * np.pi
    if axis == "x":
        r = np.array(
            [1, 0, 0, 0, np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta)]
        ).reshape([3, 3])
    elif axis == "y":
        r = np.array(
            [np.cos(theta), 0, np.sin(theta), 0, 1, 0, -np.sin(theta), 0, np.cos(theta)]
        ).reshape([3, 3])
    elif axis == "z":
        r = np.array(
            [np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta), 0, 0, 0, 1]
        ).reshape([3, 3])
    else:
        raise ValueError("Unsupported axis for rotation: {}".format(axis))

    return r


def rotation_prediction(path, select, rotation="stepwise", axis="x", degree_gap=10):
    keep_raw = []
    task = "classification"
    if task == "regression":
        metric1_mean = ["mae_mean"]
        metric1_std = ["mae_std"]
        metric2_mean = ["rmse_mean"]
        metric2_std = ["rmse_std"]
    else:
        metric1_mean = ["f1_mean"]
        metric1_std = ["f1_std"]
        metric2_mean = ["acc_mean"]
        metric2_std = ["acc_std"]

    degrees = list(range(0, 360, degree_gap))
    fold = ["trial_" in filename for filename in os.listdir(path)].count(True)
    for degree in degrees:
        # Iterate over trials
        raw_results = []
        task = None
        for i in range(0, fold):
            trial_path = path + "/trial_" + str(i)
            trainer, hyper = load_model(trial_path, axis=axis)

            # Rotate test dataset
            dataset_path = "/test_" + select + "_" + rotation + axis + str(degree) + ".sdf"
            if not os.path.exists(trial_path + dataset_path):
                mols = Chem.SDMolSupplier(trial_path + "/test.sdf")

                rotated_mols = []
                # print("Rotating Molecules... Rule: {}".format(rotation))
                for mol in mols:
                    if select == "active":
                        if mol.GetProp("active") == "0":
                            continue
                    elif select == "inactive":
                        if mol.GetProp("active") == "1":
                            continue
                    if degree == 0:
                        rotated_mols.append(mol)
                        continue
                    elif rotation == "random":
                        rotation_matrix = random_rotation_matrix()
                    elif rotation == "stepwise":
                        rotation_matrix = degree_rotation_matrix(axis, float(degree))
                    else:
                        raise ValueError("Unsupported rotation mechanism: {}".format(rotation))

                    for atom in mol.GetAtoms():
                        atom_idx = atom.GetIdx()

                        pos = list(mol.GetConformer().GetAtomPosition(atom_idx))
                        pos_rotated = np.matmul(rotation_matrix, pos)

                        mol.GetConformer().SetAtomPosition(atom_idx, pos_rotated)
                    rotated_mols.append(mol)

                # Save rotated test dataset
                w = Chem.SDWriter(trial_path + dataset_path)
                for m in rotated_mols:
                    if m is not None:
                        w.write(m)

            # Load rotation test dataset
            trainer.data.replace_dataset(
                trial_path + dataset_path, subset="test", target_name="true"
            )

            # Predict
            if hyper["loss"] == "mse":
                test_loss = trainer.model.evaluate_generator(trainer.data.generator("test"))
                pred = trainer.model.predict_generator(
                    trainer.data.generator("test", task="input_only")
                )
                raw_results.append([test_loss[1], test_loss[2]])

            else:
                f1, acc, y_true, y_pred = calculate_f1_acc(
                    trainer.model, trainer.data.generator("test"), return_pred=True
                )
                raw_results.append([f1, acc])

        # Save results
        keep_raw.append(np.transpose(np.array(raw_results)))
        results_mean = np.array(raw_results).mean(axis=0)
        results_std = np.array(raw_results).std(axis=0)
        metric1_mean.append(results_mean[0])
        metric1_std.append(results_std[0])
        metric2_mean.append(results_mean[1])
        metric2_std.append(results_std[1])

        # print(axis, degree, '   f1_result: ', results_mean[0], '+-', results_std[0])
        print(axis, degree, "finished.")

    header = [axis] + [str(degree) for degree in degrees]
    with open(
        path + "/" + select + "_rotation_" + axis + ".csv", "w"
    ) as file:  # path + "/rotation_" + rotation + axis + str(degree) + ".csv", "w"
        writer = csv.writer(file, delimiter=",")
        writer.writerow(header)
        writer.writerow(metric1_mean)
        writer.writerow(metric1_std)
        writer.writerow(metric2_mean)
        writer.writerow(metric2_std)

    header = [str(fold) + "-fold"]
    with open(path + "/" + select + "_rotation_" + axis + "_raw.csv", "w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(header)
        trial = 0
        degree = 0
        for r in keep_raw:
            writer.writerow([axis + str(degree) + "_f1"] + list(r[0]))
            writer.writerow([axis + str(degree) + "_acc"] + list(r[1]))
            trial += 1
            degree += degree_gap


def rotate_molecule(mol, axis, degree):
    rotation_matrix = degree_rotation_matrix(axis, float(degree))
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        pos = list(mol.GetConformer().GetAtomPosition(atom_idx))
        pos_rotated = np.matmul(rotation_matrix, pos)
        mol.GetConformer().SetAtomPosition(atom_idx, pos_rotated)
    return mol


def single_molecule_rotation_sigmoid(path, fold, axis):
    # Iterate over trials
    trial_path = path + "/trial_" + str(fold)

    trainer, hyper = load_model(trial_path, axis=axis)

    data = Chem.SDMolSupplier(trial_path + "/test.sdf")
    names = []
    results = []
    for mol in data:
        if mol is not None:  # and mol.GetProp("active") == "1":
            ligand_name = mol.GetProp("_Name")
            print(ligand_name, "is processing")
            names.append(ligand_name)
            target_ligand = mol
            ligand_path = "/" + ligand_name + "_rotated_" + axis + ".sdf"

            rotated_target_ligand = []
            degrees = list(range(0, 360, 10))
            for degree in degrees:
                rotation_matrix = degree_rotation_matrix(axis, float(degree))
                rotate_ligand = Chem.Mol(target_ligand)
                for atom in rotate_ligand.GetAtoms():
                    atom_idx = atom.GetIdx()
                    pos = list(rotate_ligand.GetConformer().GetAtomPosition(atom_idx))
                    pos_rotated = np.matmul(rotation_matrix, pos)
                    rotate_ligand.GetConformer().SetAtomPosition(atom_idx, pos_rotated)
                rotated_target_ligand.append(rotate_ligand)

            if not os.path.exists(trial_path + ligand_path):
                w = Chem.SDWriter(trial_path + ligand_path)
                for m in rotated_target_ligand:
                    if m is not None:
                        w.write(m)

            trainer.data.replace_dataset(
                trial_path + ligand_path, subset="test", target_name="target"
            )
            pred = trainer.model.predict_generator(trainer.data.generator("test"))
            pred = np.asarray(pred)
            pred = np.reshape(pred, (-1))
            results.append(pred)

    header = ["Name"] + [str(degree) for degree in degrees]
    with open(trial_path + "/" + "active_rotation_sigmoid_" + axis + ".csv", "w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(header)
        for name, sigmoid in zip(names, results):
            writer.writerow([name] + list(sigmoid))
    print("end")


def global_molecule_rotation_sigmoid(path, fold, axis, excludes=[]):
    # Iterate over trials
    degrees = list(range(0, 360, 10))
    for name in excludes:
        print(name, "is processing")
        total_result = []
        for i in range(0, fold):
            trial_path = path + "/trial_" + str(i)

            trainer, hyper = load_model(trial_path, axis=axis)

            trainer.data.replace_dataset(
                path + "/" + name + "_rotated_60_" + axis + ".sdf",
                subset="test",
                target_name="target",
            )
            pred = trainer.model.predict_generator(trainer.data.generator("test"))
            pred = np.asarray(pred)
            pred = np.reshape(pred, (-1))
            total_result.append(pred)
        total_result = np.asarray(total_result)
        mean = np.mean(total_result, axis=0)
        std = np.std(total_result, axis=0)

        header = [axis] + [str(degree) for degree in degrees]
        with open(
            path + "/" + name + "_gloabl_rotation_sigmoid_rotated_60_" + axis + ".csv", "w"
        ) as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerow(header)
            writer.writerow(["mean"] + list(mean))
            writer.writerow(["std"] + list(std))
    print("end")


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=0, type=int, help="GPU number you want to use")
parser.add_argument(
    "--select", default="active", help="Select data type: active or inactive"
)  # 'active' or 'inactive'
parser.add_argument("--exp_name", default="", help="name of training experiment")
parser.add_argument("--date", default="062219", help="the date of experiment")
parser.add_argument("--axis", default="x", help="axis for testing rotation invariance")
parser.add_argument(
    "--dg", "--degree_gap", default=10, type=int, help="Degree gap you want to rotate"
)
parser.add_argument(
    "--split_type",
    default="stratified_augmented_sampling",
    choices=[
        "stratified_sampling",
        "stratified_augmented_sampling",
        "stratified_random_degree_augmented_sampling",
        "stratified_random_degree_axis_augmented_sampling",
    ],
    help="choose type of dataset sampling you used for training",
)

args = parser.parse_args()
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    path = "./result/model_3DGCN/bace_cla/16_c128_d128_l2_p" + args.exp_name + args.date
    # rotation_prediction(new_path, select="active", rotation="stepwise", degree_gap=args.dg, axis="z")
    # single_molecule_rotation_sigmoid(path, 3, "x")
    # global_molecule_rotation_sigmoid(path, 10, "x", ["CHEMBL1916159"])
