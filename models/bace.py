from trainer import Trainer
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pooling', default='max', help="Pooling type")
parser.add_argument('--augmented_axis', default='x', help='axis you want to augment')
parser.add_argument('--gpu', default='0', help="GPU number you want to use")

parser.add_argument('--fold', default=10, type=int, help="k-fold cross validation")
parser.add_argument('--degrees', nargs="*", default=[180, -90, 90], help="list of degrees which will be used to make augmented data")
parser.add_argument('--split_type', default="stratified_augmented_sampling", 
choices=["stratified_sampling", "stratified_augmented_sampling", "stratified_random_degree_augmented_sampling", "stratified_random_degree_axis_augmented_sampling"],
help="choose type of dataset sampling")
parser.add_argument('--exlcude_list', nargs="*", default=[], help="list for names of exclued ligand for global exlusion experiment")
args = parser.parse_args()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    trainer = Trainer("bace_cla", split_type=args.split_type, axis=args.augmented_axis, degrees=args.degrees, exclude=args.exclude_list)
    hyperparameters = {"epoch": 150, "batch": 16, "fold": args.fold, "units_conv": 128, "units_dense": 128, "pooling": args.pooling,
                       "num_layers": 2, "loss": "binary_crossentropy", "monitor": "val_roc", "label": "", "split_type": args.split_type}

    features = {"use_atom_symbol": True, "use_degree": True, "use_hybridization": True, "use_implicit_valence": True,
                "use_partial_charge": True, "use_ring_size": True, "use_hydrogen_bonding": True,
                "use_acid_base": True, "use_aromaticity": True, "use_chirality": True, "use_num_hydrogen": True}

    trainer.fit("model_3DGCN", **hyperparameters, **features)

