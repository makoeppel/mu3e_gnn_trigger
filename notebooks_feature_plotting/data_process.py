import uproot
import numpy as np
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
import pickle
import os
from scipy.constants import c
import sys

sys.path.append("..")

DATA_DIR = "../mu3e_trigger_data"
ROOT_DATA_DIR = "../mu3e_root_data"
signal_data_file = f"{ROOT_DATA_DIR}/run_sig_simi-sort.root"
background_data_file = f"{ROOT_DATA_DIR}/run_bg_simi-sort.root"
e5_data_file = f"{ROOT_DATA_DIR}/run42_5e-sort.root"
familong_data_file = f"{ROOT_DATA_DIR}/run_familon_simi-sort.root"
signal_only_data_file = f"{ROOT_DATA_DIR}/run42_sig_only-sort.root"

HIT_COUNT_CUTOFF = 256
from src.data_preparation import convert_root_to_npy
convert_root_to_npy(
    file_path=signal_data_file,
out_dir=DATA_DIR,
    out_name="sig",
    padding_value=-1,
    hit_cutoff=HIT_COUNT_CUTOFF,
    add_layer_as_feature=True,
)
convert_root_to_npy(
    file_path=background_data_file,
    out_dir=DATA_DIR,
    out_name="bg",
    padding_value=-1,
    hit_cutoff=HIT_COUNT_CUTOFF,
    add_layer_as_feature=True,
)
convert_root_to_npy(
    file_path=e5_data_file,
    out_dir=DATA_DIR,
    out_name="5e",
    padding_value=-1,
    hit_cutoff=HIT_COUNT_CUTOFF,
    add_layer_as_feature=True,
)
convert_root_to_npy(
    file_path=familong_data_file,
    out_dir=DATA_DIR,
    out_name="familon",
    padding_value=-1,
    hit_cutoff=HIT_COUNT_CUTOFF,
    add_layer_as_feature=True,
)
convert_root_to_npy(
    file_path=signal_only_data_file,
    out_dir=DATA_DIR,
    out_name="sig_only",
    padding_value=-1,
    hit_cutoff=HIT_COUNT_CUTOFF,
    add_layer_as_feature=True,
)