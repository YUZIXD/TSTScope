
import numpy as np
import torch
from typing import Optional
import re
import sys
import collections.abc as container_abcs

def one_hot_encoder(idx, n_cls):
    assert torch.max(idx).item() < n_cls
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n_cls)
    onehot = onehot.to(idx.device)
    onehot.scatter_(1, idx.long(), 1)
    return onehot

def label_encoder(adata, encoder, condition_key=None):
     
    """Encode labels of Annotated `adata` matrix.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
       encoder: Dict
            dictionary of encoded labels.
       condition_key: String
            column name of conditions in `adata.obs` data frame.

       Returns
       -------
       labels: `~numpy.ndarray`
            Array of encoded labels
       label_encoder: Dict
            dictionary with labels and encoded labels as key, value pairs.
    """
    unique_conditions = list(np.unique(adata.obs[condition_key]))
    labels = np.zeros(adata.shape[0])

    if not set(unique_conditions).issubset(set(encoder.keys())):
        missing_labels = set(unique_conditions).difference(set(encoder.keys()))
        print(f"Warning: Labels in adata.obs[{condition_key}] is not a subset of label-encoder!")
        print(f"The missing labels are: {missing_labels}")
        print("Therefore integer value of those labels is set to -1")
        for data_cond in unique_conditions:
            if data_cond not in encoder.keys():
                labels[adata.obs[condition_key] == data_cond] = -1

    for condition, label in encoder.items():
        labels[adata.obs[condition_key] == condition] = label
    return labels

class EarlyStopping(object):
    """Class for EarlyStopping. This class contains the implementation of early stopping for TRVAE/CVAE training.

       This early stopping class was inspired by:
       Title: scvi-tools
       Authors: Romain Lopez <romain_lopez@gmail.com>,
                Adam Gayoso <adamgayoso@berkeley.edu>,
                Galen Xing <gx2113@columbia.edu>
       Date: 24th December 2020
       Code version: 0.8.1
       Availability: https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/trainers/trainer.py

           Parameters
           ----------
           early_stopping_metric: : String
                The metric/loss which the early stopping criterion gets calculated on.
           threshold: Float
                The minimum value which counts as improvement.
           patience: Integer
                Number of epochs which are allowed to have no improvement until the training is stopped.
           reduce_lr: Boolean
                If 'True', the learning rate gets adjusted by 'lr_factor' after a given number of epochs with no
                improvement.
           lr_patience: Integer
                Number of epochs which are allowed to have no improvement until the learning rate is adjusted.
           lr_factor: Float
                Scaling factor for adjusting the learning rate.
        """
    def __init__(self,
                 early_stopping_metric: str = "val_unweighted_loss",
                 mode: str = "min",
                 threshold: float = 0,
                 patience: int = 20,
                 reduce_lr: bool = True,
                 lr_patience: int = 13,
                 lr_factor: float = 0.1):

        self.early_stopping_metric = early_stopping_metric
        self.mode = mode
        self.threshold = threshold
        self.patience = patience
        self.reduce_lr = reduce_lr
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor

        self.epoch = 0
        self.wait = 0
        self.wait_lr = 0
        self.current_performance = np.inf
        if self.mode == "min":
            self.best_performance = np.inf
            self.best_performance_state = np.inf
        else:
            self.best_performance = -np.inf
            self.best_performance_state = -np.inf

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, scalar):
        self.epoch += 1
        if self.epoch < self.patience:
            continue_training = True
            lr_update = False
        elif self.wait >= self.patience:
            continue_training = False
            lr_update = False
        else:
            if not self.reduce_lr:
                lr_update = False
            elif self.wait_lr >= self.lr_patience:
                lr_update = True
                self.wait_lr = 0
            else:
                lr_update = False
            # Shift
            self.current_performance = scalar
            if self.mode == "min":
                improvement = self.best_performance - self.current_performance
            else:
                improvement = self.current_performance - self.best_performance

            # updating best performance
            if improvement > 0:
                self.best_performance = self.current_performance

            if improvement < self.threshold:
                self.wait += 1
                self.wait_lr += 1
            else:
                self.wait = 0
                self.wait_lr = 0

            continue_training = True

        if not continue_training:
            print("\nStopping early: no improvement of more than " + str(self.threshold) +
                  " nats in " + str(self.patience) + " epochs")
            print("If the early stopping criterion is too strong, "
                  "please instantiate it with different parameters in the train method.")
        return continue_training, lr_update

    def update_state(self, scalar):
        if self.mode == "min":
            improved = (self.best_performance_state - scalar) > 0
        else:
            improved = (scalar - self.best_performance_state) > 0

        if improved:
            self.best_performance_state = scalar
        return improved

def print_progress(epoch, logs, n_epochs=10000, only_val_losses=True):
    """Creates Message for '_print_progress_bar'.

       Parameters
       ----------
       epoch: Integer
            Current epoch iteration.
       logs: Dict
            Dictionary of all current losses.
       n_epochs: Integer
            Maximum value of epochs.
       only_val_losses: Boolean
            If 'True' only the validation dataset losses are displayed, if 'False' additionally the training dataset
            losses are displayed.

       Returns
       -------
    """
    message = ""
    for key in logs:
        if only_val_losses:
            if "val_" in key and "unweighted" not in key:
                message += f" - {key:s}: {logs[key][-1]:7.10f}"
        else:
            if "unweighted" not in key:
                message += f" - {key:s}: {logs[key][-1]:7.10f}"

    _print_progress_bar(epoch + 1, n_epochs, prefix='', suffix=message, decimals=1, length=20)

def _print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """Prints out message with a progress bar.

       Parameters
       ----------
       iteration: Integer
            Current epoch.
       total: Integer
            Maximum value of epochs.
       prefix: String
            String before the progress bar.
       suffix: String
            String after the progress bar.
       decimals: Integer
            Digits after comma for all the losses.
       length: Integer
            Length of the progress bar.
       fill: String
            Symbol for filling the bar.

       Returns
       -------
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_len = int(length * iteration // total)
    bar = fill * filled_len + '-' * (length - filled_len)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def train_test_split(adata, train_frac=0.85, condition_key=None, cell_type_key=None, labeled_array=None):
    """Splits 'Anndata' object into training and validation data.

       Parameters
       ----------
       adata: `~anndata.AnnData`
            `AnnData` object for training the model.
       train_frac: float
            Train-test split fraction. the model will be trained with train_frac for training
            and 1-train_frac for validation.

       Returns
       -------
       Indices for training and validating the model.
    """
    indices = np.arange(adata.shape[0])

    if train_frac == 1:
        return indices, None

    if cell_type_key is not None:
        labeled_array = np.zeros((len(adata), 1)) if labeled_array is None else labeled_array
        labeled_array = np.ravel(labeled_array)

        labeled_idx = indices[labeled_array == 1]
        unlabeled_idx = indices[labeled_array == 0]

        train_labeled_idx = []
        val_labeled_idx = []
        train_unlabeled_idx = []
        val_unlabeled_idx = []
        #TODO this is horribly inefficient
        if len(labeled_idx) > 0:
            cell_types = adata[labeled_idx].obs[cell_type_key].unique().tolist()
            for cell_type in cell_types:
                ct_idx = labeled_idx[adata[labeled_idx].obs[cell_type_key] == cell_type]
                n_train_samples = int(np.ceil(train_frac * len(ct_idx)))
                np.random.shuffle(ct_idx)
                train_labeled_idx.append(ct_idx[:n_train_samples])
                val_labeled_idx.append(ct_idx[n_train_samples:])
        if len(unlabeled_idx) > 0:
            n_train_samples = int(np.ceil(train_frac * len(unlabeled_idx)))
            train_unlabeled_idx.append(unlabeled_idx[:n_train_samples])
            val_unlabeled_idx.append(unlabeled_idx[n_train_samples:])
        train_idx = train_labeled_idx + train_unlabeled_idx
        val_idx = val_labeled_idx + val_unlabeled_idx

        train_idx = np.concatenate(train_idx)
        val_idx = np.concatenate(val_idx)

    elif condition_key is not None:
        train_idx = []
        val_idx = []
        conditions = adata.obs[condition_key].unique().tolist()
        for condition in conditions:
            cond_idx = indices[adata.obs[condition_key] == condition]
            n_train_samples = int(np.ceil(train_frac * len(cond_idx)))
            #np.random.shuffle(cond_idx)
            train_idx.append(cond_idx[:n_train_samples])
            val_idx.append(cond_idx[n_train_samples:])

        train_idx = np.concatenate(train_idx)
        val_idx = np.concatenate(val_idx)

    else:
        n_train_samples = int(np.ceil(train_frac * len(indices)))
        np.random.shuffle(indices)
        train_idx = indices[:n_train_samples]
        val_idx = indices[n_train_samples:]

    return train_idx, val_idx

def custom_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)

    elif isinstance(elem, container_abcs.Mapping):
        output = {key: custom_collate([d[key] for d in batch]) for key in elem}
        return output