from collections import Counter, defaultdict
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler
from scipy import sparse

from .utils import label_encoder,train_test_split,print_progress,EarlyStopping,custom_collate
import time

def loss(model, total_batch=None, coeff = 1.0, tcr_weight = 0.001, gene_weight = 1.0, align_weight = 0.1, hsic_weight = 1.0):
    
    alpha_kl = coeff
    gene_recon_loss, tcr_recon_loss, tcr_hsic_loss, kl_loss, align_loss = model(**total_batch)
    loss = gene_weight*gene_recon_loss + tcr_weight*tcr_recon_loss + alpha_kl*kl_loss + hsic_weight*tcr_hsic_loss + align_weight*align_loss

    return loss, gene_recon_loss, tcr_recon_loss, tcr_hsic_loss, kl_loss, align_loss

class tcrDataset(Dataset):
    """Dataset handler for TSTScope model and trainer.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
       condition_key: String
            column name of conditions in `adata.obs` data frame.
       condition_encoder: Dict
            dictionary of encoded conditions.
       cell_type_keys: List
            List of column names of different celltype hierarchies in `adata.obs` data frame.
       cell_type_encoder: Dict
            dictionary of encoded celltypes.
    """
    def __init__(self,
                 adata,
                 condition_key=None,
                 condition_encoder=None,
                 cell_type_keys=None,
                 cell_type_encoder=None,
                 labeled_array=None
                 ):

        self.condition_key = condition_key
        self.condition_encoder = condition_encoder
        self.cell_type_keys = cell_type_keys
        self.cell_type_encoder = cell_type_encoder

        self._is_sparse = sparse.issparse(adata.X)
        self.data = adata.X if self._is_sparse else torch.tensor(adata.X)

        # convert tcr data: scCVC embeddings
        self.tcremb = torch.tensor(adata.obsm['tcremb']).to(torch.float32)
        self.tcrlabel = torch.tensor(adata.obs['tcrlabel']).to(torch.float32)

        size_factors = np.ravel(adata.X.sum(1))
        self.size_factors = torch.tensor(size_factors)

        labeled_array = np.zeros((len(adata), 1)) if labeled_array is None else labeled_array
        self.labeled_vector = torch.tensor(labeled_array)

        # Encode condition strings to integer
        if self.condition_key is not None:
            self.conditions = label_encoder(
                adata,
                encoder=self.condition_encoder,
                condition_key=condition_key,
            )
            self.conditions = torch.tensor(self.conditions, dtype=torch.long)

        # Encode cell type strings to integer
        if self.cell_type_keys is not None:
            self.cell_types = list()
            for cell_type_key in cell_type_keys:
                level_cell_types = label_encoder(
                    adata,
                    encoder=self.cell_type_encoder,
                    condition_key=cell_type_key,
                )
                self.cell_types.append(level_cell_types)

            self.cell_types = np.stack(self.cell_types).T
            self.cell_types = torch.tensor(self.cell_types, dtype=torch.long)

    def __getitem__(self, index):

        outputs = dict()
        if self._is_sparse:
            x = torch.tensor(np.squeeze(self.data[index].toarray()))
        else:
            x = self.data[index]
            
        outputs["x"] = x
        # tcr emb
        outputs["tcremb"] = self.tcremb[index]
        outputs["tcrlabel"] = self.tcrlabel[index]
        
        outputs["labeled"] = self.labeled_vector[index]
        outputs["sizefactor"] = self.size_factors[index]

        if self.condition_key:
            outputs["batch"] = self.conditions[index]

        if self.cell_type_keys:
            outputs["celltypes"] = self.cell_types[index, :]

        return outputs

    def __len__(self):
        return self.data.shape[0]

    @property
    def condition_label_encoder(self) -> dict:
        return self.condition_encoder

    @condition_label_encoder.setter
    def condition_label_encoder(self, value: dict):
        if value is not None:
            self.condition_encoder = value

    @property
    def cell_type_label_encoder(self) -> dict:
        return self.cell_type_encoder

    @cell_type_label_encoder.setter
    def cell_type_label_encoder(self, value: dict):
        if value is not None:
            self.cell_type_encoder = value

    @property
    def stratifier_weights(self):
        conditions = self.conditions.detach().cpu().numpy()
        condition_coeff = 1. / len(conditions)

        condition2count = Counter(conditions)
        counts = np.array([condition2count[cond] for cond in conditions])
        weights = condition_coeff / counts
        return weights.astype(float)

# make tcr dataset, add "tcremb" key
def make_tcr_dataset(adata,
                 train_frac=0.9,
                 condition_key=None,
                 cell_type_keys=None,
                 condition_encoder=None,
                 cell_type_encoder=None,
                 labeled_indices=None,
                 ):
    """Splits 'adata' into train and validation data and converts them into 'CustomDatasetFromAdata' objects.

       Parameters
       ----------

       Returns
       -------
       Training 'CustomDatasetFromAdata' object, Validation 'CustomDatasetFromAdata' object
    """
    # Preprare data for semisupervised learning
    print(f"Preparing {adata.shape}")
    labeled_array = np.zeros((len(adata), 1))
    if labeled_indices is not None:
        labeled_array[labeled_indices] = 1

    if cell_type_keys is not None:
        finest_level = None
        n_cts = 0
        for cell_type_key in cell_type_keys:
            if len(adata.obs[cell_type_key].unique().tolist()) >= n_cts:
                n_cts = len(adata.obs[cell_type_key].unique().tolist())
                finest_level = cell_type_key
        print(f"Splitting data {adata.shape}")
        train_idx, val_idx = train_test_split(adata, train_frac, cell_type_key=finest_level,
                                              labeled_array=labeled_array)
    
    elif condition_key is not None:
        train_idx, val_idx = train_test_split(adata, train_frac, condition_key=condition_key)
    else:
        train_idx, val_idx = train_test_split(adata, train_frac)
        
    print("Instantiating dataset")

    # build tcr specific Dataset
    data_set_train = tcrDataset(
        adata if train_frac == 1 else adata[train_idx],
        condition_key=condition_key,
        cell_type_keys=cell_type_keys,
        condition_encoder=condition_encoder,
        cell_type_encoder=cell_type_encoder,
        labeled_array=labeled_array[train_idx]
    )
    if train_frac == 1:
        return data_set_train, None
    else:
        data_set_valid = tcrDataset(
            adata[val_idx],
            condition_key=condition_key,
            cell_type_keys=cell_type_keys,
            condition_encoder=condition_encoder,
            cell_type_encoder=cell_type_encoder,
            labeled_array=labeled_array[val_idx]
        )
        return data_set_train, data_set_valid

# trainer of tstscopeTrainer
class tcrTrainer:
    """ScArches base Trainer class. This class contains the implementation of the base CVAE/TRVAE Trainer.

       Parameters
       ----------
       model: trVAE
            Number of input features (i.e. gene in case of scRNA-seq).
       adata: : `~anndata.AnnData`
            Annotated data matrix. Has to be count data for 'nb' and 'zinb' loss and normalized log transformed data
            for 'mse' loss.
       condition_key: String
            column name of conditions in `adata.obs` data frame.
       cell_type_keys: List
            List of column names of different celltype levels in `adata.obs` data frame.
       batch_size: Integer
            Defines the batch size that is used during each Iteration
       alpha_epoch_anneal: Integer or None
            If not 'None', the KL Loss scaling factor (alpha_kl) will be annealed from 0 to 1 every epoch until the input
            integer is reached.
       alpha_kl: Float
            Multiplies the KL divergence part of the loss.
       alpha_iter_anneal: Integer or None
            If not 'None', the KL Loss scaling factor will be annealed from 0 to 1 every iteration until the input
            integer is reached.
       use_early_stopping: Boolean
            If 'True' the EarlyStopping class is being used for training to prevent overfitting.
       reload_best: Boolean
            If 'True' the best state of the model during training concerning the early stopping criterion is reloaded
            at the end of training.
       early_stopping_kwargs: Dict
            Passes custom Earlystopping parameters.
       train_frac: Float
            Defines the fraction of data that is used for training and data that is used for validation.
       n_samples: Integer or None
            Defines how many samples are being used during each epoch. This should only be used if hardware resources
            are limited.
       use_stratified_sampling: Boolean
            If 'True', the sampler tries to load equally distributed batches concerning the conditions in every
            iteration.
       monitor: Boolean
            If `True', the progress of the training will be printed after each epoch.
       monitor_only_val: Boolean
            If `True', only the progress of the validation datset is displayed.
       clip_value: Float
            If the value is greater than 0, all gradients with an higher value will be clipped during training.
       weight decay: Float
            Defines the scaling factor for weight decay in the Adam optimizer.
       n_workers: Integer
            Passes the 'n_workers' parameter for the torch.utils.data.DataLoader class.
       seed: Integer
            Define a specific random seed to get reproducable results.
    """
    def __init__(self,
                 model,
                 adata,
                 condition_key: str = None,
                 cell_type_keys: str = None,
                 batch_size: int = 128,
                 alpha_epoch_anneal: int = None,
                 alpha_kl: float = 1.,
                 use_early_stopping: bool = True,
                 reload_best: bool = True,
                 early_stopping_kwargs: dict = None,
                 **kwargs):

        self.adata = adata
        self.model = model
        self.condition_key = condition_key
        self.cell_type_keys = cell_type_keys

        self.batch_size = batch_size
        self.alpha_epoch_anneal = alpha_epoch_anneal
        self.alpha_iter_anneal = kwargs.pop("alpha_iter_anneal", None)
        self.use_early_stopping = use_early_stopping
        self.reload_best = reload_best

        self.alpha_kl = alpha_kl

        early_stopping_kwargs = (early_stopping_kwargs if early_stopping_kwargs else dict())

        self.n_samples = kwargs.pop("n_samples", None)
        self.train_frac = kwargs.pop("train_frac", 0.9)
        self.use_stratified_sampling = kwargs.pop("use_stratified_sampling", True)

        self.weight_decay = kwargs.pop("weight_decay", 0.04)
        self.clip_value = kwargs.pop("clip_value", 0.0)

        self.n_workers = kwargs.pop("n_workers", 0)
        self.seed = kwargs.pop("seed", 2020)
        self.monitor = kwargs.pop("monitor", True)
        self.monitor_only_val = kwargs.pop("monitor_only_val", True)

        self.early_stopping = EarlyStopping(**early_stopping_kwargs)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        torch.manual_seed(self.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            self.model.cuda()
            logging.info("GPU available: True, GPU used: True")
        else:
            logging.info("GPU available: False")

        self.epoch = -1
        self.n_epochs = None
        self.iter = 0
        self.best_epoch = None
        self.best_state_dict = None
        self.current_loss = None
        self.previous_loss_was_nan = False
        self.nan_counter = 0
        self.optimizer = None
        self.training_time = 0

        self.train_data = None
        self.valid_data = None
        self.sampler = None
        self.dataloader_train = None
        self.dataloader_valid = None

        self.iters_per_epoch = None
        self.val_iters_per_epoch = None

        self.logs = defaultdict(list)

        # Create Train/Valid AnnotatetDataset objects
        self.train_data, self.valid_data = make_tcr_dataset(
            self.adata,
            train_frac=self.train_frac,
            condition_key=self.condition_key,
            cell_type_keys=self.cell_type_keys,
            condition_encoder=self.model.condition_encoder,
            cell_type_encoder=self.model.cell_type_encoder,
        )

    def initialize_loaders(self):
        """
        Initializes Train-/Test Data and Dataloaders with custom_collate and WeightedRandomSampler for Trainloader.
        Returns:

        """
        if self.n_samples is None or self.n_samples > len(self.train_data):
            self.n_samples = len(self.train_data)
        self.iters_per_epoch = int(np.ceil(self.n_samples / self.batch_size))

        if self.use_stratified_sampling:
            # Create Sampler and Dataloaders
            stratifier_weights = torch.tensor(self.train_data.stratifier_weights, device=self.device)

            self.sampler = WeightedRandomSampler(stratifier_weights,
                                                 num_samples=self.n_samples,
                                                 replacement=True)
            self.dataloader_train = torch.utils.data.DataLoader(dataset=self.train_data,
                                                                batch_size=self.batch_size,
                                                                sampler=self.sampler,
                                                                collate_fn=custom_collate,
                                                                num_workers=self.n_workers)
        else:
            self.dataloader_train = torch.utils.data.DataLoader(dataset=self.train_data,
                                                                batch_size=self.batch_size,
                                                                shuffle=True,
                                                                collate_fn=custom_collate,
                                                                num_workers=self.n_workers)
        if self.valid_data is not None:
            val_batch_size = self.batch_size
            if self.batch_size > len(self.valid_data):
                val_batch_size = len(self.valid_data)
            self.val_iters_per_epoch = int(np.ceil(len(self.valid_data) / self.batch_size))
            self.dataloader_valid = torch.utils.data.DataLoader(dataset=self.valid_data,
                                                                batch_size=val_batch_size,
                                                                shuffle=True,
                                                                collate_fn=custom_collate,
                                                                num_workers=self.n_workers)

    def calc_alpha_coeff(self):
        """Calculates current alpha coefficient for alpha annealing.

           Parameters
           ----------

           Returns
           -------
           Current annealed alpha value
        """
        if self.alpha_epoch_anneal is not None:
            alpha_coeff = min(self.alpha_kl * self.epoch / self.alpha_epoch_anneal, self.alpha_kl)
        elif self.alpha_iter_anneal is not None:
            alpha_coeff = min((self.alpha_kl * (self.epoch * self.iters_per_epoch + self.iter) / self.alpha_iter_anneal), self.alpha_kl)
        else:
            alpha_coeff = self.alpha_kl
        return alpha_coeff

    def train(self,
              n_epochs=400,
              lr=1e-3,
              eps=0.01):

        self.initialize_loaders()
        begin = time.time()
        
        self.model.train()
        self.n_epochs = n_epochs

        # filter the masked paramters
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, eps=eps, weight_decay=self.weight_decay)

        self.before_loop()
        for self.epoch in range(n_epochs):
            self.on_epoch_begin(lr, eps)
            self.iter_logs = defaultdict(list)

            for self.iter, batch_data in enumerate(self.dataloader_train):
                for key, batch in batch_data.items():
                    batch_data[key] = batch.to(self.device)

                # Loss Calculation
                self.on_iteration(batch_data)

            # Validation of Model, Monitoring, Early Stopping
            self.on_epoch_end()
            if self.use_early_stopping:
                if not self.check_early_stop():
                    break

        if self.best_state_dict is not None and self.reload_best:
            print("Saving best state of network...")
            print("Best State was in Epoch", self.best_epoch)
            self.model.load_state_dict(self.best_state_dict)

        self.model.eval()
        self.after_loop()

        self.training_time += (time.time() - begin)

    def before_loop(self):
        pass

    def on_epoch_begin(self, lr, eps):
        pass

    def after_loop(self):
        pass

    def on_iteration(self, batch_data):
        
        # Dont update any weight on first layers except condition weights
        if self.model.freeze:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.BatchNorm1d):
                    if not module.weight.requires_grad:
                        module.affine = False
                        module.track_running_stats = False

        # Calculate Loss depending on Trainer/Model
        self.current_loss = loss = self.loss(batch_data)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        if self.clip_value > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)

        self.optimizer.step()

    def on_epoch_end(self):
        # Get Train Epoch Logs
        for key in self.iter_logs:
            self.logs["epoch_" + key].append(np.array(self.iter_logs[key]).mean())

        # Validate Model
        if self.valid_data is not None:
            self.validate()

        # Monitor Logs
        if self.monitor:
            print_progress(self.epoch, self.logs, self.n_epochs, self.monitor_only_val)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.iter_logs = defaultdict(list)
        # Calculate Validation Losses
        for val_iter, batch_data in enumerate(self.dataloader_valid):
            for key, batch in batch_data.items():
                batch_data[key] = batch.to(self.device)

            val_loss = self.loss(batch_data)

        # Get Validation Logs
        for key in self.iter_logs:
            self.logs["val_" + key].append(np.array(self.iter_logs[key]).mean())

        self.model.train()

    def check_early_stop(self):
        # Calculate Early Stopping and best state
        early_stopping_metric = self.early_stopping.early_stopping_metric
        if self.early_stopping.update_state(self.logs[early_stopping_metric][-1]):
            self.best_state_dict = self.model.state_dict()
            self.best_epoch = self.epoch

        continue_training, update_lr = self.early_stopping.step(self.logs[early_stopping_metric][-1])
        if update_lr:
            print(f'\nADJUSTED LR')
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.early_stopping.lr_factor

        return continue_training
    
'''class tcrVAETrainer(tcrTrainer):
    """ScArches Unsupervised Trainer class. This class contains the implementation of the unsupervised CVAE/TRVAE
       Trainer.

           Parameters
           ----------
           model: trVAE
                Number of input features (i.e. gene in case of scRNA-seq).
           adata: : `~anndata.AnnData`
                Annotated data matrix. Has to be count data for 'nb' and 'zinb' loss and normalized log transformed data
                for 'mse' loss.
           condition_key: String
                column name of conditions in `adata.obs` data frame.
           cell_type_key: String
                column name of celltypes in `adata.obs` data frame.
           train_frac: Float
                Defines the fraction of data that is used for training and data that is used for validation.
           batch_size: Integer
                Defines the batch size that is used during each Iteration
           n_samples: Integer or None
                Defines how many samples are being used during each epoch. This should only be used if hardware resources
                are limited.
           clip_value: Float
                If the value is greater than 0, all gradients with an higher value will be clipped during training.
           weight decay: Float
                Defines the scaling factor for weight decay in the Adam optimizer.
           alpha_iter_anneal: Integer or None
                If not 'None', the KL Loss scaling factor will be annealed from 0 to 1 every iteration until the input
                integer is reached.
           alpha_epoch_anneal: Integer or None
                If not 'None', the KL Loss scaling factor will be annealed from 0 to 1 every epoch until the input
                integer is reached.
           use_early_stopping: Boolean
                If 'True' the EarlyStopping class is being used for training to prevent overfitting.
           early_stopping_kwargs: Dict
                Passes custom Earlystopping parameters.
           use_stratified_sampling: Boolean
                If 'True', the sampler tries to load equally distributed batches concerning the conditions in every
                iteration.
           use_stratified_split: Boolean
                If `True`, the train and validation data will be constructed in such a way that both have same distribution
                of conditions in the data.
           monitor: Boolean
                If `True', the progress of the training will be printed after each epoch.
           n_workers: Integer
                Passes the 'n_workers' parameter for the torch.utils.data.DataLoader class.
           seed: Integer
                Define a specific random seed to get reproducable results.
        """
    def __init__(
            self,
            model,
            adata,
            **kwargs
    ):
        super().__init__(model, adata, **kwargs)

    def loss(self, total_batch=None):
        recon_loss, kl_loss, mmd_loss = self.model(**total_batch)
        loss = recon_loss + self.calc_alpha_coeff()*kl_loss + mmd_loss
        self.iter_logs["loss"].append(loss.item())
        self.iter_logs["unweighted_loss"].append(recon_loss.item() + kl_loss.item() + mmd_loss.item())
        self.iter_logs["recon_loss"].append(recon_loss.item())
        self.iter_logs["kl_loss"].append(kl_loss.item())
        if self.model.use_mmd:
            self.iter_logs["mmd_loss"].append(mmd_loss.item())
        return loss'''