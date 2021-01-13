import random
import os
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch_geometric.utils import remove_self_loops
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# binary class
class FocalLoss(nn.Module):
    # nn.Module: base class for all neural network modules.

    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        # super supports cooperative multiple inheritance in a dynamic execution environment.
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        # input:size is M*2. M　is the batch　number
        # target:size is M.
        target = target.float()
        pt = torch.softmax(input, dim=1)
        # Applies the Softmax to a n-dimensional input Tensor rescaling the elements so that they lie in the range
        # [0,1] and sum to 1 (dim=1 -> it normalizes values along axis 1, the columns(?))
        p = pt[:, 1]
        loss = -self.alpha * (1 - p) ** self.gamma * (target * torch.log(p)) - \
               (1 - self.alpha) * p ** self.gamma * ((1 - target) * torch.log(1 - p))
        return loss.mean()


class Option(object):
    def __init__(self, d):
        self.__dict__ = d


def seed_set(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def onehot_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element. It will indeed be applied to convert the data
    related to the atom species, and the last element of the set is simply 'other atoms" """
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def split_multi_label_containNan(df, tasks, seed):
    """
    df is obtained when a comma-separated values (csv) file is returned as two-dimensional data structure with
    labeled axes. This function reads a file and randomly divides the data for training, validation and testing
    (separately for positive and negative samples). Than, again for each task, it produces a weight related to each t
    ask depending on the ratio betweeen positive and negative training samples. The final results are a list of weights and
    a concatenation (for each task) of training\validation\testing samples.
    """
    weights = []
    random_seed = seed
    for i, task in enumerate(tasks):
        negative_df = df[df[task] == 0][["smiles", task]]
        positive_df = df[df[task] == 1][["smiles", task]]
        negative_test = negative_df.sample(frac=1 / 10, random_state=random_seed)
        # Return a random sample of items from an axis of object. frac= Fraction of axis items to return.
        negative_valid = negative_df.drop(negative_test.index).sample(frac=1 / 9, random_state=random_seed)
        # Takes negative_df and drops what has been extracted for negative_test. Than it randomly samples what remains.
        negative_train = negative_df.drop(negative_test.index).drop(negative_valid.index)
        # What remains is used for the training

        positive_test = positive_df.sample(frac=1 / 10, random_state=random_seed)
        positive_valid = positive_df.drop(positive_test.index).sample(frac=1 / 9, random_state=random_seed)
        positive_train = positive_df.drop(positive_test.index).drop(positive_valid.index)

        weights.append([(positive_train.shape[0] + negative_train.shape[0]) / negative_train.shape[0], \
                        (positive_train.shape[0] + negative_train.shape[0]) / positive_train.shape[0]])
        train_df_new = pd.concat([negative_train, positive_train])
        valid_df_new = pd.concat([negative_valid, positive_valid])
        test_df_new = pd.concat([negative_test, positive_test])

        if i == 0:
            train_df = train_df_new
            test_df = test_df_new
            valid_df = valid_df_new
        else:
            train_df = pd.merge(train_df, train_df_new, on='smiles', how='outer')
            test_df = pd.merge(test_df, test_df_new, on='smiles', how='outer')
            valid_df = pd.merge(valid_df, valid_df_new, on='smiles', how='outer')
    return train_df, valid_df, test_df, weights

# copy from xiong et al. attentivefp
class ScaffoldGenerator(object):
    """
    Generate molecular scaffolds.

    Parameters
    ----------
    include_chirality : : bool, optional (default False)
        Include chirality in scaffolds.
    """

    def __init__(self, include_chirality=False):
        self.include_chirality = include_chirality

    def get_scaffold(self, mol):
        """
        Get Murcko scaffolds for molecules.

        Murcko scaffolds are described in DOI: 10.1021/jm9602928. They are
        essentially that part of the molecule consisting of rings and the
        linker atoms between them.

        Parameters
        ----------
        mols : array_like
            Molecules.
        """
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=self.include_chirality)

# copy from xiong et al. attentivefp
def generate_scaffold(smiles, include_chirality=False):
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    engine = ScaffoldGenerator(include_chirality=include_chirality)
    scaffold = engine.get_scaffold(mol)
    return scaffold

# copy from xiong et al. attentivefp
def split(scaffolds_dict, smiles_tasks_df, tasks, weights, sample_size, random_seed=0):
    count = 0
    minor_count = 0
    minor_class = np.argmax(weights[0])  # weights are inverse of the ratio
    minor_ratio = 1 / weights[0][minor_class]
    optimal_count = 0.1 * len(smiles_tasks_df)
    while (count < optimal_count * 0.9 or count > optimal_count * 1.1) \
            or (minor_count < minor_ratio * optimal_count * 0.9 \
                or minor_count > minor_ratio * optimal_count * 1.1):
        random_seed += 1
        random.seed(random_seed)
        scaffold = random.sample(list(scaffolds_dict.keys()), sample_size)
        count = sum([len(scaffolds_dict[scaffold]) for scaffold in scaffold])
        index = [index for scaffold in scaffold for index in scaffolds_dict[scaffold]]
        minor_count = len(smiles_tasks_df.iloc[index, :][smiles_tasks_df[tasks[0]] == minor_class])
    #     print(random)
    return scaffold, index

# copy from xiong et al. attentivefp
def scaffold_randomized_spliting(smiles_tasks_df, tasks=['HIV_active'], random_seed=8):
    """ For each molecule a scaffold is generated and saved (its index is separately saved too). Next they are split for
    training, testing and validation"""
    weights = []
    for i, task in enumerate(tasks):
        negative_df = smiles_tasks_df[smiles_tasks_df[task] == 0][["smiles", task]]
        positive_df = smiles_tasks_df[smiles_tasks_df[task] == 1][["smiles", task]]
        weights.append([(positive_df.shape[0] + negative_df.shape[0]) / negative_df.shape[0], \
                        (positive_df.shape[0] + negative_df.shape[0]) / positive_df.shape[0]])
    print('The dataset weights are', weights)
    print('generating scaffold......')
    scaffold_list = []
    all_scaffolds_dict = {}
    # It will contain the index of all the scaffolds created (for each task)
    for index, smiles in enumerate(smiles_tasks_df['smiles']):
        scaffold = generate_scaffold(smiles)
        scaffold_list.append(scaffold)
        if scaffold not in all_scaffolds_dict:
            all_scaffolds_dict[scaffold] = [index]
        else:
            all_scaffolds_dict[scaffold].append(index)
            #!!! smiles_tasks_df['scaffold'] = scaffold_list

    samples_size = int(len(all_scaffolds_dict.keys()) * 0.1)
    test_scaffold, test_index = split(all_scaffolds_dict, smiles_tasks_df, tasks, weights, samples_size,
                                      random_seed=random_seed)
    training_scaffolds_dict = {x: all_scaffolds_dict[x] for x in all_scaffolds_dict.keys() if x not in test_scaffold}
    valid_scaffold, valid_index = split(training_scaffolds_dict, smiles_tasks_df, tasks, weights, samples_size,
                                        random_seed=random_seed)

    training_scaffolds_dict = {x: training_scaffolds_dict[x] for x in training_scaffolds_dict.keys() if
                               x not in valid_scaffold}
    train_index = []
    for ele in training_scaffolds_dict.values():
        train_index += ele
    assert len(train_index) + len(valid_index) + len(test_index) == len(smiles_tasks_df)

    return train_index, valid_index, test_index, weights
