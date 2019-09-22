from openchem.data.utils import read_smiles_property_file, save_smiles_property_file, get_tokens
from sklearn.model_selection import train_test_split

#for log p
data = read_smiles_property_file('./benchmark_datasets/logp_dataset/logP_labels.csv', cols_to_read=[1, 2])
smiles = data[0]
labels = data[1]
X_train, X_test, y_train, y_test = train_test_split(smiles, labels, test_size=0.2, random_state=42)
save_smiles_property_file('./benchmark_datasets/logp_dataset/train.csv', X_train, y_train)
save_smiles_property_file('./benchmark_datasets/logp_dataset/test.csv', X_test, y_test)

#for tox21
# data = read_smiles_property_file('./benchmark_datasets/tox21/tox21.csv',
#                                  cols_to_read=[13] + list(range(0,12)))
# smiles = data[0]
# labels = np.array(data[1:])
# labels[np.where(labels=='')] = '999'
# labels = labels.T
# tokens, _, _ = get_tokens(smiles)
# tokens = tokens + ' '
# X_train, X_test, y_train, y_test = train_test_split(smiles, labels, test_size=0.2, random_state=42)
# save_smiles_property_file('./benchmark_datasets/tox21/train.csv', X_train, y_train)
# save_smiles_property_file('./benchmark_datasets/tox21/test.csv', X_test, y_test)
