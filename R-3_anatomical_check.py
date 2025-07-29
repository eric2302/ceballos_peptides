# %%
from math import dist
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from utils import index_structure
from neuromaps.stats import compare_images
from plot_utils import divergent_green_orange

# %%
distance = np.load('data/template_parc-Schaefer400_desc-distance.npy')
sc = np.load('data/template_parc-Schaefer400_desc-SC.npy')

# load ligand-receptor pairs
pairs_available = pd.read_csv('data/peptide_receptor_ligand_pairs.csv', index_col=0)

# load receptor and ligand from qc
receptor_genes = pd.read_csv('data/receptor_filtered.csv', index_col=0).index.to_list()
precursor_genes = pd.read_csv('data/precursor_qc.csv', index_col=0).index.to_list()
genes_of_interest = receptor_genes + precursor_genes

# keep only pairs that are in the gene library
pairs_available = pairs_available[pairs_available['Peptide'].isin(genes_of_interest)]
pairs_available = pairs_available[pairs_available['Receptor'].isin(genes_of_interest)]
pairs_available.sort_values('Receptor', inplace=True, ignore_index=True)
pairs_available = pairs_available[['Receptor', 'Peptide']].reset_index(drop=True)

# load genes of interest from all genes in gene library
all_genes = pd.read_csv('data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

# in all_genes column names, select only ones that are in genes_of_interest
genes = all_genes[all_genes.columns.intersection(genes_of_interest)]
genes = index_structure(genes, structure='CTX')

all_mat = []
all_norms = []

for index, row in pairs_available.iterrows():
    ligand = genes[row['Peptide']].values[:, np.newaxis]
    receptor = genes[row['Receptor']].values[:, np.newaxis]
    
    mat = ligand @ receptor.T
    mat = mat
    all_mat.append(mat)

all_mat = np.array(all_mat)
    
# %% do same for average of all_mat
avg_mat = np.mean(all_mat, axis=0)
avg_mat_flat = avg_mat.flatten()
mask_indices = np.where(sc.flatten() > 0)[0]
avg_mat_filtered = avg_mat_flat[mask_indices]
distance_filtered = distance.flatten()[mask_indices]

r, p = pearsonr(avg_mat_filtered, distance_filtered)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=distance_filtered, y=avg_mat_filtered, s=5, color='grey')
plt.ylabel('Interaction')
plt.xlabel('Distance')


# %%
