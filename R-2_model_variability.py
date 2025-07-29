# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nibabel as nib
from scipy.stats import pearsonr
from utils import index_structure, compute_dfc_var

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                           LOAD DATA
# #################################################################
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

for index, row in pairs_available.iterrows():
    ligand = genes[row['Peptide']].values[:, np.newaxis]
    receptor = genes[row['Receptor']].values[:, np.newaxis]
    
    # make sure ligand and receptor have no zero values
    # if so, add minimal value to avoid division by zero
    if np.any(ligand == 0):
        ligand[np.where(ligand == 0)] = 1e-10
    if np.any(receptor == 0):
        receptor[np.where(receptor == 0)] = 1e-10
    
    mat = ligand @ receptor.T
    all_mat.append(mat)

all_mat = np.array(all_mat)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#          COMPARE WITH S-F COUPLING WITH DIFF STABILITY
# #################################################################

sf_coupling = pd.read_csv('results/comm_pred_peptide_ligand_pairs.csv', index_col=0).iloc[:, 1:]
gene_qc = pd.read_csv('data/gene_qc.csv', index_col=0)

# differential stability of each receptor in receptor ligand pairs
for l,r in zip(pairs_available['Peptide'], pairs_available['Receptor']):  
    index = pairs_available[(pairs_available['Peptide'] == l) & (pairs_available['Receptor'] == r)].index[0]
    pairs_available.loc[index, 'diff_stability'] = gene_qc.loc[r, 'diff_stability']

sf_coupling_rank = sf_coupling.rank().copy()

# compare diff_stability of ligand-receptor pairs with sf_coupling (rank)
fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
sns.regplot(x=pairs_available['diff_stability'], 
            y=np.mean(sf_coupling_rank.values, axis=1),
            color='grey', ci=None, ax=ax)
r, p = pearsonr(pairs_available['diff_stability'], np.mean(sf_coupling_rank.values, axis=1))

plt.xlabel('Differential stability\n' + 'more variable ⬌ less variable')
plt.ylabel('Mean S-F coupling (rank)')
plt.ylim(10, 30)
plt.yticks([10, 15, 20, 25, 30], ['10', '15', '20', '25', '30'])
plt.text(0.05, 0.98,  f'r ={r:.2f}, \nP = {p:.4f}', ha='left', va='top', transform=plt.gca().transAxes)
sns.despine()
# plt.savefig('figs/sf_coupling_vs_diff_stability.pdf')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#       COMPARE WITH S-F COUPLING WITH FC VARIABILITY
# #################################################################

# load the data
data_dir = '/poolz2/nnl-data/processed-data/HCP-YA-release/HCP_1200-rfMRI-deriv/schaefer400x7/'
files = [f for f in os.listdir(data_dir) if f.endswith('.ptseries.nii')]
files.sort()

# check if connectivity variance is in the data
dfc_var_fn = 'data/template_parc-Schaefer400_desc-FC_variability.npy'

if os.path.exists(dfc_var_fn):
    dfc_var = np.load(dfc_var_fn)
    
else:
    subjects = set()
    for f in files:
        subject = f.split('_')[0]
        if subject not in subjects:
            subjects.add(subject)
    subjects = sorted(list(subjects))
    all_dfc_var = []
    for subject in subjects:
        subject_files = [f for f in files if f.startswith(subject)]
        subject_files.sort()
        
        session_dfc_var = []
        for f in subject_files:
            img = nib.load(os.path.join(data_dir, f))
            tseries = img.get_fdata()
            dfc_var = compute_dfc_var(tseries, window_size=60, step_size=30)
            session_dfc_var.append(dfc_var)
        
        session_dfc_var = np.array(session_dfc_var)
        subject_dfc_var = np.mean(session_dfc_var, axis=0)
        all_dfc_var.append(subject_dfc_var)
        
    all_dfc_var = np.array(all_dfc_var)
    dfc_var = np.mean(all_dfc_var, axis=0)

    np.save('data/template_parc-Schaefer400_desc-FC_variability.npy', all_dfc_var)

all_r = []
for i in range(all_mat.shape[0]):
    mat = all_mat[i]
    r, p = pearsonr(dfc_var.flatten(), mat.flatten())
    all_r.append(r)

# compare diff_stability of ligand-receptor pairs with sf_coupling (rank)
fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
sns.regplot(x=all_r, 
            y=np.mean(sf_coupling_rank.values, axis=1),
            color='grey', ci=None, ax=ax)
r, p = pearsonr(all_r, np.mean(sf_coupling_rank.values, axis=1))
plt.xlabel('Correlation with connectivity variance\n' + 'less variable ⬌ less variable')
plt.ylabel('Mean S-F coupling (rank)')
plt.ylim(10, 30)
plt.yticks([10, 15, 20, 25, 30], ['10', '15', '20', '25', '30'])
plt.text(0.75, 0.98,  f'r ={r:.2f}, \nP = {p:.4f}', ha='left', va='top', transform=plt.gca().transAxes)
sns.despine()
# plt.savefig('figs/sf_coupling_vs_connectivity_variance.pdf')