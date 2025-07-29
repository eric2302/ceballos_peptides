# %%
import brainconn as bc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sstats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from utils import non_diagonal_elements, fit_to_connectivity
from plot_utils import divergent_green_orange
from joblib import Parallel, delayed

savefig = False

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              LOAD DATA
###############################################################################
dist_mat = np.load("data/template_parc-Schaefer400_desc-distance.npy")
sc = np.load("data/template_parc-Schaefer400_desc-SC.npy")
precursor_nulls = np.load('data/precursor_spatial_nulls_Schaefer400_TianS4.npy')
receptor_nulls = np.load('data/receptor_spatial_nulls_Schaefer400_TianS4.npy')

# load functional connectivity
bold_fc = np.load("data/template_parc-Schaefer400_desc-FC.npy")
delta_fc = np.load("data/annotations/meg_delta_mat_Schaefer400.npy")
theta_fc = np.load("data/annotations/meg_theta_mat_Schaefer400.npy")
alpha_fc = np.load("data/annotations/meg_alpha_mat_Schaefer400.npy")
beta_fc = np.load("data/annotations/meg_beta_mat_Schaefer400.npy")
lgamma_fc = np.load("data/annotations/meg_lgamma_mat_Schaefer400.npy")
hgamma_fc = np.load("data/annotations/meg_hgamma_mat_Schaefer400.npy")
fc_mats = np.stack([bold_fc, delta_fc, theta_fc, alpha_fc, beta_fc, lgamma_fc, hgamma_fc])
fc_names = ['bold', 'delta', 'theta', 'alpha', 'beta', 'lgamma', 'hgamma']

# load peptide-receptor pairs from Zhang et al. 2021 PNAS
pairs_available = pd.read_csv('data/peptide_receptor_ligand_pairs.csv', index_col=0)

# load receptor and peptide from qc
receptor_genes = pd.read_csv('data/receptor_filtered.csv', index_col=0).index.to_list()
precursor_genes = pd.read_csv('data/precursor_qc.csv', index_col=0).index.to_list()
genes_of_interest = receptor_genes + precursor_genes

# keep only pairs that are in the gene library
pairs_available = pairs_available[pairs_available['Peptide'].isin(genes_of_interest)]
pairs_available = pairs_available[pairs_available['Receptor'].isin(genes_of_interest)]
pairs_available.sort_values('Receptor', inplace=True, ignore_index=True)
pairs_available = pairs_available[['Receptor', 'Peptide']]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                 ANNOTATE CONNECTOME WITH ALL PAIRS AVAILABLE
###############################################################################
all_rsq = []
pair_names = []
n_nulls = 1000

for index, row in pairs_available.iterrows():
    pair_name = row['Receptor'] + '-' + row['Peptide']
    print(f"Processing {pair_name}")
    pair_names.append(pair_name)
    
    peptide_idx = precursor_genes.index(row['Peptide'])
    receptor_idx = receptor_genes.index(row['Receptor'])
    
    pnulls = precursor_nulls[peptide_idx].T
    rnulls = receptor_nulls[receptor_idx].T
    
    fc_rsq = []
    for j, fc in enumerate(fc_mats):
        print(f"Processing {fc_names[j]}")
        y = non_diagonal_elements(fc)
    
        rsq = Parallel(n_jobs=64, verbose=1)(delayed(fit_to_connectivity) \
                    (pnulls[i, :400], rnulls[i, :400], sc, dist_mat, y) \
                    for i in range(n_nulls))
        rsq = np.array(rsq)
        fc_rsq.append(rsq)
        
    fc_rsq = np.array(fc_rsq)
    all_rsq.append(fc_rsq)
    
all_rsq = np.array(all_rsq)
print(all_rsq.shape)
np.save('results/sf-coupling_nulls.npy', all_rsq)

# %%
# load empirical data
emp = pd.read_csv('results/comm_pred_peptide_ligand_pairs.csv', index_col=0)

# for every peptide-receptor pair, count how many times the r2s are greater than the empirical r2
all_rsq = np.load('results/sf-coupling_nulls.npy')
n_nulls = all_rsq.shape[2]
pvals = []

for fc_idx in range(all_rsq.shape[1]):
    # load empirical data
    emp_sf = emp.iloc[:, fc_idx].values
    pvals_fc = []
    for pair_idx in range(all_rsq.shape[0]):
        pair_sf = emp_sf[pair_idx]
        null_sf = all_rsq[pair_idx, fc_idx]
    
        count = np.sum(null_sf > pair_sf)
        pval = (1 + count) / (1 + n_nulls)
        pvals_fc.append(pval)
    pvals_fc = np.array(pvals_fc)
    pvals.append(pvals_fc)

# which ones are significant?
pvals = np.array(pvals)

# from available pairs, keep only the ones that are significant
for i in range(pvals.shape[0]):
    print(f"{fc_names[i]}")
    print(emp.iloc[:, i][pvals[i] < 0.05].to_frame())
    print('---')
    print(f"Total number of significant pairs: {np.sum(pvals[i] < 0.05)}\n")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                 S-F COUPLING WITH ANNOTATED CONNECTOME
###############################################################################
# predicatbility of fc using peptide communication
all_rsq = []
y = non_diagonal_elements(fc_cons)
for comm in pep_comm:
    reg = LinearRegression(fit_intercept=True, n_jobs=-1)
    X = sstats.zscore(comm, ddof=1, axis=1).T
    reg_res = reg.fit(X, y)
    yhat = reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    num = (1 - r_squared) * (len(y) - 1)
    denom = len(y) - X.shape[1] - 1
    adjusted_r_squared = 1 -  num / denom
    all_rsq.append(adjusted_r_squared)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#               S-F COUPLING WITHOUT ANNOTATED WEIGHTS
###############################################################################

# add to pred_df
pred_df = pd.DataFrame({'Pair': pair_names, 'R2_FC': all_rsq})
pred_df.loc[len(pred_df)] = ['no_annotation', adjusted_r_squared]
pred_df = pred_df.sort_values('R2_FC', ascending=False).reset_index(drop=True)
# store pred_df
pred_df.to_csv('results/fc_pred_peptide_ligand_pairs.csv', index=False)
