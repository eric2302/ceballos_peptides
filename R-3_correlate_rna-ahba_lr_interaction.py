# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from plot_utils import divergent_green_orange
from utils import index_structure

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                            LOAD DATA
###############################################################################
# load rna data
rna = pd.read_csv('results/abagen_rnaseq_interpolation_normalized.csv')
microarray = pd.read_csv('data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
rna.index = microarray.index

# keep only overlapping genes in rna and microarray
rna = rna[rna.columns.intersection(microarray.columns)]
microarray = microarray[microarray.columns.intersection(rna.columns)]

# load ligand-receptor pairs
pairs_available = pd.read_csv('data/peptide_receptor_ligand_pairs.csv', index_col=0)
receptor_genes = pd.read_csv('data/receptor_filtered.csv', index_col=0).index.to_list()
precursor_genes = pd.read_csv('data/precursor_qc.csv', index_col=0).index.to_list()
genes_of_interest = receptor_genes + precursor_genes

pairs_available = pairs_available[pairs_available['Peptide'].isin(genes_of_interest)]
pairs_available = pairs_available[pairs_available['Receptor'].isin(genes_of_interest)]
pairs_available.sort_values('Receptor', inplace=True, ignore_index=True)
pairs_available = pairs_available[['Receptor', 'Peptide']].reset_index(drop=True)


rna = rna[rna.columns.intersection(genes_of_interest)]
microarray = microarray[microarray.columns.intersection(genes_of_interest)]

rna = index_structure(rna, structure='CTX')
microarray = index_structure(microarray, structure='CTX')

# %%

all_r = []
fig, axs = plt.subplots(7, 5, figsize=(10, 14), dpi=200)
for (index, row), ax  in zip(pairs_available.iterrows(), axs.flatten()):
    ligand_rna = rna[row['Peptide']].values[:, np.newaxis]
    receptor_rna = rna[row['Receptor']].values[:, np.newaxis]
    
    ligand_microarray = microarray[row['Peptide']].values[:, np.newaxis]
    receptor_microarray = microarray[row['Receptor']].values[:, np.newaxis]
    
    mat_rna = ligand_rna @ receptor_rna.T
    mat_microarray = ligand_microarray @ receptor_microarray.T
    
    # calculate correlation
    r, p = pearsonr(mat_rna.flatten(), mat_microarray.flatten())
    all_r.append(r)
    
    # compare the two matrices
    sns.regplot(x=mat_rna.flatten(), y=mat_microarray.flatten(),
               line_kws={'color': 'black'},
               scatter_kws={'s': 1, 'alpha': 0.1},
               color='gray', ax=ax, ci=None)
    ax.set_title(f"{row['Peptide']} - {row['Receptor']}")
    ax.text(0.05, 0.95, f"r = {r:.2f}",
            transform=ax.transAxes, va='top', ha='left')
    ax.set_xlabel(f"RNAseq")
    ax.set_ylabel(f"Microarray")
    sns.despine(ax=ax)
fig.set_tight_layout(True)
fig.savefig('figs/rnaseq_microarray_interaction_similarity_individual.eps')

fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
sns.histplot(all_r, kde=True, color='gray', ax=ax)
ax.set_xlabel('Microarray-RNAseq interaction similarity (r)', fontsize=14)
sns.despine(ax=ax)
plt.savefig('figs/rnaseq_microarray_interaction_similarity_histogram.pdf')

# what's the average correlation
mean_r = np.mean(all_r)

# bootstrap to get the standard error
# do bootstrap with 1000 iterations
n_iterations = 1000
bootstrapped_r = []
for _ in range(n_iterations):
    sample_r = np.random.choice(all_r, size=len(all_r), replace=True)
    bootstrapped_r.append(np.mean(sample_r))
std_r = np.std(bootstrapped_r)


print(f"Average correlation: {mean_r:.2f} ± {std_r:.2f}")