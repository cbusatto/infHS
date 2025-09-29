# infHS
Informative Horseshoe regression

This package implement the methods introduced in Busatto and van de Wiel (2023), where the authors discuss the inclusion of external information in the regression model.

A tutorial can be found in file infHS_example.R.

# Original datasets:
- See Lappalainen et al. (2013) for more details on the p38MAPK dataset
- full methylation dataset can be found at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE99511

# References
Busatto, Claudio and van de Wiel, Mark (2023). "Informative co-data learning for high-dimensional Horseshoe regression". Arxiv.

Lappalainen, Tuuli and Sammeth, Michael and Friedländer, Marc and al. (2013). "Transcriptome and genome sequencing uncovers functional variation in humans". Nature, 501.


# STRUCTURE of infHS_pkg:

- /infHS: contains the R package for infHS methods

- /real_case_studies: contains the data for the two real dataset and the R files for reproducing the results of Section 6, in particular:
                      - subfolder /SNP contains: - P38MAPKpathwayKb100K.RData (data of case study 1)
                                                 - real_data_SNP.R: run all models for case study 1
                 
                      - subfolder /methylation contains: - methylation_data.RData (data of case study 2)
                                                         - meth_run_infhs.R: run infHS for case study 2
                                                         - meth_run_lasso.R: run LASSO and ridge for case study 2 

- /simulations: contains R files for reproducing the results of the simulations, in particular:
                - simulation_competitors_corr.R: reproduces the results of Tables 1 and 2 of the main paper for n = 50 in the independent scenario (for cluster and scale-free the huge package must be used)
                - simulations_codata_sensitivity.R: reproduces the results of Table 3 of the Supplementary Materials for n = 50