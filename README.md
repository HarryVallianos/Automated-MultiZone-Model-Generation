# AutomatedMultiZoneModelGeneration
This repository provides the scripts to automatically develop and generate multi-zone thermal model for residential buildings from measured data. The structure of the model is defined in an iterative way, using both forward and backward selection procedures. The repository also includes an example dataset of measured data from a residential building equiped with 9 individually controlled electric thermostats. The methodology creates a folder named 'Results' and further subfolders where it saves and plots the metrics and the progress of the procedure, as long as the final results.

The code was tested using python 3.11.6, pandas 2.1.3, numpy 1.26.0, plotly 5.18.0, sklearn 1.3.2, scipy 1.11.3, and numba 0.58.1.

For more information please refer to our article:
>   Charalampos Vallianos, Andreas Athienitis, Benoit Delcroix, Automatic generation of multi-zone RC models using smart thermostat data from homes, Energy and Buildings, Volume 277, 2022, 112571, ISSN 0378-7788, https://doi.org/10.1016/j.enbuild.2022.112571

## Repository structure
* data_in
  * house_data.csv : Includes the measured data of 9 thermostats of a residential building. Includes temperature and heating output of each electric baseboard associated with each thermostat. Also includes the solar radiation and exterior temperature.

* automated_multi_zone_generation.py : Python file that includes the main model development procedure.
* calibrate.py : Python file that includes functions to create the model equations from a parameter dataframe and calibrate the parameters using measured data.
* plots_and_metrics.py : Python file that includes functions to evaluate the model, plot some of the results and save them to the 'Results' folder.
* README

## Citation

If you use the data of this repository please reference the software on zenodo:
>TBD

and our article:
>   Charalampos Vallianos, Andreas Athienitis, Benoit Delcroix, Automatic generation of multi-zone RC models using smart thermostat data from homes, Energy and Buildings, Volume 277, 2022, 112571, ISSN 0378-7788, https://doi.org/10.1016/j.enbuild.2022.112571.

## Publications using data form this repository

1. Charalampos Vallianos, Matin Abtahi, Andreas Athienitis, Benoit Delcroix & Luis Rueda (2023) Online model-based predictive control with smart thermostats: application to an experimental house in Québec, Journal of Building Performance Simulation, DOI: 10.1080/19401493.2023.2243602 
2. Charalampos Vallianos, Andreas Athienitis, Benoit Delcroix, Automatic generation of multi-zone RC models using smart thermostat data from homes, Energy and Buildings, Volume 277, 2022, 112571, ISSN 0378-7788, https://doi.org/10.1016/j.enbuild.2022.112571
3. Vallianos, C. et al. (2023). Automated RC Model Generation for MPC Applications to Energy Flexibility Studies in Quebec Houses. In: Wang, L.L., et al. Proceedings of the 5th International Conference on Building Energy and Environment. COBEE 2022. Environmental Science and Engineering. Springer, Singapore. https://doi.org/10.1007/978-981-19-9822-5_73