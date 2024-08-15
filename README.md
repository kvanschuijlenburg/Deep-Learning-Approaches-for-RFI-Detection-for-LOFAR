# Deep Learning Approaches for Radio Frequency Interference Detection for LOFAR 📡

Welcome to the repository for our project on **Deep Learning Approaches for Radio Frequency Interference Detection for LOFAR**. This repository contains all the necessary code to reproduce all experiments, figures, tables, and statistical tests from our thesis.

## Reproducing the Thesis 📚

We have included a script, `reproduceThesis.py`, which performs all steps in the same order as the sections in the thesis. The results of these steps are stored in the `./plots` directory.

With this script, all experiments can be repeated, including training the deep learning models and performing k-means clustering searches. Please note that several tasks may take hours or even days to complete.

## Saving and Loading Intermediate Results 💾

Our software saves intermediate results in the `./models` directory. This includes, for example, model parameters, embeddings, k-means search results, and cluster labels. These results are automatically loaded when available, saving time and computational resources.

To ensure consistency with our results, we recommend downloading the model data. This will prevent any slight variations in results due to the nondeterministic nature of some algorithms.

## Requirements 💻

### Required
- Memory: 32 GB
- Storage: 45 GB

### Optional
- GPU with 12 GB memory for training RFI-GAN and DiMoGAN models
- GPU with 45 GB memory for training DINO models
- Measurement sets: The first 2400 seconds of subbands L2014581_SAP000_SB052 - L2014581_SAP000_SB081 from LOFAR observation L2014581 for training on the entire dataset

# Setup 🛠️

## Recommended Steps
- Install the dependencies listed in `requirements.txt`.
- Download the `./data` and `./models` directories from [here](https://1drv.ms/f/s!AoUdhf01k9Ri1bM3MApmSSpBw_e7sw?e=UTfQjh) this link and place them in `./data/` and `./models/`, respectively.
- **Optional**: The default paths for the data, models, and plots are `./data`, `./models`, and `./plots`, respectively. If you wish to change these paths, you can do so at the first lines after the imports in `./utils/functions.py`.

## Optional Steps

### Flagging MS Set and Converting it to H5
Reproducing the results of this thesis can be done by loading the cached data in the Setup. However, the DINO model can only be trained on the entire dataset. Therefore, the original dataset is required. This requires a measurement set as specified in the requirements, and the following actions:

- Install `casacore` and `python`:
  - `sudo apt-get install casacore-dev`
  - `pip install python-casacore`
- Download and install AOFlagger from [here](https://gitlab.com/aroffringa/aoflagger/).
- Flag the `.ms` files with AOFlagger.
- Save the flagged `.ms` files in `./data/LOFAR_L2014581 (recording)/ms`.
- Run the file `datasets/ms_to_h5.py`. The `h5` files are saved at `./data/LOFAR_L2014581 (recording)/h5`.


# Running the Experiments

To run the experiments, simply execute `reproduceThesis.py`.

Please note that the last step involves measuring the inference time. This might take a while, but since it is the last step, the script can be terminated if needed.

By default, training the RFI-GAN, DINO, and DiMoGAN models is disabled due to the long training times. If you wish to train the deep learning models, set `trainRfiGan`, `trainDinoSearch`, and `trainDiMoGan` in `reproduceThesis.py` to `True`.

Each experiment is selected by an experiment number:
- The DINO hyperparameter search with a ViT as the backbone corresponds to Table 5 in the Thesis.
- The DINO hyperparameter search with a CNN as the backbone corresponds to Table 6 in the Thesis, but offset by 100.
- The RFI-GAN and DiMoGAN experiments can be found at the top of the `reproduceThesis.py` file.

## Limitations
Without the original `.ms` sets:
- Visualization of the dataset over all subbands (Figure 3.1) cannot be made.
- RFI percentage per subband  (Figure 3.3) cannot be plotted.
- DINO ViT experiments cannot be trained.
- DINO experiments 31 and 32 cannot be trained.

# Acknowledgements 🙏
We would like to acknowledge the following resources that have been instrumental in our project:
- [hera_sim](https://github.com/HERA-Team/hera_sim) (Custom License)
- [HIDE](https://github.com/cosmo-ethz/hide) (GPL-3.0 License)
- [DINO](https://github.com/facebookresearch/dino) (Apache-2.0 License)
- [DINO TensorFlow 2 Implementation](https://github.com/TanyaChutani/DINO_Tf2.x)
- [RFI-GAN](https://github.com/lizhen-3/RFI-GAN/)
- [AOFlagger](https://gitlab.com/aroffringa/aoflagger/) (GNU GPLv3 License)
- [Example code from scikit-learn](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
- [StyleGAN 2 Implementation](https://github.com/NVlabs/stylegan2) (Nvidia Source Code License-NC)