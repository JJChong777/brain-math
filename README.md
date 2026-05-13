# AI for Neuroscience: Modelling Math Learning Difficulties

## Environment
To copy the conda environment run this on your computer
```
conda create --file environment.yml
```

## Description
THe dataset is located on my T9 External Hard Drive on my M4 Mac Mini so its under `/Volumes/T9/ds001486` just replace this line with the path your dataset is located on it should be named ds001486.

select_participants.ipynb is run on the meta data file in the dataset to find the 15 MLD patients and 15 control group matched on demographics

The entire docker command under `run_fmriprep.txt` is used to run the fmriPrep container to preprocess the data for the selected participants. You will probably need to download [Docker Desktop](https://www.docker.com/products/docker-desktop/), although I used [OrbStack](https://orbstack.dev/download) becuase it seems to have better performance on Apple Silicon.

`feature_extract.ipynb` is where the 1D correlation matrix of the neuroimaging data is extracted based on the Schaefer 100 Atlas

`final_clean_test.ipynb` is where the Support Vector Machine, Multi Layer Perceptron and XGBoost machine learning is run. The evaluation metrics and the top 10 network connections are found

`visualization.ipynb` is where the glass brain visualizations are made based on the JSON from the earlier training file. You will need to run it on your own device to see the visualizations it seems like you can't see it on GitHub.

`clean_nifti_GLM.ipynb` is where I got the 3D GLM maps from the raw neuroimages

`folds_GLM.ipynb` is where I get the folds for training the 3D CNN

`test_3d_cnn.ipynb` is where the 3D CNN is trained

The archive folder is just full of stuff I tried but didn't work its just left there for future reference

## Demo Video

The demo video is a walkthrough of the code but less details or more focused on showcasing what's inside and explaining stuff. It's [here](https://youtu.be/QXdYtzJazlI) on YouTube

## Presentation

In case you want to see it, the presentation is found [here](https://docs.google.com/presentation/d/1Z3o49S1zNsOMZi_14hafdxxDTJtxHmgbp-tuK3D0Vxo/edit?usp=sharing) on Google Slides

## Poster

[Here](https://canva.link/cd3c0oz5uqd0c2m) on Canva

`
