# <center> Anime Popularity Prediction: A Multimodal Approach Using Deep Learning </center>

This repository is for the results reproduction of the paper "Anime Popularity Prediction: A Multimodal Approach Using Deep Learning".


## Reproducibility instructions

<ol>
  <li>
    You can clone this repo into a local directory. For instance, if you are using Ubuntu 20.04 (as we did), you can prompt into the terminal:
    
```
git clone https://github.com/JesusASmx/Popularity-Prediction-in-Anime-with-Deep-Learning.git YourFolder
```
  </li>
  <li>
    Once clonned, unzip your Database.zip file inside YourFolder/Database. The result must looks like this (unzipped files highlighted in red):

```diff
./YourFolder
-├─Database
-│   ├─char test
-│   │ └─(2,821 .png files)
-│   ├─char train
-│   │ └─(11,861 .png files)
-│   ├─Anime_test.csv
-│   ├─Anime_train.csv
-│   ├─Char_test.csv
-│   └─Char_train.csv
 ├─Final Results
 │   ├─ETC...
 ├─Ca Naxca.py
 ETC...
```
  </li>
  <li>
    Create a virtual enviroment. To install the required ubuntu package you can run in terminal:

```bash
  apt-get install python3-venv
```

And finally:

```python
>>> python3 venv anime_virtual_enviroment
>>> source anime_virtual_enviroment/bin/activate
```

We personally created the virtual enviroment in our equivalent of ```./YourFolder```, so we encourage to follow the same practice.
  </li>
  <li>

Inside ```anime_virtual_enviroment```, install the follow packages:
    
```python
>>> pip install pandas 
>>> pip install numpy 
>>> pip install Pillow
>>> pip install tqdm #only if not already installed
>>> pip install scikit-learn
>>> pip install matplotlib
    
>>> pip install torch==1.10.1
>>> pip install transformers
```
  </li>
  <li>
    Finally, this is the list of script with respect of each experiment:
    <ul>
      <li>all_inputs.py for the experiment with all inputs. </li>
      <li>char_inputs.py for the experiment who only considers main character descriptions and portraits as inputs.</li>
      <li>img_inputs.py for the experiment who only considers main character portraits as inputs.</li>
      <li>syn_inputs.py for the experiment who only considers anime synopsis as inputs.</li>
      <li>trad_inputs.py for the experiment with all inputs but using TF-IDF and PILtotensor vectorizations.</li>
    </ul>
  </li>
</ol>

The outputs to plot the learning curves are stored inside ```./YourFolder/Final Results/<EXPERIMENT NAME>/```. reporte_full.csv contains the metrics of the experiment (e.g. MSE and correlation coeficients), while train_val_loss.json contains the train-validation loss for each epoch of the experiment in the follow list format: ```list[i] == [train loss in epoch i, validation loss in epoch i]```. From this, it is possible to reconstruct the learning curves by running ```lcurves.ipynb``` (in our case, not available in the ubuntu server).


IMPORTANT NOTE: When the ```.ipynb``` files share name with a ```.py``` files, they are the same script but in the Jupyter Notebook format. If you have access to a powerfull GPU supporting JNB, feel free to try running the experiments from these files instead.


### Hardware Specifications

The experiments were performed with the follow hardware:
<ul>
    <li>Graphic card: NVIDIA Quadro RTX 6000/8000</li>
    <li>Processor: Intel Xeon E3-1200</li>
    <li>RAM: 62 gb</li>
    <li>VRAM: 46 gb</li>
</ul>


### Software Specifications

The employed software was the follow:
<ul>
    <li>CUDA  V10.1.243</li>
    <li>OS: Ubuntu Server 20.04.3 LTS</li>
    <li>Vim version 8.1</li>
    <li>Python version: 3.9.5</li>
</ul>

### English language codebook

Here is a list of relevant translations and explanations of the variables and comments employed across the code:

*Ca_Naxca.py
<ul>
  <li>Line 11:</li>
  Original (Spanish): #IDs es lista t.q. IDs[x] es el ID del sample con label y_test[x] y predicción y_pred[x]
  
  Translated: #IDs is a list such that IDs[x] is the ID of the sample with label y_test[x] and prediction y_pred[x]  
</ul>

*The rest of the code:
<ul>
  <li>"Personaje" means "character" in Spanish.</li>
  <li>"Retratos" means "portraits" in Spanish.</li>
</ul>

### How to cite this material

For this repository only, you should employ this DOI: 10.5281/zenodo.13824604

For this repository plus the dataset, you should employ this DOI: 10.5281/zenodo.13835115
