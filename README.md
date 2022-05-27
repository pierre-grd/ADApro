# Advanced Data Analytics
## Spotify Sequential Skip Prediction
### Predict if users will skip or listen to the music they're streamed


## Group members
* Pierre Gérard : pierre.gerard@unil.ch
* Louis del Perugia : louis.delperugia@unil.ch

## Academic Supervisors
* Simon Scheidegger: simon.scheidegger@unil.ch
* Antoine Didisheim: Antoine.Didisheim@unil.ch
* Aleksandra Malova: aleksandra.malova@unil.ch

# Presentation of the project

## Abstract

# Project structure

```
├── README.md
│
├── requirements.txt
│
├── Dataset Description.pdf
│
├── data
│   └──log_mini.csv           
│   └──tf_mini.csv
│
├── models
│   └── model1          
│   └── model2
│   └── model3
│   └── model4
│
├── plot
│
├── src          
│   └── data_preprocessing.py
│   └── EDA.py
│   └── models.py
│   └── Neunets.py
├── main.py

```

# Install our code on your machine

1) Clone Project
```
git clone https://github.com/pierre-grd/ADApro.git
```

2) Go into project folder

```
cd ADVpro
```
3) Create your virtual environment


```
python3.9 -m venv venv
```


4) Enter in your environment

Linux / OSX

```
source venv/bin/activate venv venv
```

Windows

```
.\venv\Scripts\activate
```

5) Install Libraries
```
pip3 install -r requirements.txt or pip install -r requirements.txt
```

6) Run project

```
python main.py
```

### If you want to see the RNN performances and architecture, please run this in your console
```
python
import os

script = """
tensorboard --logdir models/logs
"""

os.system("bash -c '%s'" % script)

#Then click on the http://localhost to analyse the training process and the performance of the model

```
