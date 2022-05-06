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
### If you want to see the RNN performances and architecture, please run this in your console
```
script = """
tensorboard --logdir models/logs
"""
os.system("bash -c '%s'" % script)
#Then click on the http://localhost to analyse the training process and the performance of the model

```
