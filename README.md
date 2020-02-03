# **ChatBot**
From my interest in Natural language processing I intend to make a chatbot using python and NLTK
### **Motivation**
In this modern era digital marketing and onlile shopping is increasing day by day.Real time customer support is a growing need for all type of buisness holder. From this my interset grwon and wanted to make a chatbot.
### **BlogPost**
For detailed overview, here is the accompanying blog[a link](https://medium.com/analytics-vidhya/building-a-simple-chatbot-in-python-using-nltk-7c8c8215ac6e)
### **Pre-requisites**
##### **NLTK(Natural Language Toolkit)**

Natural Language Processing with Python provides a practical introduction to programming for language processing.

For platform-specific instructions, [read here](https://www.nltk.org/install.html)
### **Project files short description**
#### **Data**
For making a chatbot it's necessary to have a large dataset according to ones domain. I had my own dataset where one column shows the queries and another coulmn displays correspondent  intents. Another data file contains intents and theie replies.
### **Data Preprocessing**
```
data_loader.py
```
This file contains all sorts of data preproceesing.For starters:Tokenization,stemming.Bag of words,lower casing.
### **Models**
I tried many models to get better accuracy for better results.For instance:
```
base_model.py
```
```
cnn_bigru.py
```
```
cnn_bilstm.py
```
```
cnn_lstm.py
```
```
tflearn.py
```
To call them I used factory method
```
factory.py
```

### **Train**
For training and prediction and results:
```
training.py
```
### **Configurartion**
For configuring which model,batch size and path to use upadte config.py file:
```

class Config:

    @staticmethod
    def get_model_name() -> str:
        return "CnnBilstm"

    @staticmethod
    def get_batch_size() -> int:
        return 16

    @staticmethod
    def get_epochs() -> int:
        return 200

    @staticmethod
    def get_dataset_path() -> str:
        return "../../data/All_intent.csv"
```

### **Deployment **
Run from terminal
```
python deploy.py
```
