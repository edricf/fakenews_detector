# Import Packages

# Web app
from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 

# Preproc
import pandas as pd 
import numpy as np 
from keras.preprocessing import sequence
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# load models
import tensorflow as tf
import _pickle as pickle 
from keras.models import load_model
import joblib

# Models
with open('./models/tok.pickle', 'rb') as handle:
	tok = pickle.load(handle) # tokenizer

LR_model = joblib.load('./models/Logistic_model.sav') # Logistic Regression
LSTM_model = load_model('./models/LSTM.h5') # LSTM

global graph
graph = tf.get_default_graph() 

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	title = request.form['title']
	news = request.form['News']
	title_news = title + " " + news

	if int(request.form.get('myList'))==2:
		# Logistic Regression
		pred = LR_model.predict([title_news])[0] 
	else:
		# LSTM Neural Network
		max_len = 175
		sequences = tok.texts_to_sequences([title_news])
		test_padded = sequence.pad_sequences(sequences, maxlen=max_len)

		with graph.as_default():
			pred = LSTM_model.predict_classes(test_padded, batch_size=1, verbose=1)[0][0]

	return render_template('index.html', pred=pred)

if __name__ == '__main__':
	app.run(debug=True)