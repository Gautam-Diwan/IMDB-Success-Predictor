# IMDB Success Predictor
Project involves Web Scraping custom IMDB data between 2020 and 2021 of 10000 movies and shows sorted by number of votes ,fine tuning a pre trained DistilBERT Transformer using Transfer Learning and then saving and reusing the saved model for further use.

## Stack
* DistilBERT Transformer
* Tensorflow
* Numpy and Pandas
* Selenium, BeautifulSoup4 and requests

## Metrics
- Accuracy achieved: 81.3492%
- ROC_AUC_Score achieved: 0.7217

## Installation
<p>
 1) Ensure Python and Jupyter Notebook are installed. Optionally Conda environment can also be used.
 
 2) Install the required modules using 
 <pre><code>pip install -r requirements.txt</code> <br>
or <code>conda install -r requirements.txt</code><br>
or <code>!pip install -r requirements.txt</code> for Google Colab.
     </pre>
 
 3) Selenium requires browser specific drivers. Guides for Chrome and Firefox are mentioned below. Alternatively,this step is optional if the notebook is run on Google Colab.<br>
   Chrome: https://chromedriver.chromium.org/getting-started <br>
   Firefox: https://www.lambdatest.com/blog/selenium-firefox-driver-tutorial/
</p>

## Training
<p>
 1)(Optional) Run the <code>IMDB Web scraper</code> . This generates the already provided csv file and imdb_movies pickle file.
 
 2) Run the <code>IMDB Web scraper</code> on an environment which has GPU acceleration. Here it is used with Google Colab where Nvidia Tesla T4 or Nvidia Tesla K80 are allocated.<br><pre>
  Training Time: Roughly 20-25 mins
  Epochs: 10 
  Training Batch Size: 8
  Max length of each Sentence: 512 <br></pre>
  A <code>Movie_prediction_model</code> directory is created with <code>config.json</code> file(provided) and a <code>tf_model.h5</code> (not provided due to space constraints).
 </p>
 
 ## Usage
 <p>
 1) Ensure the model has been created inside <code>Movie_prediction_model</code> directory.
 
 2) Run the python file using <code>python DistilBERT_Movie_Classifier.py</code>
 
 3) Enter the description of the movie or TV show you want to predict for. An output will be generated with the binary prediction of success based of IMDB Ratings.
 </p>
 
