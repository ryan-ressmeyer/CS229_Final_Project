import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


for i in stopwords.words('english'):
    print(i)



