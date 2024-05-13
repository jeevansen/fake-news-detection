import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import re
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('News2.csv',index_col=0)
#data.head()
data.shape
data = data.drop(["title", "subject","date"], axis = 1)
data.isnull().sum()
# Shuffling
data = data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(["index"], axis=1, inplace=True)
sns.countplot(data=data,
            x='class',
            order=data['class'].value_counts().index)
plt.show()

# Function to preprocess text data
from wordcloud import WordCloud
def preprocess_text(text_data):
    preprocessed_text = []
    for sentence in tqdm(text_data):
        sentence = re.sub(r'[^\w\s]', '', sentence)
        preprocessed_text.append(' '.join(token.lower()
                                for token in str(sentence).split()
                                if token not in stopwords.words('english')))
    return preprocessed_text
preprocessed_review = preprocess_text(data['text'].values)
data['text'] = preprocessed_review

from sklearn.feature_extraction.text import CountVectorizer
def get_top_n_words(corpus, n=None):
	vec = CountVectorizer().fit(corpus)
	bag_of_words = vec.transform(corpus)
	sum_words = bag_of_words.sum(axis=0)
	words_freq = [(word, sum_words[0, idx])
				for word, idx in vec.vocabulary_.items()]
	words_freq = sorted(words_freq, key=lambda x: x[1],
						reverse=True)
	return words_freq[:n]


common_words = get_top_n_words(data['text'], 20)
df1 = pd.DataFrame(common_words, columns=['Review', 'count'])

print(df1)

df1.groupby('Review').sum()['count'].sort_values(ascending=False).plot(
	kind='bar',
	figsize=(10, 6),
	xlabel="Top Words",
	ylabel="Count",
	title="Bar Chart of Top Words Frequency"
)
plt.show()


###########################################################################
name_list=['Logistic Regression', 'Decision Tree', 'Support Vector Machine', 
           'Naive Bayes', 'K Nearest Neighbour', 'Random Forest', 
           'Convolutional Neural Network']
acc_list=[]
from prettytable import PrettyTable
t=PrettyTable(['Classifier', 'Accuracy'])

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

x_train, x_test, y_train, y_test = train_test_split(data['text'],
			data['class'],test_size=0.25, random_state=25)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
x_train = vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test)

model = LogisticRegression()
model.fit(x_train, y_train)

# testing the model
accuracy = accuracy_score(y_test, model.predict(x_test))
print("Logistic Regression Accuracy:", accuracy)
t.add_row(['Logistic Regression', accuracy])
acc_list.append(accuracy)

from sklearn.tree import DecisionTreeClassifier

model2 = DecisionTreeClassifier()
model2.fit(x_train, y_train)

# testing the model
accuracy = accuracy_score(y_test, model2.predict(x_test))
print("Decision Tree Accuracy:", accuracy)
t.add_row(['Decision Tree', accuracy])
acc_list.append(accuracy)

# Confusion matrix of Results from Decision Tree classification
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, model2.predict(x_test))

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
											display_labels=[False, True])

cm_display.plot()
plt.show()
#######################

from sklearn.svm import SVC
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(x_train, y_train)
y_pred = svm_classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("SVM Accuracy:", accuracy)
t.add_row(['Support Vector Machine', accuracy])
acc_list.append(accuracy)

from sklearn.naive_bayes import GaussianNB
naive_bayes_classifier = GaussianNB()
x1_train = x_train.toarray()
x1_test = x_test.toarray()
naive_bayes_classifier.fit(x1_train, y_train)
y_pred = naive_bayes_classifier.predict(x1_test)
accuracy2 = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy:", accuracy2)
t.add_row(['Naive Bayes', accuracy2])
acc_list.append(accuracy2)

from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=2)
knn_classifier.fit(x_train, y_train)
y_pred = knn_classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)   
print("KNN Accuracy:", accuracy)
t.add_row(['K Nearest Neighbour', accuracy])
acc_list.append(accuracy)

from sklearn.ensemble import RandomForestClassifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(x_train, y_train)
y_pred = random_forest_classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", accuracy)
t.add_row(['Random Forest', accuracy])
acc_list.append(accuracy)

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(data['text'])
X = tokenizer.texts_to_sequences(data['text'])
X = pad_sequences(X, maxlen=100)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['class'])
# Split the dataset into training and testing sets
X_train, X_test, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the CNN model
model = Sequential([
    Embedding(input_dim=10000, output_dim=100),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train2, epochs=5, batch_size=64, validation_data=(X_test, y_test2))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test2)
print("CNN Test Loss:", loss)
print("CNN Test Accuracy:", accuracy)

t.add_row(['Convolutional Neural Network', accuracy])
acc_list.append(accuracy)

print(t)
plt.bar(name_list, acc_list, color= ('red', 'green', 'cyan', 'purple', 'magenta', 
                                     'orange', 'blue'))
plt.xticks(rotation='vertical')
plt.show()
###########################################################################


# Function to predict labels for new data using trained classifier
def predict_label(new_data):
    new_X = vectorization.transform(new_data)
    prediction = random_forest_classifier.predict(new_X)
    return prediction[0]

# Function to handle button click event
def on_predict(file_path):
    new_data = pd.read_csv(file_path, index_col=0)
    # Clear the text widget
    text_widget.delete(1.0, tk.END)
    #display the text
    text_widget.insert(tk.END, "Text: {}\n".format("\n".join(new_data["text"])))
    new_data = new_data.drop(["title", "subject", "date"], axis=1)

    new_data['text'] = preprocess_text(new_data['text'].values)

    # Predict label for new data
    prediction = predict_label(new_data['text'])
    # Display prediction
    if prediction:
        result_label.config(text='The given news is Real', foreground='green', font=('Arial', 12, 'bold'))
    else:
        result_label.config(text='The given news is Fake', foreground='red', font=('Arial', 12, 'bold'))
    # Display prediction
   # result_label.config(text=f"Prediction: {prediction}")
    
# Create Tkinter window
window = tk.Tk()
window.title("Fake News Detection")
text_widget = tk.Text(window, width=80, height=20)
text_widget.pack()
def browse_file():
    file_path = file_path = filedialog.askopenfilename(title="Select a CSV file", filetypes=[("CSV files", "*.csv")])
    if file_path:
        on_predict(file_path)

# Button to trigger prediction
browse_button = ttk.Button(window, text="Browse File", command=browse_file)
browse_button.pack()

# Label to display prediction result
result_label = ttk.Label(window, text="")
result_label.pack()

# Train classifier and vectorizer

window.mainloop()

