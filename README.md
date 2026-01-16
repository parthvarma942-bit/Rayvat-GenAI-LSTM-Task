# Rayvat-GenAI-LSTM-Task
Generative AI: LSTM Text Generation
Overview
This project involves developing a text generator using a Long Short-Term Memory (LSTM) model. The model is trained on a dataset of text sequences to predict and generate new, coherent text based on a seed input.


Dataset

Source: Shakespeare's Complete Works (Project Gutenberg).



Format: The dataset is a .txt file containing a large body of text.

Task Implementation
1. Preprocessing 

Loaded the raw text and converted it to lowercase.

Removed punctuation to reduce noise in the training data.


Tokenized the text into sequences of characters.

Created input-output pairs where the input is a sequence of tokens and the output is the next token.

2. Model Architecture 

The model was built using TensorFlow/Keras with the following layers:


Embedding Layer: To represent tokens in a dense vector space.


LSTM Layers: Two LSTM layers (256 units each) to capture long-term dependencies.



Dense Layer: A final dense layer with softmax activation for token prediction.


Compilation: Compiled using the Adam optimizer and categorical crossentropy loss function.

3. Training Process 

The dataset was split into training and validation sets.


Early Stopping was implemented to monitor loss and prevent overfitting during the training process.

Text Generation Logic 

The model generates text by taking a seed sequence of 100 characters.

It iteratively predicts the next token and appends it to the sequence to generate a continuous flow of text.

Token indices are converted back into readable characters for the final output.

Sample Outputs 

Seed: "to be or not to be that is the question"


Generated Text: "...whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune or to take arms against a sea of troubles..." 

How to Run
Ensure you have Python 3.x and TensorFlow installed.

Place the shakespeare.txt file in the same directory as the script.

Run the script: Rayvat-GenAI-Task-checkpoint.ipynb
