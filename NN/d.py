import numpy as np
import pandas as pd
import sys
import os
from neural_network import NeuralNetwork
from sklearn.metrics import classification_report
from PIL import Image
import matplotlib.pyplot as plt

train_data_path = sys.argv[1]
test_data_path = sys.argv[2]
output_path = sys.argv[3]

# train_data_path is a folder which contains many folders named as 01,02,... and inside each folder there are multiple images of the same digit class. the numbers on the folder names represent the digit class. the numbers upper bound can be anything
def load_data(data_path):
    X = []
    Y = []
    for folder in sorted(os.listdir(data_path)):  # Sort to ensure consistent ordering
        if folder.startswith('.'):  # Skip hidden files
            continue
        digit = int(folder)
        digit_path = os.path.join(data_path, folder)
        for img_file in os.listdir(digit_path):
            if img_file.startswith('.'):  # Skip hidden files
                continue
            img_path = os.path.join(digit_path, img_file)
            img = Image.open(img_path)
            img_array = np.array(img)
            img_vector = img_array.flatten() / 255.0  # normalize pixel values
            X.append(img_vector)
            Y.append(digit)
    
    print(f"Loaded {len(X)} images from {data_path}")
    X = np.array(X)
    Y = np.array(Y)
    
    # Convert labels to 0-indexed (if they start from 1)
    min_label = Y.min()
    if min_label > 0:
        Y = Y - min_label  # Shift labels to start from 0
        print(f"Labels shifted: original range [{min_label}, {Y.max() + min_label}] → new range [0, {Y.max()}]")
    
    print(f"Loaded {len(X)} images from {data_path}")
    return X, Y

X_train, Y_train = load_data(train_data_path)
X_test, Y_test = load_data(test_data_path)

f1_scores = []

for hidden_layer in [[512],[512,256],[512,256,128],[512,256,128,64]]:
    nn = NeuralNetwork(mini_batch_size=32, n_features=1024, hidden_layers=hidden_layer, n_classes=36, activation='relu', learning_rate=0.01)
    print(f"Training NN with hidden layer size: {hidden_layer}")
    print(X_train.shape, Y_train.shape)
    nn.train(X_train, Y_train, epochs=100)

    Y_pred_train = nn.predict(X_train)
    Y_pred_test = nn.predict(X_test)
    report = classification_report(Y_test, Y_pred_test, output_dict=True)
    f1_score = report['weighted avg']['f1-score']
    f1_scores.append((hidden_layer, f1_score))

    with open(f"{output_path}/output_hidden_{hidden_layer}.txt", "w") as f:
        f.write("Training Classification Report:\n")
        f.write(classification_report(Y_train, Y_pred_train))
        f.write("\nTesting Classification Report:\n")
        f.write(classification_report(Y_test, Y_pred_test))

# plot f1 scores vs network depth
depths = [len(h) for h, _ in f1_scores]
scores = [score for _, score in f1_scores]
plt.figure()
plt.plot(depths, scores, marker='o')
plt.title('F1 Score vs Network Depth')
plt.xlabel('Network Depth (Number of Hidden Layers)')
plt.ylabel('F1 Score')
plt.xticks(depths)
plt.grid()
plt.savefig(f"{output_path}/f1_score_vs_depth.png") 
plt.show()
plt.close() 