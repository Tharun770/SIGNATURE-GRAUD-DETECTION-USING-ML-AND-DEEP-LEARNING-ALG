import numpy as np
import os
import matplotlib.image as mpimg
from scipy import ndimage
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import tensorflow.compat.v1 as tf
import pandas as pd
from glob import glob
from flask import Flask, render_template, request
import shutil
from sklearn.preprocessing import StandardScaler

tf.disable_v2_behavior()

app = Flask(__name__)

# Paths
genuine_image_paths = r"C:\Users\harsh\OneDrive\Desktop\me\real"
forged_image_paths = r"C:\Users\harsh\OneDrive\Desktop\me\forged"
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Image preprocessing functions
def rgbgrey(img):
    greyimg = np.zeros((img.shape[0], img.shape[1]))
    for row in range(len(img)):
        for col in range(len(img[row])):
            greyimg[row][col] = np.average(img[row][col])
    return greyimg

def greybin(img):
    blur_radius = 0.8
    img = ndimage.gaussian_filter(img, blur_radius)
    thres = threshold_otsu(img)
    binimg = img > thres
    binimg = np.logical_not(binimg)
    return binimg

def preproc(path, img=None):
    if img is None:
        img = mpimg.imread(path)
    grey = rgbgrey(img)
    binimg = greybin(grey)
    r, c = np.where(binimg == 1)
    if len(r) == 0 or len(c) == 0:
        return binimg
    signimg = binimg[r.min():r.max() + 1, c.min():c.max() + 1]
    return signimg

# Feature extraction functions
def Ratio(img): 
    return np.sum(img) / (img.shape[0] * img.shape[1]) if img.shape[0] * img.shape[1] > 0 else 0

def Centroid(img):
    numOfWhites = np.sum(img)
    if numOfWhites == 0: 
        return 0, 0
    a = np.array([0, 0], dtype=float)
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col]: 
                a = np.add(a, [row, col])
    rowcols = np.array([img.shape[0], img.shape[1]])
    centroid = a / numOfWhites
    centroid = centroid / rowcols
    return centroid[0], centroid[1]

def EccentricitySolidity(img):
    r = regionprops(img.astype("int8"))
    return (r[0].eccentricity if r else 0, r[0].solidity if r else 0)

def SkewKurtosis(img):
    h, w = img.shape
    x, y = np.arange(w), np.arange(h)
    xp, yp = np.sum(img, axis=0), np.sum(img, axis=1)
    total = np.sum(img)
    if total == 0: 
        return (0, 0), (0, 0)
    cx, cy = np.sum(x * xp) / total, np.sum(y * yp) / total
    sx = np.sqrt(np.sum((x - cx) ** 2 * xp) / total) if total > 0 else 0
    sy = np.sqrt(np.sum((y - cy) ** 2 * yp) / total) if total > 0 else 0
    skewx = np.sum(xp * (x - cx) ** 3) / (total * sx ** 3) if sx > 0 else 0
    skewy = np.sum(yp * (y - cy) ** 3) / (total * sy ** 3) if sy > 0 else 0
    kurtx = np.sum(xp * (x - cx) ** 4) / (total * sx ** 4) - 3 if sx > 0 else 0
    kurty = np.sum(yp * (y - cy) ** 4) / (total * sy ** 4) - 3 if sy > 0 else 0
    return (skewx, skewy), (kurtx, kurty)

def getFeatures(path, img=None):
    if img is None: 
        img = mpimg.imread(path)
    img = preproc(path, img=img)
    return (Ratio(img), Centroid(img), *EccentricitySolidity(img), *SkewKurtosis(img))

def getCSVFeatures(path, img=None):
    temp = getFeatures(path, img)
    return (temp[0], temp[1][0], temp[1][1], temp[2], temp[3], temp[4][0], temp[4][1], temp[5][0], temp[5][1])

# Generate CSVs
def makeCSV():
    features_dir = r"C:\Users\harsh\OneDrive\Desktop\me\Features"
    os.makedirs(features_dir, exist_ok=True)
    train_dir = os.path.join(features_dir, "Training")
    test_dir = os.path.join(features_dir, "Testing")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    genuine_files = glob(os.path.join(genuine_image_paths, "*.*"))
    forged_files = glob(os.path.join(forged_image_paths, "*.*"))

    if not genuine_files or not forged_files:
        print("No images found in real or forged directories.")
        return

    for person in range(1, 13):
        per = f"{person:03d}"
        print(f"Generating CSV for user {per}")
        start_idx = (person - 1) * 5
        train_genuine = genuine_files[start_idx:start_idx + 3] if start_idx + 3 <= len(genuine_files) else genuine_files[start_idx:]
        test_genuine = genuine_files[start_idx + 3:start_idx + 5] if start_idx + 5 <= len(genuine_files) else []
        train_forged = forged_files[start_idx:start_idx + 3] if start_idx + 3 <= len(forged_files) else forged_files[start_idx:]
        test_forged = forged_files[start_idx + 3:start_idx + 5] if start_idx + 5 <= len(forged_files) else []

        with open(os.path.join(train_dir, f"training_{per}.csv"), 'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            for source in train_genuine:
                features = getCSVFeatures(source)
                handle.write(','.join(map(str, features)) + ',1\n')
            for source in train_forged:
                features = getCSVFeatures(source)
                handle.write(','.join(map(str, features)) + ',0\n')

        with open(os.path.join(test_dir, f"testing_{per}.csv"), 'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            for source in test_genuine:
                features = getCSVFeatures(source)
                handle.write(','.join(map(str, features)) + ',1\n')
            for source in test_forged:
                features = getCSVFeatures(source)
                handle.write(','.join(map(str, features)) + ',0\n')

# Prediction function (updated to return dictionary)
def predict(user_id, image_path):
    train_path = rf"C:\Users\harsh\OneDrive\Desktop\me\Features\Training\training_{user_id}.csv"
    test_path = r"C:\Users\harsh\OneDrive\Desktop\me\TestFeatures\testcsv.csv"

    if not os.path.exists(train_path):
        return {"error": f"Training data for user {user_id} not found. Please ensure CSVs are generated."}

    # Process test image
    os.makedirs(r"C:\Users\harsh\OneDrive\Desktop\me\TestFeatures", exist_ok=True)
    feature = getCSVFeatures(image_path)
    with open(test_path, 'w') as handle:
        handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y\n')
        handle.write(','.join(map(str, feature)) + '\n')

    # Read data
    df = pd.read_csv(train_path, usecols=range(9))
    train_input = np.array(df.values, dtype=np.float32)
    df = pd.read_csv(train_path, usecols=[9])
    corr_train = tf.keras.utils.to_categorical(df.values.flatten(), 2)
    df = pd.read_csv(test_path, usecols=range(9))
    test_input = np.array(df.values, dtype=np.float32)

    # Normalize inputs
    scaler = StandardScaler()
    train_input = scaler.fit_transform(train_input)
    test_input = scaler.transform(test_input)

    # Define neural network
    n_input, n_classes = 9, 2
    learning_rate, training_epochs = 0.001, 200
    n_hidden_1, n_hidden_2, n_hidden_3 = 7, 10, 30

    tf.reset_default_graph()
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], seed=1)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes], seed=2))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1], seed=3)),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes], seed=4))
    }

    def multilayer_perceptron(x):
        layer_1 = tf.tanh(tf.matmul(x, weights['h1']) + biases['b1'])
        layer_2 = tf.tanh(tf.matmul(layer_1, weights['h2']) + biases['b2'])
        layer_3 = tf.tanh(tf.matmul(layer_2, weights['h3']) + biases['b3'])
        out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
        return out_layer

    logits = multilayer_perceptron(X)
    pred = tf.nn.softmax(logits)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    init = tf.global_variables_initializer()

    # Train and predict
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            _, cost = sess.run([train_op, loss_op], feed_dict={X: train_input, Y: corr_train})
            if cost < 0.0001:
                break
        prediction = pred.eval({X: test_input})
        prob_forged, prob_genuine = prediction[0]
        result = "Genuine" if prob_genuine > prob_forged else "Forged"
        return {
            "user_id": user_id,
            "result": result,
            "forged_prob": f"{prob_forged:.4f}",
            "genuine_prob": f"{prob_genuine:.4f}"
        }

# Generate CSVs at startup
makeCSV()

# Flask route (updated to match the additional snippet and templates)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        if 'image' not in request.files:
            return render_template('result.html', error="No image uploaded")
        image = request.files['image']
        if image.filename == '':
            return render_template('result.htmlinux', error="No image selected")

        # Validate user_id
        if not user_id or not user_id.isdigit() or int(user_id) < 1 or int(user_id) > 12 or len(user_id) != 3:
            return render_template('result.html', error="User ID must be a 3-digit number between 001 and 012")

        # Save uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

        # Predict
        result = predict(user_id, image_path)

        # Clean up uploaded image
        os.remove(image_path)

        # Render result or error
        if "error" in result:
            return render_template('result.html', error=result["error"])
        return render_template('result.html', **result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)








