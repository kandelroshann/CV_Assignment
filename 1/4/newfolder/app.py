from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

click_count = 0
uv1, uv2 = np.zeros((2,)), np.zeros((2,))
K_inv = np.linalg.inv(
    np.array([
        [2553, 0, 1152],
        [0, 2553, 530],
        [0, 0, 1]])
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/estimate_length', methods=['POST'])
def estimate_length():
    global uv1, uv2, click_count
    
    zc = float(request.form['zc'])
    uv1 = np.array([float(coord) for coord in request.form['uv1'].split(',')])
    uv2 = np.array([float(coord) for coord in request.form['uv2'].split(',')])

    xyz1 = K_inv.dot(np.append(uv1, 1)) * zc
    xyz2 = K_inv.dot(np.append(uv2, 1)) * zc
    length = np.linalg.norm((xyz2 - xyz1))

    return str(length)

if __name__ == '__main__':
    app.run(debug=True)
