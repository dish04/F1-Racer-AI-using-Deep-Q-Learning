from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from agent import DQLAgent
from agent import prep_image
import json
import os
import neat
import numpy as np
import cv2
import base64
import re

app = Flask(__name__)
cors = CORS(app, origins='*')
agent = DQLAgent()
iters  = 0

@app.route('/getdata', methods=['GET'])
def get_data():
    with open('map.json', 'r') as file:
            data = json.load(file)
            # print(data)
            if(len(data) == 0): print('Sending empty map')
            else: print('Map sent\n',data)
            return jsonify(data),200


@app.route('/receive_map', methods=['POST'])
def submit():
    try:
        data = request.json  # Get the JSON data sent from the frontend
        print("Received data:", data)
        with open('map.json', 'w') as file:
            json.dump(data, file, indent=4)
        return jsonify({'message': 'Map received successfully', 'received_data': data}), 200
    except Exception as e:
        return jsonify({'message': 'Failed to process Map', 'error': str(e)}), 400

@app.route('/receive_env', methods=['POST'])
def env():
    try:
        data = request.json  # Get the JSON data sent from the frontend
        print("Received data:", data)
        with open('env.json', 'w') as file:
            json.dump(data, file, indent=4)
        return jsonify({'message': 'Env received successfully', 'received_data': data}), 200
    except Exception as e:
        return jsonify({'message': 'Failed to process Env', 'error': str(e)}), 400

@app.route('/receive_screen', methods=['POST'])
def screen():
    global agent, iters
    data = request.json
    img_data = data['image']
    img_data = re.sub('^data:image/.+;base64,', '', img_data)
    img_data = base64.b64decode(img_data)
    with open('screenshot.png', 'wb') as f:
            f.write(img_data)
    prep_image(data['pos'],'screenshot.png')
    agent.update_state(cv2.imread('prepd.png'),data['r'])
    if(iters>=15000):
        # agent.target_train()
        iters = 0
    else:
        iters+=1
    print("reward = ",data['r'])
    return jsonify({'message': 'Image saved successfully'}), 200

@app.route('/receive_begin', methods=['POST'])
def begin():
    data = request.json
    
    img_data = data['image']
    img_data = re.sub('^data:image/.+;base64,', '', img_data)
    img_data = base64.b64decode(img_data)
    with open('screenshot.png', 'wb') as f:
            f.write(img_data)
    global agent
    prep_image(data['pos'],'screenshot.png')
    agent.begin(cv2.imread('prepd.png'))
        # print(prep_image)
    return jsonify({'message': 'Begin state received successfully'}), 200

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/make')
def maker():
    return render_template('maker.html')

@app.route('/get_move', methods=['GET'])
def get_move():
    # Get a move
    global agent
    move = agent.out_action()
    return jsonify({'move':move}),200


if __name__ == '__main__':
    # Start the server
    app.run(debug=True)
