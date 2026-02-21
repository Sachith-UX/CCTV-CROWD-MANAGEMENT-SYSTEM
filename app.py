from flask import Flask, render_template, Response, jsonify
import cv2
from crowd_counter import CrowdCounter
import threading

app = Flask(__name__)

# --- Load YOLO model once ---
crowd_counter = CrowdCounter(model_path='models/yolov5s.pt')

# --- Define cameras with building names ---
# Format: building_id: (building_name, camera_source)
cameras = {
    "b_1": ("Corridor", 0),  # Local default webcam
    "b_2": ("Department of Chemical and Process Engineering", "http://172.20.10.2:8080/video"),
    "b_3": ("Department Engineering Mathematics/Department Engineering Management/Computer Center", "http://192.168.1.102:8080/video"),
    "b_4": ("Drawing Office 1", "http://192.168.1.103:8080/video"),
    "b_5": ("Professor E.O.E. Pereira Theatre", "http://192.168.1.104:8080/video"),
    "b_6": ("Administrative Building", "http://192.168.1.105:8080/video"),
    "b_7": ("Security Unit", "http://192.168.1.106:8080/video"),
    "b_8": ("Electronic Lab", "http://192.168.1.107:8080/video"),
    "b_9": ("Department of Electrical and Electronic Engineering", "http://10.40.16.217:8080/video"),
    "b_10": ("Department of Computer Engineering", "http://192.168.1.109:8080/video"),
    "b_11": ("Electrical and Electronic Workshop", "http://192.168.1.110:8080/video"),
    "b_12": ("Surveying Lab", "http://192.168.1.111:8080/video"),
    "b_13": ("Soil Lab", "http://192.168.1.112:8080/video"),
    "b_14": ("Materials Lab", "http://192.168.1.113:8080/video"),
    "b_15": ("Environmental Lab", "http://192.168.1.114:8080/video"),
    "b_16": ("Fluids Lab", "http://192.168.1.115:8080/video"),
    "b_17": ("New Mechanics Lab", "http://192.168.1.116:8080/video"),
    "b_18": ("Applied Mechanics Lab", "http://192.168.1.117:8080/video"),
    "b_19": ("Thermodynamics Lab", "http://192.168.1.118:8080/video"),
    "b_20": ("Generator Room", "http://192.168.1.119:8080/video"),
    "b_21": ("Engineering Workshop", "http://192.168.1.120:8080/video"),
    "b_22": ("Engineering Carpentry Shop", "http://192.168.1.121:8080/video"),
    "b_23": ("Drawing Office 2", "http://192.168.1.122:8080/video"),
    "b_24": ("Lecture Room (middle-right)", "http://192.168.1.123:8080/video"),
    "b_25": ("Structures Laboratory", "http://192.168.1.124:8080/video"),
    "b_26": ("Lecture Room (bottom-right)", "http://192.168.1.125:8080/video"),
    "b_27": ("Engineering Laboratory", "http://192.168.1.126:8080/video"),
    "b_28": ("Department of Manufacturing and Industrial Engineering", "http://192.168.1.127:8080/video"),
    "b_29": ("Faculty Canteen", "http://192.168.1.128:8080/video"),
    "b_30": ("High Voltage Laboratory", "http://192.168.1.129:8080/video")
}

# --- Store crowd counts ---
crowd_data = {b: 0 for b in cameras.keys()}

# --- Thread-safe lock for updating crowd_data ---
data_lock = threading.Lock()


# --- Video streaming generator ---
def gen_frames(building_id, camera_source):
    cap = cv2.VideoCapture(camera_source)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # YOLO people detection
        count, annotated_frame = crowd_counter.count_people(frame)

        # Update crowd count safely
        with data_lock:
            crowd_data[building_id] = count

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


# --- Routes ---
@app.route('/')
def index():
    return render_template('index2.html', buildings=cameras.keys())


@app.route('/video_feed/<building_id>')
def video_feed(building_id):
    if building_id not in cameras:
        return "Invalid building ID", 404
    return Response(gen_frames(building_id, cameras[building_id][1]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --- Function to get building crowd info ---
def get_building_crowd(building_id):
    if building_id not in cameras:
        return None
    building_name, _ = cameras[building_id]
    with data_lock:
        count = crowd_data[building_id]
    return {
        "building_id": building_id,
        "building_name": building_name,
        "current_crowd": count
    }


# --- API for a building ---
@app.route('/api/crowd/<building_id>')
def api_single_building(building_id):
    data = get_building_crowd(building_id)
    if data is None:
        return jsonify({"error": "Invalid building ID"}), 404
    return jsonify(data)


# --- API for all buildings ---
@app.route('/api/crowd_all')
def api_all_buildings():
    
    all_data = []
    for building_id in cameras.keys():
        all_data.append(get_building_crowd(building_id))
    return jsonify(all_data)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)