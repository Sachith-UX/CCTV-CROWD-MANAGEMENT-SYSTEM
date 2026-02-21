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
    "B11": ("Main Hall", 0),  # Local default webcam
    "B2": ("Library", "http://172.20.10.5:8080/video")  # IP camera stream
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
