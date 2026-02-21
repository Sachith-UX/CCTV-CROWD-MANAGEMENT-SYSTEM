from flask import Flask, render_template, Response, jsonify
import cv2
from crowd_counter import CrowdCounter
import threading
import time

app = Flask(__name__)

# --- Load YOLO model once ---
crowd_counter = CrowdCounter(model_path='models/yolov5s.pt')

# --- Define cameras with building names (SAME STRUCTURE AS BEFORE) ---
# Format: building_id: (building_name, camera_source)
cameras = {
    "b_1": ("Corridor", 0),  # Local default webcam
    "b_2": ("Production Building in", "rtsp://admin:abc123ABC@10.40.34.2:554/Streaming/Channels/101"),
    "b_3": ("ELECTRICAL BUILDING 1", "rtsp://admin:FYP12345@10.40.16.236:554/Streaming/Channels/101"),  # RTSP Camera 1 - Entrance
    "b_4": ("ELECTRICAL BUILDING 2", "rtsp://admin:@Deee123@10.40.16.217:554/Streaming/Channels/101"),  # RTSP Camera 1 - Exit
    "b_5": ("ELECTRICAL BUILDING 3", "rtsp://admin:@Deee123@10.40.16.196:554/Streaming/Channels/101"),  # RTSP Camera 2 - Entrance
    "b_6": ("ELECTRICAL BUILDING 4", "rtsp://admin:DeeeEngex@10.40.16.214:554/Streaming/Channels/101"),  # RTSP Camera 2 - Exit
    "b_7": ("Production Building in", "rtsp://admin:abc123ABC@10.40.34.2:554/Streaming/Channels/101"),
    "b_8": ("Production Building Out", "rtsp://admin:abc123ABC@10.40.34.6:554/Streaming/Channels/101"),
    "b_9": ("Department of Electrical and Electronic Engineering", "http://10.40.16.217:8080/video"),
    "b_10": ("Department of Computer Engineering", "http://192.168.1.109:8080/video"),
    "b_11": ("Electrical and Electronic Workshop", "http://192.168.1.110:8080/video"),
    "b_12": ("Surveying Lab", "http://192.168.1.111:8080/video"),
    "b_13": ("Soil Lab", "http://192.168.1.112:8080/video"),
    "b_14": ("Materials Lab", "http://192.168.1.113:8080/video"),
    "b_15": ("Environmental Lab", "http://192.168.1.114:8080/video"),
    "b_16": ("Fluids Lab", "http://10.40.34.2:8080/video"),
    "b_17": ("New Mechanics Lab", "http://192.168.1.116:8080/video"),
    "b_18": ("Applied Mechanics Lab", "http://192.168.1.117:8080/video"),
    "b_19": ("Thermodynamics Lab", "http://192.168.1.118:8080/video"),
    "b_20": ("Generator Room", "http://192.168.1.119:8080/video"),
    "b_21": ("Engineering Workshop", "http://192.168.1.120:8080/video"),
    "b_22": ("Engineering Carpentry Shop", "http://192.168.1.121:8080/video"),
    "b_23": ("Drawing Office 2", "http://192.168.1.122:8080/video"),
    "b_24": ("Outside camera 1", "rtsp://Admin:Admin123@10.50.29.254:554/Streaming/Channels/101"),
    "b_25": ("AR Office 6", "rtsp://admin:abcd1234@10.40.29.248:554/Streaming/Channels/601"),
    "b_26": ("AR Office 5", "rtsp://admin:abcd1234@10.40.29.248:554/Streaming/Channels/501"),
    "b_27": ("AR Office 4", "rtsp://admin:abcd1234@10.40.29.248:554/Streaming/Channels/401"),
    "b_28": ("AR Office 3", "rtsp://admin:abcd1234@10.40.29.248:554/Streaming/Channels/301"),
    "b_29": ("AR Office 2", "rtsp://admin:abcd1234@10.40.29.248:554/Streaming/Channels/201"),
    "b_30": ("AR Office 1", "rtsp://admin:abcd1234@10.40.29.248:554/Streaming/Channels/101")
}

# --- Store crowd counts ---
crowd_data = {b: 0 for b in cameras.keys()}

# --- Thread-safe lock for updating crowd_data ---
data_lock = threading.Lock()

# --- Connection retry settings ---
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds

# --- Enhanced video streaming generator with RTSP support ---
def gen_frames(building_id, camera_source):
    """Generate video frames with people counting"""
    retry_count = 0
    
    while retry_count < MAX_RETRY_ATTEMPTS:
        cap = None
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(camera_source)
            
            # Configure settings based on source type
            if isinstance(camera_source, str) and camera_source.startswith('rtsp://'):
                print(f"Connecting to RTSP stream for {building_id}: {camera_source}")
                # RTSP specific settings for better performance
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 15)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                # Set timeout for RTSP
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            
            # Test if camera is accessible
            if not cap.isOpened():
                print(f"Failed to open camera for {building_id}, attempt {retry_count + 1}")
                if cap:
                    cap.release()
                retry_count += 1
                time.sleep(RETRY_DELAY)
                continue
            
            print(f"Successfully connected to {building_id} camera")
            consecutive_failures = 0
            
            while True:
                success, frame = cap.read()
                
                if not success:
                    consecutive_failures += 1
                    print(f"Failed to read frame from {building_id}, failure count: {consecutive_failures}")
                    
                    # If too many consecutive failures, try to reconnect
                    if consecutive_failures >= 10:
                        print(f"Too many failures for {building_id}, attempting reconnection...")
                        break
                    
                    # Skip this frame and continue
                    time.sleep(0.1)
                    continue
                
                # Reset failure counter on successful frame
                consecutive_failures = 0
                
                # Resize frame if too large (for better performance)
                height, width = frame.shape[:2]
                if width > 1280:
                    scale = 1280 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # YOLO people detection
                count, annotated_frame = crowd_counter.count_people(frame)
                
                # Add building info overlay
                building_name = cameras[building_id][0]
                cv2.putText(annotated_frame, f"Building: {building_id.upper()}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"{building_name}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"People Count: {count}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add connection type indicator
                if isinstance(camera_source, str) and camera_source.startswith('rtsp://'):
                    cv2.putText(annotated_frame, "RTSP LIVE", 
                               (annotated_frame.shape[1] - 120, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Update crowd count safely
                with data_lock:
                    crowd_data[building_id] = count
                
                # Encode frame for streaming
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.033)  # ~30 FPS
        
        except Exception as e:
            print(f"Exception in {building_id} video stream: {str(e)}")
            retry_count += 1
            if cap:
                cap.release()
            
            if retry_count < MAX_RETRY_ATTEMPTS:
                print(f"Retrying connection for {building_id} in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Max retry attempts reached for {building_id}")
                break
    
    # Cleanup
    if cap:
        cap.release()
    
    # If all retries failed, yield error frame
    error_frame = create_error_frame(building_id, "Connection Failed")
    _, buffer = cv2.imencode('.jpg', error_frame)
    frame_bytes = buffer.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def create_error_frame(building_id, error_message):
    """Create an error frame when camera connection fails"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add error text
    cv2.putText(frame, f"Building: {building_id.upper()}", 
               (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, error_message, 
               (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "Please check camera connection", 
               (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return frame

# Import numpy for error frame creation
import numpy as np


# --- Routes (SAME AS BEFORE) ---
@app.route('/')
def index():
    return render_template('index3.html', buildings=cameras.keys())

@app.route('/video_feed/<building_id>')
def video_feed(building_id):
    if building_id not in cameras:
        return "Invalid building ID", 404
    
    camera_source = cameras[building_id][1]
    print(f"Starting video feed for {building_id} with source: {camera_source}")
    
    return Response(gen_frames(building_id, camera_source),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Function to get building crowd info (SAME AS BEFORE) ---
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

# --- API for a building (SAME AS BEFORE) ---
@app.route('/api/crowd/<building_id>')
def api_single_building(building_id):
    data = get_building_crowd(building_id)
    if data is None:
        return jsonify({"error": "Invalid building ID"}), 404
    return jsonify(data)

# --- API for all buildings (SAME AS BEFORE) ---
@app.route('/api/crowd_all')
def api_all_buildings():
    all_data = []
    for building_id in cameras.keys():
        all_data.append(get_building_crowd(building_id))
    return jsonify(all_data)

# --- New API for camera connection status ---
@app.route('/api/camera_status')
def camera_status():
    """Check which cameras are RTSP vs HTTP"""
    status = {}
    for building_id, (building_name, camera_source) in cameras.items():
        source_type = "RTSP" if isinstance(camera_source, str) and camera_source.startswith('rtsp://') else "HTTP/Local"
        status[building_id] = {
            "building_name": building_name,
            "source_type": source_type,
            "camera_source": camera_source if not isinstance(camera_source, str) or not camera_source.startswith('rtsp://') else "rtsp://***:***@***"  # Hide credentials
        }
    return jsonify(status)

if __name__ == "__main__":
    print("=== Enhanced Crowd Counter System ===")
    print("Buildings with RTSP cameras:")
    for building_id, (building_name, camera_source) in cameras.items():
        if isinstance(camera_source, str) and camera_source.startswith('rtsp://'):
            print(f"  {building_id}: {building_name}")
    
    print(f"\nTotal buildings: {len(cameras)}")
    print("Starting Flask server on http://0.0.0.0:5000")
    print("Dashboard available at: http://localhost:5000")
    
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)