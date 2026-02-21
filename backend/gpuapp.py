from flask import Flask, render_template, Response, jsonify
import cv2
from crowd_counter import CrowdCounter
import threading
import time
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Initialize YOLO model once ---
try:
    crowd_counter = CrowdCounter(model_path='models/yolov5s.pt')
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    crowd_counter = None

# --- Enhanced Building Configuration with Realistic Capacities ---
cameras = {
    "b_1": ("Corridor", 0,30),  # Local default webcam
    "b_2": ("Production Building in", "rtsp://admin:abc123ABC@10.40.34.2:554/Streaming/Channels/101",35),
    "b_3": ("ELECTRICAL BUILDING 1", "rtsp://admin:FYP12345@10.40.16.236:554/Streaming/Channels/101",30),  # RTSP Camera 1 - Entrance
    "b_4": ("ELECTRICAL BUILDING 2", "rtsp://admin:@Deee123@10.40.16.217:554/Streaming/Channels/101",30),  # RTSP Camera 1 - Exit
    "b_5": ("ELECTRICAL BUILDING 3", "rtsp://admin:@Deee123@10.40.16.196:554/Streaming/Channels/101",30),  # RTSP Camera 2 - Entrance
    "b_6": ("ELECTRICAL BUILDING 4", "rtsp://admin:DeeeEngex@10.40.16.214:554/Streaming/Channels/101",30),  # RTSP Camera 2 - Exit
    "b_7": ("Production Building in", "rtsp://admin:abc123ABC@10.40.34.2:554/Streaming/Channels/101",35),
    "b_8": ("Production Building Out", "rtsp://admin:abc123ABC@10.40.34.6:554/Streaming/Channels/101",35),
    "b_9": ("Department of Electrical and Electronic Engineering", "http://10.40.16.217:8080/video",30),
    "b_10": ("Department of Computer Engineering", "http://192.168.1.109:8080/video",30),
    "b_11": ("Electrical and Electronic Workshop", "http://192.168.1.110:8080/video",30),
    "b_12": ("Surveying Lab", "http://192.168.1.111:8080/video",30),
    "b_13": ("Soil Lab", "http://192.168.1.112:8080/video",30),
    "b_14": ("Materials Lab", "http://192.168.1.113:8080/video",30),
    "b_15": ("Environmental Lab", "http://192.168.1.114:8080/video",30),
    "b_16": ("Fluids Lab", "http://10.40.34.2:8080/video",30),
    "b_17": ("New Mechanics Lab", "http://192.168.1.116:8080/video",30),
    "b_18": ("Applied Mechanics Lab", "http://192.168.1.117:8080/video",30),
    "b_19": ("Thermodynamics Lab", "http://192.168.1.118:8080/video",30),
    "b_20": ("Generator Room", "http://192.168.1.119:8080/video",30),
    "b_21": ("Engineering Workshop", "http://192.168.1.120:8080/video",30),
    "b_22": ("Engineering Carpentry Shop", "http://192.168.1.121:8080/video",30),
    "b_23": ("Drawing Office 2", "http://192.168.1.122:8080/video",30),
    "b_24": ("Outside camera 1", "rtsp://admin:admin123@10.50.29.254:554/Streaming/Channels/401",30),
    "b_25": ("AR Office 6", "rtsp://admin:abcd1234@10.40.29.248:554/Streaming/Channels/601",30),
    "b_26": ("AR Office 5", "rtsp://admin:abcd1234@10.40.29.248:554/Streaming/Channels/501",30),
    "b_27": ("AR Office 4", "rtsp://admin:abcd1234@10.40.29.248:554/Streaming/Channels/401",30),
    "b_28": ("AR Office 3", "rtsp://admin:abcd1234@10.40.29.248:554/Streaming/Channels/301",30),
    "b_29": ("AR Office 2", "rtsp://admin:abcd1234@10.40.29.248:554/Streaming/Channels/201",30),
    "b_30": ("AR Office 1", "rtsp://admin:abcd1234@10.40.29.248:554/Streaming/Channels/101",30),
    "b_31": ("Computer Lab",0,90)## computer lab building that i have add newly
}
    ##"b_31: : ("Computer lab2", "rtsp://admin:abcd1234@10.40.29.248:554/Streaming/Channels/101",90)"

returnIDs = {
    "b_1": "B1",
    "b_2": "B2",
    "b_3": "B3",
    "b_4": "B4",
    "b_5": "B5",
    "b_6": "B6",
    "b_7": "B7",
    "b_8": "B15",
    "b_9": "B9",
    "b_10": "B10",
    "b_11": "B11",
    "b_12": "B12",
    "b_13": "B13",
    "b_14": "B14",
    "b_15": "B8",
    "b_16": "B16",
    "b_17": "B17",
    "b_18": "B18",
    "b_19": "B19",
    "b_20": "B20",
    "b_21": "B21",
    "b_22": "B22",
    "b_23": "B23",
    "b_24": "B24",
    "b_25": "B25",
    "b_26": "B26",
    "b_27": "B27",
    "b_28": "B28",
    "b_29": "B29",
    "b_30": "B30",
    "b_31":"B31",
}

# --- Simple Data Structures ---
crowd_data = {}
data_lock = threading.Lock()

# Initialize data structures
for building_id, (name, source, capacity) in cameras.items():
    crowd_data[building_id] = {
        'current_count': 0,
        'max_capacity': capacity,
        'occupancy_rate': 0.0,
        'last_updated': 0,
        'status': 'offline'
    }

# --- Simplified Video Streaming ---
def gen_frames(building_id, camera_source):
    """Simple video frame generator with people counting"""
    cap = None
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            logger.info(f"Attempting to connect to {building_id}: {camera_source}")
            cap = cv2.VideoCapture(camera_source)
            
            # Configure capture settings
            if isinstance(camera_source, str) and camera_source.startswith('rtsp://'):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 15)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            elif isinstance(camera_source, str) and camera_source.startswith('http://'):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                cap.set(cv2.CAP_PROP_FPS, 20)
            
            if not cap.isOpened():
                logger.warning(f"Failed to open {building_id}, attempt {retry_count + 1}")
                retry_count += 1
                time.sleep(2)
                continue
            
            logger.info(f"Successfully connected to {building_id}")
            consecutive_failures = 0
            frame_count = 0
            last_count_update = time.time()
            
            while True:
                success, frame = cap.read()
                
                if not success:
                    consecutive_failures += 1
                    logger.warning(f"Frame read failed for {building_id}, count: {consecutive_failures}")
                    
                    if consecutive_failures >= 31:##Adding one more second to reconnecting 
                        logger.error(f"Too many failures for {building_id}, reconnecting...")
                        break
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0
                frame_count += 1
                
                # Process every 5th frame to reduce CPU load
                if frame_count % 5 == 0 and crowd_counter is not None:
                    try:
                        # Resize frame for faster processing
                        height, width = frame.shape[:2]
                        if width > 640:
                            scale = 640 / width
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            frame_resized = cv2.resize(frame, (new_width, new_height))
                        else:
                            frame_resized = frame.copy()
                        
                        # YOLO people detection
                        count, annotated_frame = crowd_counter.count_people(frame_resized)
                        
                        # Resize annotated frame back to original size if needed
                        if width > 640:
                            annotated_frame = cv2.resize(annotated_frame, (width, height))
                        
                        # Update crowd data every 3 seconds
                        if time.time() - last_count_update > 3:
                            building_name = cameras[building_id][0]
                            max_capacity = cameras[building_id][2]
                            occupancy_rate = (count / max_capacity) * 100 if max_capacity > 0 else 0
                            
                            with data_lock:
                                crowd_data[building_id]['current_count'] = count
                                crowd_data[building_id]['occupancy_rate'] = occupancy_rate
                                crowd_data[building_id]['last_updated'] = time.time()
                                crowd_data[building_id]['status'] = 'online'
                            
                            last_count_update = time.time()
                            logger.info(f"{building_id}: {count} people detected")
                        
                        frame = annotated_frame
                        
                    except Exception as e:
                        logger.error(f"Error in people detection for {building_id}: {e}")
                        count = 0
                
                # Add overlay information
                building_name = cameras[building_id][0]
                max_capacity = cameras[building_id][2]
                
                with data_lock:
                    current_count = crowd_data[building_id]['current_count']
                    occupancy_rate = crowd_data[building_id]['occupancy_rate']
                
                # Color-coded overlay
                if occupancy_rate > 80:
                    color = (0, 0, 255)  # Red
                elif occupancy_rate > 60:
                    color = (0, 165, 255)  # Orange
                elif occupancy_rate > 30:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 255, 0)  # Green
                
                cv2.putText(frame, f"Building: {building_id.upper()}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"{building_name}", 
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Count: {current_count}/{max_capacity}", 
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Occupancy: {occupancy_rate:.1f}%", 
                           (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Add connection type indicator
                if isinstance(camera_source, str) and camera_source.startswith('rtsp://'):
                    cv2.putText(frame, "RTSP LIVE", 
                               (frame.shape[1] - 120, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif isinstance(camera_source, str) and camera_source.startswith('http://'):
                    cv2.putText(frame, "HTTP LIVE", 
                               (frame.shape[1] - 120, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Encode frame
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.05)  # 20 FPS max
                
        except Exception as e:
            logger.error(f"Exception in {building_id} stream: {str(e)}")
            retry_count += 1
            if cap:
                cap.release()
            
            with data_lock:
                crowd_data[building_id]['status'] = 'error'
            
            if retry_count < max_retries:
                logger.info(f"Retrying {building_id} in 3 seconds...")
                time.sleep(3)
        finally:
            if cap:
                cap.release()
    
    # If all retries failed, yield error frame
    logger.error(f"All connection attempts failed for {building_id}")
    with data_lock:
        crowd_data[building_id]['status'] = 'offline'
    
    error_frame = create_error_frame(building_id, "Connection Failed")
    _, buffer = cv2.imencode('.jpg', error_frame)
    frame_bytes = buffer.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def create_error_frame(building_id, error_message):
    """Create an error frame when camera connection fails"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    cv2.putText(frame, f"Building: {building_id.upper()}", 
               (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, error_message, 
               (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "Check camera connection", 
               (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return frame

# --- Routes ---
@app.route('/')
def index():
    return render_template('indexgpu.html', buildings=cameras.keys())

@app.route('/video_feed/<building_id>')
def video_feed(building_id):
    if building_id not in cameras:
        return "Invalid building ID", 404
    
    camera_source = cameras[building_id][1]
    logger.info(f"Starting video feed for {building_id}")
    
    return Response(gen_frames(building_id, camera_source),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- API Routes ---
@app.route('/api/crowd/<building_id>')
def api_single_building(building_id):
    if building_id not in cameras:
        return jsonify({"error": "Invalid building ID"}), 404
    
    building_name = cameras[building_id][0]
    
    with data_lock:
        data = crowd_data[building_id].copy()
    
    return jsonify({
        "building_id": building_id,
        "building_name": building_name,
        "current_crowd": data['current_count'],
        "max_capacity": data['max_capacity'],
        "occupancy_rate": round(data['occupancy_rate'], 1),
        "status": data['status'],
        "last_updated": data['last_updated']
    })

@app.route('/api/crowd_all')
def api_all_buildings():
    all_data = []
    
    with data_lock:
        for building_id in cameras.keys():
            building_name = cameras[building_id][0]
            data = crowd_data[building_id].copy()
            
            all_data.append({
                "building_id": returnIDs[building_id] ,
                "building_name": building_name,
                "total_count": data['current_count'],
                "max_capacity": data['max_capacity'],
                "occupancy_rate": round(data['occupancy_rate'], 1),
                "status": data['status'],
                "last_updated": data['last_updated']
            })
    
    return jsonify(all_data)

@app.route('/api/heat_map')
def api_heat_map():
    """Get heat map data with detailed building info"""
    heat_map_data = []
    
    with data_lock:
        for building_id, (name, source, capacity) in cameras.items():
            data = crowd_data[building_id].copy()
            
            # Determine heat level
            occupancy_rate = data['occupancy_rate']
            if data['status'] != 'online':
                heat_level = 'offline'
            elif occupancy_rate == 0:
                heat_level = 'empty'
            elif occupancy_rate <= 25:
                heat_level = 'low'
            elif occupancy_rate <= 50:
                heat_level = 'medium'
            elif occupancy_rate <= 75:
                heat_level = 'high'
            else:
                heat_level = 'critical'
            
            heat_map_data.append({
                "building_id": building_id,
                "building_name": name,
                "current_count": data['current_count'],
                "max_capacity": capacity,
                "occupancy_rate": round(occupancy_rate, 1),
                "heat_level": heat_level,
                "status": data['status'],
                "last_updated": data['last_updated']
            })
    
    return jsonify(heat_map_data)
#api/system_stats 
@app.route('/api/system_stats')
def api_system_stats():
    """Get system performance statistics"""
    with data_lock:
        total_people = sum(data['current_count'] for data in crowd_data.values())
        total_capacity = sum(data['max_capacity'] for data in crowd_data.values())
        avg_occupancy = (total_people / total_capacity * 100) if total_capacity > 0 else 0
        
        online_cameras = sum(1 for data in crowd_data.values() if data['status'] == 'online')
        high_occupancy_buildings = sum(1 for data in crowd_data.values() if data['occupancy_rate'] > 75)
    
    return jsonify({
        "total_people": total_people,
        "total_capacity": total_capacity,
        "avg_occupancy": round(avg_occupancy, 1),
        "high_occupancy_count": high_occupancy_buildings,
        "online_cameras": online_cameras,
        "total_cameras": len(cameras),
        "active_streams": 1,  # Only one stream at a time in this simplified version
        "system_health": "Good"
    })

if __name__ == "__main__":
    logger.info("=== ENGEX 2025 Simplified Crowd Monitoring System ===")
    logger.info(f"Total Buildings: {len(cameras)}")
    
    # Print camera types
    rtsp_cameras = [bid for bid, (name, source, cap) in cameras.items() 
                   if isinstance(source, str) and source.startswith('rtsp://')]
    http_cameras = [bid for bid, (name, source, cap) in cameras.items() 
                   if isinstance(source, str) and source.startswith('http://')]
    
    logger.info(f"RTSP Cameras: {len(rtsp_cameras)} - {rtsp_cameras}")
    logger.info(f"HTTP Cameras: {len(http_cameras)} - {http_cameras}")
    logger.info(f"Local Cameras: 1 - ['b_1']")
    
    logger.info("Starting Flask server on http://0.0.0.0:5000")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)


