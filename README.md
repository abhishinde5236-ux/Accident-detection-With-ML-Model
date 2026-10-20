# Accident Detection With ML Model

# Road Safety AI Suite — HelmGuard

Multi-model AI system for **helmet violation enforcement** + **accident detection & alerting** using YOLOv8. Built with Flask + OpenCV + Firebase.

---

## Architecture

```
User (Browser)
    │
    ├─ /helmet  → Helmet enforcement dashboard
    │               ├─ Image upload  → POST /api/detect/image
    │               ├─ Live camera   → POST /api/detect/frame  (via WebRTC)
    │               └─ Challan registry + review queue
    │
    ├─ /accident → Accident monitoring dashboard
    │               ├─ Image/video upload → POST /api/accident/detect/{image,video}
    │               ├─ Live monitor       → POST /api/accident/detect/frame
    │               ├─ Alert history      → GET /api/accident/alerts
    │               └─ Map (Leaflet + OSM tiles + auto-detected hospitals)
    │
    └─ /         → Landing page (module selector)
```

**Backend modules** (`modules/`):

```
modules/
├── __init__.py
├── app.py                    → Flask entry point, registers all routes
├── model_loader.py           → Helmet YOLOv8 model load + warmup
├── detector.py               → Helmet detection, rider filtering, OCR
├── accident_model_loader.py  → Accident YOLOv8 model load (separate)
├── accident_detector.py      → Accident detection inference, severity
├── hospital_locator.py       → [NEW] Auto-discover hospitals via OpenStreetMap
├── location_config.py        → Camera/hospital location management
├── accident_alert.py         → Alert dispatch (local, Firebase, ESP32)
├── accident_routes.py        → Accident + location API endpoints
├── routes.py                 → Helmet detection + challan API endpoints
├── challan_generator.py      → PDF / PNG challan generation (ReportLab / PIL)
└── database.py               → Firebase Realtime DB + local JSON fallback
```

---

## Feature Details

### 1. Helmet Violation Detection

**Goal:** Detect motorcyclists riding without helmets from camera feeds.

#### Flow

```
Frame / Image
    ↓
YOLOv8 inference (model: helmet_yolov8.pt — 4 classes)
    ├─ bike
    ├─ with helmet
    ├─ withouthelmet
    └─ numberplate
    ↓
detect_objects() → raw detections
    ↓
filter_riders_only()
    └─ Keeps only persons ON a bike (spatial overlap check via _is_on_bike())
    └─ Removes pedestrians, pillion riders
    └─ Fallback: if no bike detected, still flags no-helmet
    ↓
get_front_rider_only()
    └─ Multiple no-helmet riders on same bike → keep ONLY the front rider
    └─ Uses bounding-box area (largest = closest to camera)
    ↓
EasyOCR on numberplate bbox
    └─ Grayscale → 2x upscale → Otsu threshold → OCR with allowlist [A-Z0-9]
    └─ Validates against Indian plate pattern (MH12AB1234)
    └─ Result stored in detection dict as plate_text
    ↓
is_violation() → True if no-helmet rider confirmed
    ↓
[if violation]
    ├─ build_challan_data() → challan dict with plate, fine (₹500), timestamp
    ├─ save_challan()       → Firebase RTDB + local JSON
    └─ generate_challan()   → PDF (ReportLab) or PNG (PIL) with evidence photo
```

**`detector.py` key functions:**

| Function | Purpose |
|----------|---------|
| `detect_objects(frame)` | YOLO inference + OCR on plates |
| `filter_riders_only(dets)` | Keep only bike riders, remove pedestrians |
| `get_front_rider_only(dets)` | Deduplicate multiple riders on same bike |
| `_is_on_bike(person_bbox, bike_bbox)` | Spatial overlap check (30% horizontal + vertical) |
| `is_violation(dets)` | Final violation boolean |
| `read_plate_text(frame, bbox)` | EasyOCR on cropped plate region |
| `_clean_plate(raw)` | Regex format check for Indian plates |
| `draw_detections(frame, dets)` | Annotate frame with boxes + labels |

**`challan_generator.py`:**

| Function | Purpose |
|----------|---------|
| `build_challan_data(dets, source)` | Assemble challan dict from detections |
| `generate_challan(data, evidence)` | Return (bytes, ext) — PDF or PNG |
| `generate_pdf_challan(data, evidence)` | ReportLab A4 PDF with header, fields, photo, footer |
| `generate_png_challan(data, evidence)` | PIL fallback (white background) |

### 2. Accident Detection

**Goal:** Detect road accidents from the same camera feed, determine severity, and alert nearby hospitals.

#### Flow

```
Frame / Image / Video
    ↓
YOLOv8 inference (model: accident_yolov8.pt — 3 classes)
    ├─ accident
    ├─ moderate
    └─ severe
    ↓
detect_accident() → detections with severity scores
    ↓
is_accident()     → True if any ALERT_CLASSES detected
    ↓
should_send_alert() → is_accident() + cooldown check (30s default)
    ↓
[if alert needed]
    ├─ Get camera GPS from location_config (lat, lng)
    ├─ Find nearest hospital via Overpass API
    ├─ send_alert() → dispatch to all channels
    └─ _dispatch_all()
         ├─ _save_local()        → static/alerts_db.json
         ├─ _save_firebase()     → /accident_alerts/ in Firebase RTDB
         ├─ _post_hospital()     → Log to console (demo)
         └─ _post_esp32()        → HTTP POST to ESP32 OLED display
```

**Severity levels:**

| Level   | Confidence Threshold | Class Trigger |
|----------|---------------------|---------------|
| CRITICAL | ≥ 0.85              | "severe" class or high-conf accident |
| HIGH     | ≥ 0.70              | "moderate" class |
| MEDIUM   | ≥ 0.55              | "accident" class (default) |
| NONE     | < 0.55              | — |

**`accident_detector.py` key functions:**

| Function | Purpose |
|----------|---------|
| `detect_accident(frame)` | YOLO inference (or random mock) |
| `is_accident(dets)` | True if any detection is an alert class |
| `should_send_alert(dets)` | Cooldown-gated alert trigger |
| `get_highest_severity(dets)` | Worst severity among detections |
| `draw_accident_detections(frame, dets)` | Annotate with severity-colored boxes |

### 3. Hospital Discovery (Local-first + OpenStreetMap fallback)

**`modules/hospital_locator.py`**

Two-tier hospital lookup:

| Priority | Source | File | Editable? |
|----------|--------|------|-----------|
| 1st | Local overrides (admin-managed) | `static/local_hospitals.json` | Yes (via UI) |
| 2nd | OpenStreetMap Overpass API | Live query (cached in `static/hospital_cache.json`) | Read-only |
| 3rd | Fallback (estimated) | Hardcoded | Auto |

#### Alert dispatch logic

```
Camera at (lat, lng) detects accident
    ↓
query_nearby_hospitals(lat, lng, radius=5km)
    ↓
Load local hospitals from static/local_hospitals.json
    ├─ Any active hospital within 5km?
    │   YES → return nearest local hospital (alert dispatches here)
    │   NO  → query Overpass API
    │         ├─ Found? → return nearest OSM hospital
    │         └─ Not found → return estimated fallback hospital
```

#### UI display logic

```
get_hospitals() → returns local + OSM merged list
    ├─ Local overrides marked with "Manual" badge (editable)
    └─ OSM results marked with "OSM" badge (read-only)
```

#### Overpass query

```
POST https://overpass-api.de/api/interpreter
Body: [out:json];(node[amenity=hospital](around:5000,lat,lng);
                   way[amenity=hospital](around:5000,lat,lng););out center;
```

Parsed response: name, lat, lng, phone. Results cached per location (60min TTL).

**`hospital_locator.py` key functions:**

| Function | Purpose |
|----------|---------|
| `query_nearby_hospitals(lat, lng, radius_km, include_osm)` | Local-first then OSM. `include_osm=True` merges both (for UI) |
| `get_local_hospitals()` | Return all local overrides |
| `add_local_hospital()` / `update_local_hospital()` / `delete_local_hospital()` | CRUD for local overrides |
| `_fallback_nearest(lat, lng, radius_km)` | Offline fallback hospitals |
| `_haversine(lat1, lng1, lat2, lng2)` | Distance in km |

**`location_config.py` key functions:**

| Function | Purpose |
|----------|---------|
| `get_hospitals()` | Local + OSM merged, dedup'd by coordinates |
| `get_nearest_hospital(lat, lng)` | Nearest hospital with local-first priority |
| `get_cameras()` | List all configured cameras |
| `add_camera()` / `update_camera()` / `delete_camera()` | Camera CRUD |
| `get_camera_by_id(id)` | Lookup single camera |
| `add_hospital()` / `update_hospital()` / `delete_hospital()` | Wraps local override CRUD |

### 4. Alert System

**`modules/accident_alert.py`**

#### Dispatch channels

| Channel | Method | Description |
|---------|--------|-------------|
| Local JSON | `_save_local()` | Writes to `static/alerts_db.json` |
| Firebase | `_save_firebase()` | Saves to `/accident_alerts/{alert_id}` |
| Console | `_post_hospital()` | Logs detected hospital name + distance (demo) |
| ESP32 | `_post_esp32()` | HTTP POST to ESP32 IP for OLED display |

#### Alert data shape

```json
{
  "alert_id":        "ACC-A1B2C3D4",
  "timestamp":       "2026-07-09 14:30:00",
  "camera_id":       "CAM-01",
  "camera_lat":      21.0164,
  "camera_lng":      79.0293,
  "severity":        "HIGH",
  "detected_classes":["accident"],
  "max_confidence":  0.87,
  "nearest_hospital": {
    "id": "HOSP-OSM-01",
    "name": "All India Institute of Medical Sciences, Nagpur",
    "lat": 21.0312,
    "lng": 79.0485,
    "distance_km": 2.16,
    "phone": ""
  },
  "status": "SENT",
  "source": "vision"
}
```

### 5. Model Loading

**`model_loader.py`** (Helmet) vs **`accident_model_loader.py`** (Accident)

Both follow the same pattern:
1. Check for PyTorch + CUDA → select device (`cuda` / `cpu`)
2. Check if model `.pt` file exists → load or fall back to **mock mode**
3. In mock mode: generates random detections for UI testing

| Feature | Helmet | Accident |
|---------|--------|----------|
| Model file | `models/helmet_yolov8.pt` | `models/accident_yolov8.pt` |
| Classes | bike, numberplate, with helmet, withouthelmet | accident, moderate, severe |
| Confidence threshold | 0.50 | 0.55 |
| Loading | Synchronous (blocks startup) | Async (background thread) |
| Mock fallback | Random bike/helmet/no-helmet/plate | Random accident with 8% chance per frame |

### 6. Camera Management

Cameras are stored in `static/location_config.json` with lat/lng coordinates:

```json
{
  "cameras": [
    {
      "id": "CAM-01",
      "label": "Main Intersection",
      "lat": 21.0164,
      "lng": 79.0293,
      "zone": "Zone A",
      "active": true
    }
  ]
}
```

Managed via UI on the accident dashboard (add/edit/delete cameras) which calls:
- `GET /api/locations/cameras`
- `POST /api/locations/cameras`
- `PUT /api/locations/cameras/{id}`
- `DELETE /api/locations/cameras/{id}`

### 7. Data Storage

#### Firebase Realtime DB (optional)

Requires `firebase_credentials.json` in project root.

- `/challans/{challan_id}` — helmet violation challans
- `/accident_alerts/{alert_id}` — accident alerts
- Falls back to local JSON files silently if Firebase is unavailable

#### Local JSON fallback

| File | Content |
|------|---------|
| `static/challans_db.json` | All challans (keyed by ID) |
| `static/alerts_db.json` | All accident alerts (keyed by ID) |
| `static/hospital_cache.json` | Cached Overpass API results |
| `static/uploads/` | Evidence photos, challan PDFs/PNGs, annotated videos |

### 8. Frontend Architecture

#### Pages

| Route | Template | Purpose |
|-------|----------|---------|
| `/` | `templates/index.html` | Landing page — choose helmet or accident module |
| `/helmet` | `templates/helmet.html` | Helmet enforcement dashboard |
| `/accident` | `templates/accident.html` | Accident monitoring + map |

#### JavaScript modules (`static/js/`)

| File | Purpose |
|------|---------|
| `api.js` | HTTP client wrapping all backend endpoints |
| `ui.js` | Toast, tabs, detection tags, progress bars, stats |
| `image_detect.js` | Image upload → YOLO detection → plate OCR → review queue |
| `live_feed.js` | WebRTC camera → frame capture → detect loop → overlay boxes |
| `pending_review.js` | Admin review queue: approve (download PDF), reject, mark paid |
| `challans_tab.js` | Challan registry table with status filter + paid/reject actions |

#### Detection tags color map

```
bike          → orange
with helmet   → green
withouthelmet → red
numberplate   → cyan
person        → purple
```

---

## API Reference

### Helmet Module

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/detect/image` | Upload image → detect helmet violation |
| POST | `/api/detect/frame` | Base64 frame → detect (for live feed) |
| GET | `/api/challans` | List all challans |
| GET | `/api/challan/{id}/file` | Download challan PDF/PNG |
| POST | `/api/challan/{id}/status` | Update status (PAID/REJECTED) |
| GET | `/api/status` | Model + Firebase health |

### Accident Module

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/accident/detect/frame` | Base64 frame → accident detection |
| POST | `/api/accident/detect/image` | Upload image → accident detection |
| POST | `/api/accident/detect/video` | Upload video → sampled frame analysis |
| GET | `/api/accident/alerts` | List all accident alerts |
| GET | `/api/accident/status` | Accident model health |
| POST | `/api/accident/device-alert` | Ingest alert from ESP32/IoT device |
| GET | `/api/accident/esp32-alert` | Poll endpoint for ESP32 OLED |

### Location Module

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/locations/cameras` | List all cameras |
| POST | `/api/locations/cameras` | Add camera |
| PUT | `/api/locations/cameras/{id}` | Update camera |
| DELETE | `/api/locations/cameras/{id}` | Delete camera |
| GET | `/api/locations/hospitals` | All hospitals (local overrides + OSM auto-detected) |
| POST | `/api/locations/hospitals` | Add local hospital override |
| PUT | `/api/locations/hospitals/{id}` | Update local hospital override |
| DELETE | `/api/locations/hospitals/{id}` | Delete local hospital override |
| GET | `/api/locations/map_config` | OSM tile config |
| GET | `/api/locations/nearest?lat=&lng=` | Nearest hospital (local-first → OSM → fallback) |

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Place model files
#   models/helmet_yolov8.pt        (optional — falls back to mock)
#   models/accident_yolov8.pt      (optional — falls back to mock)

# Optional: Firebase credentials
#   firebase_credentials.json      (optional — falls back to local JSON)

# Run
python app.py
# → http://127.0.0.1:5000
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `flask` + `flask-cors` | Web server + CORS |
| `opencv-python` | Image processing, annotation |
| `Pillow` | PNG challan fallback, image resize |
| `numpy` | Array ops |
| `ultralytics` | YOLOv8 inference (both models) |
| `firebase-admin` | Firebase Realtime DB (optional) |
| `python-dotenv` | Environment variables |
| `reportlab` | Professional PDF challan generation (optional) |
| `easyocr` | Number plate OCR |
| `requests` | Overpass API queries (hospital discovery) |
#   A c c i d e n t - d e t e c t i o n - W i t h - M L - M o d e l  
 