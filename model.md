# Karachi Smart City AI Traffic Surveillance
### AI-Powered Vehicle Detection, Tracking, ANPR & Forensic Identification

An AI-powered traffic surveillance system designed for Karachi's road infrastructure. The system processes CCTV video footage and live RTSP camera streams to detect, track, identify, and analyze vehicles using a cascaded multi-model computer vision pipeline.

The platform combines vehicle detection, multi-object tracking, Automatic Number Plate Recognition (ANPR), vehicle make/model recognition, color classification, visual attribute detection, and vision-language model (VLM) descriptions into a unified surveillance and forensic search system.

---

## 1. Project Overview

**Target Deployment:** Karachi Police Command & Control  
**Prototype Environment:** Google Colab (NVIDIA T4 GPU, 16GB VRAM)  
**Production Camera Target:** 4MP CCTV cameras  
**Input Sources:** Recorded CCTV footage (MP4) and live CCTV streams (RTSP)  
**Core Framework:** PyTorch, Ultralytics YOLO, ONNX Runtime  
**Inference Optimization:** FP16 Mixed Precision, ONNX Runtime GPU, TensorRT  
**Pipeline Architecture:** Cascaded Multi-Model Inference Pipeline

### Key Capabilities

| Feature | Description |
|---|---|
| Vehicle Detection | Detects vehicles across different categories, including cars, motorcycles, rickshaws, buses, and trucks. |
| Vehicle Tracking | Maintains persistent vehicle identities using multi-object tracking. |
| Best-Shot Selection | Selects the three sharpest vehicle images per tracked vehicle. |
| License Plate Detection | Localizes Pakistani vehicle registration plates. |
| ANPR / OCR | Extracts registration numbers from detected license plates. |
| Super-Resolution | Enhances low-resolution license plate crops when necessary. |
| Plate Classification | Identifies different Pakistani license plate categories. |
| Make & Model Recognition | Identifies vehicle manufacturers and specific models. |
| Vehicle Color Recognition | Classifies vehicle body colors while accounting for different lighting conditions. |
| Attribute Detection | Identifies distinctive visual features, modifications, and decorations. |
| Forensic Descriptions | Generates natural-language vehicle descriptions using a vision-language model. |
| Watchlist Matching | Checks extracted registration information against an authorized watchlist. |
| Searchable Records | Stores structured vehicle information for subsequent investigation and retrieval. |

---

## 2. System Architecture

The system uses a cascaded inference architecture rather than executing every model on every video frame.

The objective is to reduce redundant computation, preserve image quality, and make the complete pipeline practical on an NVIDIA T4 GPU.

### Processing Pipeline

```text
                CCTV INPUT
             MP4 / LIVE RTSP
                    |
                    v
          VIDEO FRAME INGESTION
                    |
                    v
        MEMBER 1: VEHICLE DETECTION
                 YOLOv11m
                    |
                    v
          MULTI-OBJECT TRACKING
                 BoT-SORT
                    |
                    v
            BEST-SHOT GATEKEEPER
         Top-3 Sharpest Vehicle Crops
                    |
          +---------+---------+
          |         |         |
          v         v         v
       MEMBER 2  MEMBER 3  MEMBER 4
          |         |         |
          v         v         v
        ANPR     MAKE/MODEL  ATTRIBUTE
        + OCR    + COLOR     DETECTION
          |         |         |
          v         v         v
      PLATE TEXT  VEHICLE   VISUAL
      & CATEGORY  IDENTITY  FEATURES
          |         |         |
          +---------+---------+
                    |
                    v
           MEMBER 4: INTEGRATION
                    |
                    v
          FLORENCE-2 VLM TAGGING
             (ASYNCHRONOUS)
                    |
                    v
           MASTER JSON RECORD
                    |
          +---------+---------+
          |                   |
          v                   v
     WATCHLIST MATCHING   DATA STORAGE
          |                   |
          v                   v
      POLICE ALERTS      SEARCHABLE LOGS
                              |
                              v
                     DEMO DASHBOARD /
                      ANNOTATED VIDEO
```

### Best-Shot Processing Strategy

Running every downstream model on every frame would introduce unnecessary computational overhead.

Instead:

1. Member 1 detects vehicles in incoming video frames.
2. BoT-SORT assigns persistent tracking IDs.
3. Vehicle crops are evaluated for image sharpness and bounding-box size.
4. The three highest-quality crops are retained for each `track_id`.
5. Members 2, 3, and 4 process these selected images.
6. Results are aggregated into a single vehicle record.

This strategy also allows make/model and color predictions to be combined across multiple observations instead of relying on a single frame.

---

## 3. Compute Requirements & Optimization

### Prototype Hardware

| Component | Specification |
|---|---|
| Platform | Google Colab Free / Pro |
| GPU | NVIDIA T4 |
| GPU Memory | 16GB VRAM |
| Precision | FP16 Mixed Precision |
| Model Export | FP16 ONNX |
| Runtime | ONNX Runtime GPU |
| Checkpoint Storage | Google Drive |

### Estimated VRAM Budget

| Module | Models | Estimated VRAM |
|---|---|---:|
| Member 1 | YOLOv11m + BoT-SORT | 1.4 GB |
| Member 2 | YOLOv11n + Real-ESRGAN + PaddleOCR + MobileNetV3 | 1.8 GB |
| Member 3 | ConvNeXt-Small + MobileNetV3-Large | 1.4 GB |
| Member 4 | YOLOv11s + Florence-2-large | 3.2 GB |
| **Total** | | **~7.8 GB** |

These figures are initial planning estimates, not guaranteed runtime measurements. Actual memory consumption depends on model loading, image sizes, batch sizes, preprocessing buffers, and inference concurrency.

### Mandatory Optimization Practices

- Enable FP16 mixed precision during supported training and inference operations.
- Export final model weights to ONNX.
- Use conditional model execution rather than running every model on every frame.
- Process only the Top-3 selected vehicle crops in downstream modules.
- Release unused GPU memory when appropriate.
- Save model checkpoints to Google Drive every five epochs.
- Monitor actual GPU memory usage and inference performance throughout development.

**Checkpoint Configuration:** `save_period=5`

---

## 4. Team Responsibilities & Model Specifications

The project is divided into four independently developed modules. Each member owns a specific component and must follow the shared integration contracts.

### Member 1 — Vehicle Detection, Tracking & Best-Shot Selection

**Directory:** `member1_detection_tracking/`

**Primary Responsibility:** Detect vehicles, maintain persistent identities, and generate high-quality vehicle crops for downstream processing.

#### Models and Technologies

- YOLOv11m — Vehicle Detection
- BoT-SORT — Multi-Object Tracking
- OSNet — Lightweight Re-identification
- Laplacian Variance — Image Sharpness Evaluation

#### Vehicle Classes

1. `car`
2. `motorcycle`
3. `auto_rickshaw`
4. `qingqi`
5. `pickup_hilux_shehzore`
6. `bus_coach`
7. `truck_decorated`
8. `van_hiace`

#### Training Configuration

**Stage 1 — Base Pakistani Vehicle Adaptation**

| Parameter | Value |
|---|---|
| Image Size | 960 |
| Batch Size | 16 |
| Epochs | 50 |
| Optimizer | AdamW |
| Mixed Precision | FP16 |
| Dataset | Pakistani vehicle datasets |
| Estimated Time | 2.5–3 hours |

**Stage 2 — Karachi CCTV Calibration**

| Parameter | Value |
|---|---|
| Image Size | 1280 |
| Batch Size | 8 |
| Epochs | 25 |
| Initial Learning Rate | 0.001 |
| Dataset | 1,500 locally extracted and corrected CCTV frames |
| Estimated Time | 1–1.5 hours |

The local dataset should represent different lighting conditions, including morning glare, afternoon sunlight, nighttime illumination, and LED/sodium streetlights.

#### Best-Shot Selection

Each vehicle crop receives a quality score:

```python
Score = (
    Laplacian_Variance * 0.6
    + Normalized_BBox_Area * 0.4
    - Edge_Penalty
)
```

The gatekeeper retains the three highest-scoring crops per tracked vehicle.

#### Input Contract

- MP4 video file or RTSP camera stream.

#### Output Contract

```json
{
  "track_id": 1084,
  "vehicle_type": "car",
  "det_conf": 0.94,
  "bbox": [412, 290, 688, 510]
}
```

Additional output: Native-resolution `vehicle_crop.jpg` images.

#### Expected Deliverables

- `train_yolo_vehicles.ipynb`
- `best_shot_tracker.py`
- `weights/vehicle_det_fp16.onnx`

**Estimated Training Time:** 3.5–4.5 hours.

---

### Member 2 — Automatic Number Plate Recognition (ANPR)

**Directory:** `member2_anpr_ocr/`

**Primary Responsibility:** Detect Pakistani registration plates, enhance low-resolution images, extract registration text, and classify plate categories.

#### Models and Technologies

1. YOLOv11n — License Plate Detection
2. Real-ESRGAN-x2-plus — Conditional Super-Resolution
3. PaddleOCR PP-OCRv4 / SVTR-LCNet — Text Recognition
4. MobileNetV3-Small — Plate Category Classification
5. Regex-Based Post-Processing — Registration Format Validation

#### 1. License Plate Detection

| Parameter | Value |
|---|---|
| Model | YOLOv11n |
| Input | Cropped Vehicle Images |
| Image Size | 640 |
| Batch Size | 32 |
| Epochs | 50 |
| Precision | FP16 |
| Estimated Time | 1.5–2 hours |

#### 2. Conditional Super-Resolution

**Model:** Real-ESRGAN-x2-plus

Super-resolution is applied only when the detected license plate crop has a width below 110 pixels.

```python
if plate_width < 110:
    plate_crop = super_resolution(plate_crop)
else:
    plate_crop = original_plate_crop
```

The purpose is to improve the readability of low-resolution plates without unnecessarily increasing inference time.

For future 4MP camera deployments, this step can be bypassed when the original plate crop already contains sufficient detail.

#### 3. Pakistani License Plate OCR

**Model:** PaddleOCR PP-OCRv4 / SVTR-LCNet

The OCR module must support:

- Sindh Ajrak registration plates.
- White private vehicle plates.
- Yellow commercial vehicle plates.
- Black motorcycle plates.
- Two-line stacked motorcycle and rickshaw plates.

**Training Configuration**

| Parameter | Value |
|---|---|
| Batch Size | 64 |
| Epochs | 40 |
| Precision | FP16 |
| Dataset | 20,000 synthetic Pakistani plates + real CCTV crops |
| Estimated Time | 2.5–3 hours |

Synthetic plates will be generated using Pillow and Albumentations.

Training variations should include:

- Motion blur.
- Screw bolts and plate obstructions.
- Nighttime infrared noise.
- Different plate backgrounds.
- Perspective distortion.
- Different character spacing.
- Single-line and two-line registration formats.

#### 4. Plate Category Classification

**Model:** MobileNetV3-Small

| Parameter | Value |
|---|---|
| Input Size | 128 × 64 |
| Batch Size | 64 |
| Epochs | 25 |
| Classes | 5 |
| Estimated Time | 30–45 minutes |

**Classes:**

- `sindh_ajrak`
- `private_white`
- `commercial_yellow`
- `govt_green`
- `non_standard_fancy`

#### 5. Registration Number Post-Processing

A regex-based processor will validate recognized text against supported registration formats.

Example format:

```regex
^[A-Z]{2,3}-\d{3,4}$
```

Character confusion correction:

| OCR Confusion | Possible Correction |
|---|---|
| 0 | O |
| O | 0 |
| 1 | I |
| I | 1 |
| 8 | B |
| B | 8 |
| 5 | S |
| S | 5 |

Corrections must be applied according to expected character positions and regional registration rules, not through unconditional character replacement.

#### Input Contract

Native-resolution vehicle crops from Member 1.

#### Output Contract

```json
{
  "plate_text": "BKA-492",
  "plate_category": "sindh_ajrak",
  "ocr_conf": 0.96,
  "sr_used": true
}
```

#### Expected Deliverables

- `generate_synthetic_plates.py`
- `train_anpr_pipeline.ipynb`
- `anpr_inference.py`
- `weights/`

**Estimated Training Time:** 4.5–6 hours.

---

### Member 3 — Vehicle Make, Model & Color Recognition

**Directory:** `member3_vmmr_color/`

**Primary Responsibility:** Identify vehicle manufacturers, specific models, and body colors while accounting for Karachi's challenging lighting conditions.

#### Models and Technologies

- ConvNeXt-Small — Vehicle Make and Model Recognition (VMMR)
- MobileNetV3-Large — Vehicle Color Classification
- Shades-of-Gray — Color Constancy
- CLAHE — Contrast Enhancement

#### 1. Illumination-Invariant Color Classification

**Model:** MobileNetV3-Large

The color recognition pipeline uses dedicated preprocessing to reduce incorrect predictions caused by sunlight, shadows, and artificial street lighting.

**Preprocessing Pipeline:**

1. Extract relevant vehicle body regions, primarily the hood, roof, and upper door panels.
2. Exclude tires, dark windshields, and irrelevant bumper stickers.
3. Apply Shades-of-Gray color constancy to reduce lighting-induced color casts.
4. Apply CLAHE on the L-channel in LAB color space.
5. Pass the processed crop to the color classification model.

**Training Configuration**

| Parameter | Value |
|---|---|
| Image Size | 224 × 224 |
| Batch Size | 64 |
| Epochs | 35 |
| Loss | Focal Loss |
| Precision | FP16 |
| Estimated Time | 1–1.5 hours |

**Color Classes:**

1. `white`
2. `pearl_white`
3. `silver`
4. `grey`
5. `black`
6. `blue`
7. `red`
8. `green`
9. `gold_beige`
10. `yellow_commercial`

#### 2. Fine-Grained Make & Model Recognition

**Model:** ConvNeXt-Small

The model is initialized using ImageNet-22k pretrained weights and fine-tuned on Pakistani vehicle datasets.

**Training Configuration**

| Parameter | Value |
|---|---|
| Image Size | 384 × 384 |
| Batch Size | 16 |
| Epochs | 30 |
| Loss | Label-Smoothing Cross-Entropy |
| Label Smoothing | 0.1 |
| Precision | FP16 |
| Estimated Time | 3–4 hours |

**Target Vehicle Categories**

The model should recognize commonly observed Pakistani vehicle models, including:

- Suzuki Alto
- Suzuki Cultus
- Suzuki Mehran
- Suzuki WagonR
- Suzuki Bolan
- Toyota Corolla
- Toyota Yaris
- Toyota Hilux
- Honda Civic
- Honda City
- Changan Alsvin

The final class list will be determined by the available labeled dataset.

#### Multi-Frame Prediction

Instead of predicting make and model from a single crop, the system averages the softmax probabilities from the Top-3 selected images for each `track_id`.

This helps reduce errors caused by:

- Partial occlusion.
- Poor viewing angles.
- Motion blur.
- Shadows.
- Reflections.

#### Input Contract

Top-3 vehicle crops from Member 1.

#### Output Contract

```json
{
  "make": "Toyota",
  "model": "Corolla",
  "vmmr_conf": 0.91,
  "color": "white",
  "color_conf": 0.94
}
```

#### Expected Deliverables

- `color_preprocess.py`
- `train_vmmr_color.ipynb`
- `vmmr_color_inference.py`
- `weights/`

**Estimated Training Time:** 4–5.5 hours.

---

### Member 4 — Vehicle Attributes, VLM & Pipeline Integration

**Directory:** `member4_attributes_integration/`

**Primary Responsibility:** Detect distinctive vehicle features, generate detailed forensic descriptions, integrate all modules, and produce the final vehicle records.

#### Models and Technologies

- YOLOv11s — Vehicle Attribute Detection
- Florence-2-large — Vision-Language Model
- ONNX Runtime GPU — Model Inference
- FAISS / SQLite — Searchable Forensic Descriptions

#### 1. Vehicle Attribute Detection

**Model:** YOLOv11s

The detector identifies visual characteristics and modifications that may help distinguish individual vehicles.

**Target Classes:**

1. `tinted_windows`
2. `roof_rack_carrier`
3. `truck_art_heavy_decal`
4. `hood_or_side_stripe`
5. `police_revolving_light`
6. `missing_or_damaged_bumper`
7. `no_front_plate`
8. `spare_wheel_rear`

**Training Configuration**

| Parameter | Value |
|---|---|
| Image Size | 640 |
| Batch Size | 32 |
| Epochs | 50 |
| Precision | FP16 |
| Dataset | Locally annotated Karachi CCTV vehicle crops |
| Estimated Time | 1.5–2.5 hours |

#### 2. Vision-Language Model (VLM) Tagging

**Model:** Florence-2-large

Florence-2 is used as a pretrained zero-shot model. No fine-tuning is required for the initial prototype.

It runs asynchronously on the highest-quality selected vehicle crop to generate a natural-language description.

Example:

> White Toyota Corolla with black matte hood wrap, dark tinted windows, and aftermarket alloy rims.

These descriptions can be stored alongside their corresponding vehicle records and converted into searchable embeddings.

**Search Technologies:**

- FAISS — Vector similarity search.
- SQLite — Lightweight metadata and record storage.

Example natural-language search:

```text
Find white Toyota sedans with tinted windows
and black hood modifications.
```

The system retrieves matching records based on stored descriptions and available structured attributes.

VLM descriptions are observational metadata and must not be treated as verified vehicle identity.

#### 3. Unified Pipeline Orchestrator

**File:** `main_pipeline.py`

The orchestrator connects all four modules into a single processing workflow.

Responsibilities:

- Load exported ONNX models.
- Process MP4 recordings and live RTSP streams.
- Coordinate vehicle tracking and best-shot selection.
- Execute ANPR, color, make/model, and attribute inference.
- Aggregate module outputs.
- Perform authorized watchlist matching.
- Generate structured JSON records.
- Produce annotated video output.
- Store searchable vehicle records.

#### Input Contract

- Vehicle crops from Member 1.
- ANPR results from Member 2.
- Make/model and color results from Member 3.
- Attribute detection results.

#### Output Contract

A unified Master Police Alert Record, defined in Section 5.

#### Expected Deliverables

- `train_attributes.ipynb`
- `vlm_forensic_tagger.py`
- `main_pipeline.py`
- `weights/`

**Estimated Training Time:** 1.5–2.5 hours for attribute detection. Florence-2 uses pretrained weights.

---

## 5. Frozen Master JSON Integration Contract

**File:** `configs/schema.json`

All four modules must follow a consistent data structure to avoid integration problems.

The following JSON structure defines the master vehicle record.

```json
{
  "camera_id": "KHI_CCTV_FEED_01",
  "timestamp": "2026-11-21T11:15:00Z",
  "track_id": 1084,
  "vehicle_type": "car",

  "plate": {
    "text": "BKA-492",
    "category": "sindh_ajrak",
    "confidence": 0.96,
    "super_resolution_applied": true
  },

  "appearance": {
    "make": "Toyota",
    "model": "Corolla",
    "vmmr_confidence": 0.91,
    "color": "white",
    "color_confidence": 0.94,
    "detected_attributes": [
      "tinted_windows",
      "hood_or_side_stripe"
    ],
    "forensic_caption": "White sedan with black matte hood stripe, dark tinted windows, and aftermarket alloy rims"
  },

  "watchlist_alert": {
    "is_flagged": false,
    "reason": null
  }
}
```

### Integration Rules

- All modules must use the same `track_id` for a vehicle during its tracking lifecycle.
- Timestamps must use a consistent ISO 8601 representation.
- Confidence values must remain numeric.
- Missing or unavailable predictions must be represented explicitly as `null`, rather than fabricated values.
- Plate text must preserve the OCR output and any applied corrections must be traceable.
- Attribute names must match the predefined class labels.
- Watchlist matching must use the authorized registration database.
- Low-confidence predictions must remain distinguishable from verified results.

**Important:** The JSON contract is frozen for the initial integration phase. Any structural changes must be communicated to all team members before implementation.

---

## 6. Repository Structure

The project follows a modular directory structure so that each team member can develop and test their component independently.

```text
karachi-police-anpr-cv/
│
├── README.md
├── requirements.txt
│
├── configs/
│   └── schema.json
│
├── shared_data/
│   └── karachi_cctv_crops_v1/
│
├── member1_detection_tracking/
│   ├── train_yolo_vehicles.ipynb
│   ├── best_shot_tracker.py
│   └── weights/
│       └── vehicle_det_fp16.onnx
│
├── member2_anpr_ocr/
│   ├── generate_synthetic_plates.py
│   ├── train_anpr_pipeline.ipynb
│   ├── anpr_inference.py
│   └── weights/
│
├── member3_vmmr_color/
│   ├── color_preprocess.py
│   ├── train_vmmr_color.ipynb
│   ├── vmmr_color_inference.py
│   └── weights/
│
├── member4_attributes_integration/
│   ├── train_attributes.ipynb
│   ├── vlm_forensic_tagger.py
│   ├── main_pipeline.py
│   └── weights/
│
└── demo_outputs/
    ├── annotated_videos/
    ├── vehicle_records/
    └── logs/
```

---

## 7. Dataset Requirements

The project requires both public datasets and locally collected Karachi CCTV data.

### Public Datasets

| Module | Dataset / Source |
|---|---|
| Vehicle Detection | Pak Vehicles Dataset (Roboflow) |
| Vehicle Detection | VT-LPR Pakistani Dataset (GitHub) |
| Plate Detection | Pakistan License Plate Detection (Roboflow) |
| Plate OCR | Pakistani ANPR datasets + synthetic plates |
| Vehicle Color | Car Colors Dataset (Roboflow) |
| Make/Model | Cars Dataset Pakistan (Roboflow) |
| Make/Model | Pakistani VMMR academic datasets |

Dataset availability, licenses, annotations, and actual class coverage must be verified before training.

### Local Karachi CCTV Dataset

The 24-hour Karachi CCTV recording is used for local calibration and evaluation.

**Initial Dataset Target:**

- 2,000 baseline vehicle crops.
- 1,500 calibration frames for vehicle detection.
- 1,200 locally labeled day/night vehicle crops for color classification.
- 800 locally labeled vehicle crops for make/model recognition.
- 1,000 vehicle crops for visual attribute detection.

The same source footage can contribute to multiple datasets, but annotations must be specific to each task.

### Data Annotation

CVAT or Roboflow can be used for:

- Vehicle bounding boxes.
- License plate bounding boxes.
- Vehicle make/model labels.
- Vehicle body color.
- Vehicle modifications and decorations.
- Visual quality verification.

Human verification is required for ground-truth labels, particularly for plate text, make/model, and body color.

Training, validation, and test splits should be separated by recording or time segment wherever possible to prevent near-identical frames from leaking across datasets.

---

## 8. Prototype Execution Plan — 3 Weeks

### Week 1 — Data Extraction & Baseline Training

**Days 1–2:**

- Extract approximately 2,000 diverse frames from the 24-hour Karachi CCTV recording.
- Generate vehicle crops.
- Organize shared datasets.
- Distribute the initial dataset to all four members.

**Days 3–7:**

- Train baseline models using public Pakistani datasets.
- Implement module-specific inference scripts.
- Export initial models to ONNX.
- Verify model loading and inference on the T4 GPU.

**Deliverables:**

- Initial trained models.
- Baseline inference scripts.
- ONNX model exports.
- Shared vehicle crop dataset.

### Week 2 — Karachi-Specific Fine-Tuning

Focus on adapting models to local CCTV conditions.

**Member 1:**
- Fine-tune vehicle detection on Karachi CCTV footage.
- Validate tracking IDs and best-shot selection.

**Member 2:**
- Validate license plate detection.
- Test Real-ESRGAN on low-resolution plates.
- Generate and validate synthetic Sindh plate samples.
- Improve OCR recognition.

**Member 3:**
- Validate day/night color classification.
- Test Shades-of-Gray and CLAHE preprocessing.
- Fine-tune Pakistani make/model recognition.

**Member 4:**
- Annotate truck art, tinted windows, decals, and other visual features.
- Train attribute detection.
- Implement Florence-2 descriptions.

**Deliverables:**

- Karachi-calibrated model weights.
- Module-level evaluation results.
- Updated inference scripts.

### Week 3 — Pipeline Integration & Demonstration

**Integration:**

- Combine all four modules.
- Implement `main_pipeline.py`.
- Validate the frozen JSON contract.
- Implement watchlist matching.
- Integrate asynchronous VLM tagging.
- Generate annotated video output.

**Demonstration:**

- Process the 24-hour Karachi CCTV test recording.
- Generate structured vehicle records.
- Demonstrate registration number extraction.
- Demonstrate make/model and color recognition.
- Demonstrate visual attribute detection.
- Demonstrate natural-language forensic search.

**Final Deliverables:**

- Integrated AI surveillance pipeline.
- Annotated demonstration video.
- Searchable vehicle records.
- JSON outputs.
- Model performance and evaluation report.

---

## 9. Running the Prototype

The initial development environment is Google Colab with an NVIDIA T4 GPU.

### Step 1 — Clone the Repository

```bash
git clone https://github.com/USERNAME/karachi-police-anpr-cv.git

cd karachi-police-anpr-cv
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

The dependency file should include the required versions of:

- PyTorch
- Ultralytics
- ONNX
- ONNX Runtime GPU
- PaddleOCR
- Real-ESRGAN
- OpenCV
- Transformers
- FAISS
- SQLite dependencies
- Albumentations
- Pillow

Exact package versions should be pinned and tested against the selected Colab CUDA environment.

### Step 3 — Configure Google Drive

Mount Google Drive in Colab to persist:

- Training checkpoints.
- Dataset files.
- Model weights.
- Inference outputs.

### Step 4 — Train Individual Modules

Run the corresponding notebooks:

```text
member1_detection_tracking/train_yolo_vehicles.ipynb

member2_anpr_ocr/train_anpr_pipeline.ipynb

member3_vmmr_color/train_vmmr_color.ipynb

member4_attributes_integration/train_attributes.ipynb
```

### Step 5 — Export Models

Export trained models into the appropriate ONNX weight directories.

Validate:

- Model input dimensions.
- Output tensor structure.
- FP16 compatibility.
- ONNX Runtime GPU execution.
- Inference consistency against the original PyTorch models.

### Step 6 — Execute the Integrated Pipeline

The intended orchestrator entry point is:

```bash
python member4_attributes_integration/main_pipeline.py
```

The orchestrator will support two input modes:

**Recorded Video:**

```text
--source 24h_feed.mp4
```

**Live CCTV Stream:**

```text
--source rtsp://camera_ip:554/stream1
```

The RTSP URL must be supplied through authorized camera configuration. Live-stream ingestion also requires appropriate frame buffering, reconnection handling, and stream decoding.

The same model inference interfaces can be reused for both input types.

---

## 10. Live CCTV & Future 4MP Deployment

The prototype is designed to establish the model interfaces and inference workflow before moving to dedicated GPU infrastructure.

### Prototype vs Production

| Component | Prototype | Production Target |
|---|---|---|
| GPU | NVIDIA T4 | NVIDIA L4 / RTX 4090 / A100 |
| Video Input | MP4 / RTSP | Multiple live 4MP RTSP streams |
| Model Format | ONNX FP16 | TensorRT FP16 |
| Inference | ONNX Runtime GPU | TensorRT |
| Storage | Local / Google Drive | Dedicated database and storage infrastructure |
| Processing | Prototype pipeline | Concurrent multi-camera processing |

### Production Optimization

**1. Automatic Super-Resolution Bypass**

The ANPR module skips Real-ESRGAN when plate resolution is already sufficient.

The initial threshold is 110 pixels in plate width.

**2. TensorRT Optimization**

Convert supported ONNX models to TensorRT engines using FP16:

```bash
trtexec \
  --onnx=model.onnx \
  --saveEngine=model.engine \
  --fp16
```

Actual performance gains must be benchmarked on the target hardware.

**3. RTSP Stream Integration**

The system switches from recorded MP4 input to live RTSP streams through the same ingestion interface.

**4. Multi-Camera Scaling**

Production deployment will require:

- Independent camera ingestion.
- Camera-level configuration.
- Concurrent stream processing.
- GPU workload scheduling.
- Frame buffering and backpressure.
- Automatic stream reconnection.
- Centralized vehicle record storage.
- Monitoring and operational logging.

The initial 30+ FPS per 4MP camera target is a performance objective, not a validated guarantee. Actual throughput depends on camera count, resolution, frame sampling, model complexity, and hardware allocation.

---

## 11. Evaluation & Performance Metrics

Each module must be evaluated independently before complete pipeline integration.

| Module | Evaluation Metrics |
|---|---|
| Vehicle Detection | Precision, Recall, mAP50, mAP50-95 |
| Vehicle Tracking | IDF1, HOTA, ID switches |
| Best-Shot Selection | Sharpness distribution, crop quality |
| Plate Detection | Precision, Recall, mAP |
| Plate OCR | Character Error Rate (CER), Exact Match Accuracy |
| Plate Classification | Accuracy, Precision, Recall, F1 |
| Make/Model | Top-1 Accuracy, Top-3 Accuracy |
| Color Classification | Accuracy, Macro F1 |
| Attribute Detection | Precision, Recall, mAP |
| VLM | Human-reviewed description accuracy |
| Integrated Pipeline | End-to-end processing latency, throughput, record completeness |

### Real-World Validation Conditions

Testing must include:

- Morning sunlight and glare.
- Afternoon lighting.
- Nighttime street lighting.
- Motion blur.
- Partial vehicle occlusion.
- Different camera angles.
- Low-resolution footage.
- Dense traffic.
- Motorcycles and two-line registration plates.
- Similar vehicle colors and models.

No accuracy percentage should be claimed until it has been measured against a representative, human-verified test dataset.

---

## 12. Data Security, Privacy & Human Oversight

Because the system processes vehicle registration information and surveillance footage, deployment must include appropriate operational and security controls.

- Only authorized personnel should access camera streams and surveillance records.
- RTSP credentials must be stored securely and never committed to GitHub.
- Sensitive datasets and production footage must not be included in public repositories.
- Watchlist records must come from an authorized and controlled source.
- Access to searches and alerts should be logged.
- Retention and deletion policies must be defined by the deploying authority.
- Model predictions must remain distinguishable from verified information.
- Low-confidence plate recognition and uncertain vehicle identification must be sent for human verification.
- Automated watchlist matches must be treated as investigative leads, not conclusive identification or grounds for enforcement by themselves.

---

## 13. Development Guidelines

To ensure smooth collaboration between all four members:

1. Each member is responsible for their assigned module.
2. Shared JSON contracts must not be changed independently.
3. All models must be tested independently before integration.
4. Every module must provide a consistent inference interface.
5. Model weights must not be committed unnecessarily to GitHub.
6. Training checkpoints should be stored in Google Drive.
7. Code, configuration, and model versions should be tracked.
8. Each member must document model inputs, outputs, and preprocessing requirements.
9. Integration should be tested using shared sample vehicle crops before full-video processing.
10. Evaluation results must be recorded before claiming model readiness.

---

## 14. Project Roadmap

| Phase | Objective | Status |
|---|---|---|
| Phase 1 | Dataset preparation and extraction | Planned |
| Phase 2 | Baseline model training | Planned |
| Phase 3 | Karachi-specific fine-tuning | Planned |
| Phase 4 | Independent module validation | Planned |
| Phase 5 | ONNX model export and integration | Planned |
| Phase 6 | MP4 and RTSP pipeline testing | Planned |
| Phase 7 | Searchable vehicle records and demo | Planned |
| Phase 8 | Production GPU optimization | Future |
| Phase 9 | Multi-camera 4MP deployment | Future |

---

## 15. Important Technical References

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [ONNX Documentation](https://onnx.ai/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [PaddleOCR Documentation](https://www.paddleocr.ai/)
- [Real-ESRGAN GitHub](https://github.com/xinntao/Real-ESRGAN)
- [Florence-2 GitHub](https://github.com/microsoft/Florence)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [CVAT Documentation](https://docs.cvat.ai/)

---

## 16. Project Objective

The ultimate objective is to develop a modular, scalable, and locally adapted AI-powered traffic surveillance platform capable of processing Karachi CCTV footage and live camera streams.

By combining vehicle detection, persistent tracking, Pakistani license plate recognition, make/model classification, illumination-aware color recognition, visual attribute detection, and natural-language forensic descriptions, the system aims to provide a unified vehicle intelligence and investigation-support platform.

The initial three-week implementation focuses on validating the complete pipeline using existing Karachi CCTV footage and Google Colab T4 GPUs.

Once validated, the architecture can be optimized and migrated to dedicated GPU infrastructure for live 4MP multi-camera deployment.

**This repository contains the prototype engineering implementation and integration contracts for the Karachi Smart City AI Traffic Surveillance project.**
