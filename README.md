# Karachi Smart Traffic AI

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Architecture

```
camera / RTSP / video ──► kst.detector   VehicleDetector  vehicle_det_fp16.onnx (YOLO11m, 7 classes)
                          kst.tracker    IoUTracker       track IDs + class vote smoothing
                          kst.pipeline   Pipeline         budget: N vehicles/frame, every K frames
                               │
                               ▼  per-vehicle crop (Pillow cleaned: median, autocontrast)
                          kst.analyzers  FeatureDetector  best.pt: tinted_windshield,
                                                           police_revolving_light, front_plate
                                         PlateReader      plate box -> Pillow upscale/sharpen
                                                           -> RapidOCR (PaddleOCR on ONNX)
                          kst.color_shape                 Shades-of-Gray + HSV vote, box-geometry shape
                               │
                               ▼  majority votes across a track's analyses
                          kst.storage    SQLite `sightings` + fuzzy plate / feature search
app.py  (UI only): Live monitor · Photo check · Search · Records · Models
```

| File | Role |
|---|---|
| `kst/config.py` | paths + default thresholds (kept low: vehicle 0.25, features/plate 0.15) |
| `kst/imaging.py` | all Pillow cleaning (colour constancy, plate enhancement, sharpness) |
| `kst/detector.py` | ONNX vehicle detector (uses CUDA automatically if onnxruntime-gpu is installed) |
| `kst/analyzers.py` | best.pt features, plate OCR, optional plateReader.pt slot |
| `kst/pipeline.py` | orchestration, per-track aggregation, drawing |
| `kst/storage.py` | DB, plate normalisation, O/0 B/8 S/5-tolerant search |
| `app_legacy.py` | previous single-file app (kept for reference) |

## Notes on the delivered models

- **plateReader.pt is a stock COCO YOLOv8n** (80 generic classes, trained 2022). It cannot
  read plates, so it is used as a **two-wheeler helper** instead. Bicycle/motorcycle boxes are
  merged with the rider's person box, which catches bikes the main model scores below 0.1.
- **Reading zone** at 50-100% of frame height (`READING_ZONE` in `kst/config.py`). Vehicles are
  counted when they enter it and analysed (plate, features, colour) while inside it. The live
  overlay/table shows all detections in real time; the final table keeps the unique tracks from
  analyzed frames.
- **Plate boxes** (magenta) come from: best.pt `front_plate` if conf >= 0.35 (rarely: the head is
  weak), else RapidOCR's DBNet text detector on the lower part of the vehicle, else a Pillow
  bright-rectangle search. Text is then read by RapidOCR. Burned-in camera text is masked
  out via `OSD_MASKS`.
- Videos play **once**. 1 frame is analysed, then N frames are skipped (sidebar, default 5).
- **best.pt** (the `best.pt.zip` file is the checkpoint itself and is copied to `best.pt`)
  has only 3 classes and was trained for 3 epochs, so its confidences are low. That's why
  the defaults are low.
- No colour or make/model weights were delivered. Colour is a Pillow/HSV heuristic.
- On CPU an analysed frame takes about 1.5 s with nothing in the zone, and up to ~4.5 s with
  several vehicles being read. A GPU build of onnxruntime/torch removes most of this.
- Plates are only readable when the vehicle is close (crop ≥ 120 px). Use **Photo check**
  with a close-up car photo to test OCR.
