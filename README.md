# Karachi Smart Traffic AI

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Architecture

```
camera / RTSP / video ──► kst.vehicle_detection  VehicleDetector     vehicle_det_fp16.onnx (YOLO11m, 7 classes)
                                                 TwoWheelerDetector  plateReader.pt (COCO): missed bikes
                          kst.tracking           IoUTracker          track IDs + class vote smoothing
                          kst.pipeline           Pipeline            budget: N vehicles/frame, every K frames
                               │
                               ▼  per-vehicle crop (Pillow cleaned: median, autocontrast)
                          kst.vehicle_analysis   FeatureDetector     best.pt: tinted_windshield,
                                                                     police_revolving_light, front_plate
                          kst.plate_reading      FastAlprPlateReader fast-alpr: YOLOv9 plate detector
                                                                     -> fast-plate-ocr (both ONNX)
                                                 RapidOcrPlateReader fallback: RapidOCR (PaddleOCR on ONNX)
                          kst.appearance                             Shades-of-Gray + HSV vote, box-geometry shape
                               │
                               ▼  majority votes across a track's analyses
                          kst.database           SQLite `sightings` + fuzzy plate / feature search
app.py  (UI only): Live monitor · Photo check · Search · Records · Models
```

| File | Role |
|---|---|
| `kst/config.py` | paths + default thresholds (kept low: vehicle 0.25, features/plate 0.15) |
| `kst/image_utils.py` | all Pillow cleaning (colour constancy, plate enhancement, sharpness) |
| `kst/vehicle_detection.py` | ONNX vehicle detector (uses CUDA automatically if onnxruntime-gpu is installed) + two-wheeler detector |
| `kst/tracking.py` | IoU / centre-distance tracker |
| `kst/vehicle_analysis.py` | per-crop analysis: best.pt features, plate locating/reading |
| `kst/plate_reading.py` | fast-alpr plate detection + OCR, RapidOCR fallback, plate-shape helpers |
| `kst/appearance.py` | colour and shape heuristics |
| `kst/pipeline.py` | orchestration, reading zone, per-track aggregation |
| `kst/overlay.py` | drawing boxes, plates, zone and counts on frames |
| `kst/database.py` | DB, plate normalisation, O/0 B/8 S/5-tolerant search |

## Notes on the delivered models

- **plateReader.pt is a stock COCO YOLOv8n** (80 generic classes, trained 2022). It cannot
  read plates, so it is used as a **two-wheeler detector** instead. Bicycle/motorcycle boxes are
  merged with the rider's person box, which catches bikes the main model scores below 0.1.
- **Reading zone** covers the bottom 60% of frame height, from 40% to 100%
  (`READING_ZONE` in `kst/config.py`). Vehicles are counted when they enter it and analysed
  (plate, features, colour) while inside it. The live table shows only vehicles in the zone
  for the displayed frame, synchronized with the highlighted detections; the final table keeps
  one row for every unique track that entered the zone.
- **Plate boxes** (magenta) come from [fast-alpr](https://github.com/ankandrew/fast-alpr)'s
  YOLOv9 plate detector run on each vehicle crop (`ALPR_DETECTOR`, `ALPR_DET_CONF`). Text is
  read by fast-plate-ocr (`ALPR_OCR`), with RapidOCR as fallback when the read is below
  `ALPR_OCR_MIN_CONF`. The models download to the user cache on first run. On distant CCTV
  plates (~25 px wide), detection is reliable but the text is too blurred for any OCR. Only
  if fast-alpr fails to load does the old chain run: best.pt `front_plate`, then the RapidOCR
  text detector, then a Pillow rectangle search. Burned-in camera text is masked out via
  `OSD_MASKS`.
- Videos play **once**. 1 frame is analysed, then N frames are skipped (sidebar, default 5).
- **best.pt** (the `best.pt.zip` file is the checkpoint itself and is copied to `best.pt`)
  has only 3 classes and was trained for 3 epochs, so its confidences are low. That's why
  the defaults are low.
- No colour or make/model weights were delivered. Colour is a Pillow/HSV heuristic.
- On CPU an analysed frame takes about 1.5 s with nothing in the zone, and up to ~4.5 s with
  several vehicles being read. A GPU build of onnxruntime/torch removes most of this.
- Plates are only readable when the vehicle is close (crop ≥ 120 px). Use **Photo check**
  with a close-up car photo to test OCR.
