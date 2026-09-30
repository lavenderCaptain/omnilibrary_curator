# System Prompt & Project Blueprint: Omnilibrary AI Curation App

## Core Objective
Build a native, zero-bloat macOS desktop app (Python 3.11+ / PySide6) that scans a local photo library on an M2 Studio, builds persistent vector/EXIF indexes using DuckDB, auto-clusters photos into events, detects duplicate/derivative images, scores aesthetics, and presents results in a hardware-accelerated GUI.

---

## Strict Development Constraints
1. **NO Web Frameworks:** Strictly NO Electron, Chromium, Node.js, React, or local web servers. Use **PySide6 (Qt 6 for Python)** for the GUI.
2. **Hardware Acceleration:** All AI model inferences (DINOv2, CLIP, pHash) MUST target Apple Silicon MPS GPU (`torch.device("mps")`).
3. **Non-Destructive Operations:** NEVER modify, move, or write directly to source image files. All metadata, event assignments, ratings, and crop coordinates are written ONLY to DuckDB and exported via standard `.xmp` sidecar files.
4. **Threading Rules:** Heavy CPU/GPU workloads (scanning files, extracting embeddings, database writes) MUST run inside background threads (`QThread` / `QThreadPool`). The PySide6 UI loop MUST never drop frames.
5. **RAW Performance Optimization:** Never decode full-res RAW files for thumbnailing/embedding. Extract embedded JPEGs from RAW headers (`exiftool`/`rawpy`) for vector calculations and preview rendering.

---

## Technical Stack Architecture
* **Language:** Python 3.11+
* **GUI Engine:** PySide6 (`QListView`, custom `QStyledItemDelegate`, `QThread`)
* **Vector & Metadata DB:** DuckDB (`duckdb` with vector similarity search)
* **Computer Vision Core:** PyTorch with MPS backend (`torch`, `torchvision`, `dinov2`)
* **Perceptual Hashing:** `imagehash` (pHash / dHash) + `Pillow`
* **EXIF Extraction:** `exiftool` / `pyexiftool` / `rawpy`

---

## Application Structure

omnilibrary_curator/
├── engine/
│   ├── __init__.py
│   ├── scanner.py        # Fast file tree scanner & EXIF/RAW JPEG extractor
│   ├── embeddings.py     # MPS PyTorch worker (DINOv2 / pHash / Aesthetics)
│   ├── database.py       # DuckDB interface (Metadata & Vector DB)
│   ├── clustering.py     # HDBSCAN / Multi-signal (Time + GPS + Camera Body + DINOv2)
│   ├── dedup.py          # Multi-stage duplicate finder (pHash -> DINOv2)
│   └── naming.py         # Event title generation from metadata & visual tags
├── ui/
│   ├── __init__.py
│   ├── main_window.py    # PySide6 Shell with sidebar and main grid
│   ├── photo_grid.py     # Hardware-accelerated QListView + Delegate
│   └── views/
│       ├── event_view.py # Event cluster browser
│       └── dup_view.py   # Side-by-side duplicate comparison inspector
├── xmp/
│   └── writer.py         # Non-destructive XMP sidecar generator
├── tests/                # PyTest suite
├── app.py                # Application entrypoint
├── CLAUDE.md
└── requirements.txt

---

## Detailed Functional Specifications & Phased Plan

### Phase 1: Storage Layer & Asset Scanner (`engine/database.py`, `engine/scanner.py`)
* **DuckDB Schema:** Create `media_items` table storing: `file_path`, `file_size`, `file_format`, `camera_make`, `camera_model`, `camera_serial`, `timestamp_utc`, `gps_lat_lon`, `p_hash`, `dinov2_vector` (768d), `aesthetic_score`, `crop_rect`, `color_fix_flag`, `event_id`, `duplicate_group_id`, `is_primary_master`.
* **Fast RAW/EXIF Scanner:** Extract metadata and pull embedded JPEGs from RAW headers without loading full raw payloads into RAM.

### Phase 2: Vector Engine, Clustering & Deduplication (`engine/embeddings.py`, `engine/clustering.py`, `engine/dedup.py`)
* **MPS PyTorch Worker:** Batch-process embeddings via DINOv2 (`dinov2_vits14`) on M2 MPS.
* **Camera Hardware Signatures:** Group photos by `Make + Model + Serial`. Never merge photos taken on the same date by different camera bodies unless GPS or visual embeddings prove spatial co-location.
* **Multi-Signal Clustering:** Combine time, GPS, camera body, and DINOv2 embeddings using HDBSCAN. Anchor EXIF-less/GPS-less photos to visual clusters.
* **2-Tier Deduplication & Quality Scoring:** 
  1. Fast `pHash` Hamming distance scan.
  2. DINOv2 vector cosine similarity verification.
  3. Quality heuristics: Tag "Master" vs "Derivative" based on RAW format > JPG, resolution, file size, and EXIF completeness.

### Phase 3: Hardware-Accelerated GUI (`ui/`)
* **PySide6 Layout:**
  * **Sidebar:** Tree navigation for Events (auto-named), Camera Bodies, and Duplicate Stacks.
  * **Main Canvas:** Hardware-accelerated `QListView` with custom delegate rendering star ratings, AI crop boxes (rule-of-thirds), and "Duplicate Derivative" flags.
* **Duplicate Inspector:** Side-by-side view comparing suspected duplicates with resolution/EXIF metadata to let the user confirm or swap Master designation.

### Phase 4: XMP Export Bridge (`xmp/writer.py`)
* Write human-approved event titles (to `IPTC:Headline` / `XMP:Event`), hierarchical keywords (`Events|Japan 2024|Day 01`), 1–5 star ratings, and duplicate flags out to standard `.xmp` sidecar files for Lightroom ingestion.

---

## Instructions for the AI Agent
* Build the architecture incrementally phase by phase.
* Write unit tests in `tests/` and verify execution for Phase 1 before proceeding to Phase 2.
* Ensure all heavy compute calls run off the PySide6 main thread using `QThread`/`QThreadPool`.
