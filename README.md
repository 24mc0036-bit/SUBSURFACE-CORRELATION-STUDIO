# SUBDURFACE-CORRELATION-STUDIO
# Multi-Well Viewer

An interactive **Streamlit-based application** designed for loading, visualizing, and analyzing multiple well-log datasets. The tool supports formation tops visualization, well correlation, depth alignment, and statistical analysis in a single workflow.

---

## Key Features

* **Multi-LAS Import**
  Upload and analyze multiple LAS well-log files at the same time.

* **Cross-Well Visualization**
  Display configurable log tracks such as:

  * Gamma Ray (GR)
  * Resistivity (ILD)
  * Density (RHOB)
  * Neutron Porosity (NPHI)
  * Sonic (DT)

* **Formation Tops Overlay**
  Import formation picks from CSV files and visualize them as labeled, color-coded markers across wells.

* **Depth Flattening / Alignment**
  Align wells using a selected reference formation top for improved stratigraphic correlation.

* **Statistical Analysis**
  Generate per-well statistics and histograms for selected log curves.

* **Export Support**
  Save generated cross-sections as high-quality PNG or PDF files.

---

# Repository Structure

```bash
multi-well-viewer/
├── app.py                # Main Streamlit interface
├── data_loader.py        # LAS and CSV loading utilities
├── visualizer.py         # Plotting and visualization functions
├── flattening.py         # Formation-based depth alignment logic
├── requirements.txt      # Required Python packages
└── sample_data/
    ├── WELL_A.las
    ├── WELL_B.las
    ├── WELL_C.las
    └── formation_tops.csv
```

---

# Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/AnshulChoudhary-iitism/multi-well-viewer.git
cd multi-well-viewer
```

---

## 2. Install Dependencies

Python **3.9 or later** is recommended.

```bash
pip install -r requirements.txt
```

---

## 3. Launch the Application

```bash
streamlit run app.py
```

The application will start locally at:

```bash
http://localhost:8501
```

---

# Quick Demo Using Sample Data

1. Open the sidebar and upload all LAS files from the `sample_data/` directory.
2. Upload the `formation_tops.csv` file.
3. The application automatically generates the multi-well cross-section with selected log tracks.
4. Enable **Flattening** and select a reference horizon such as **Top Sand** to align wells stratigraphically.

---

# Formation Tops CSV Format

```csv
well,formation,depth[,color]
WELL_A,Top Sand,1200.0,#3cb44b
WELL_B,Top Sand,1185.0,#3cb44b
```

### Notes

* `color` is optional.
* If no color is provided, the application assigns colors automatically.

---

# Tech Stack

| Library    | Purpose                               |
| ---------- | ------------------------------------- |
| Streamlit  | Interactive web application framework |
| lasio      | LAS file parsing                      |
| Pandas     | Data processing and manipulation      |
| NumPy      | Numerical computations                |
| Matplotlib | Scientific plotting and visualization |

---

# Contributions

Contributions, feature suggestions, and improvements are welcome.
For major updates or architectural changes, please open an issue before submitting a pull request.
