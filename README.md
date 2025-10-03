# Spirob Pipeline

This repository contains a pipeline to generate physics-ready **Spirob** models for MuJoCo.  
The pipeline automates the workflow:

1. **CSV generation** – computes element-wise geometry data from input parameters.
2. **STL generation** – builds 3D meshes of each link from the CSV.
3. **XML generation** – produces a MuJoCo-ready XML model referencing the STL meshes.

---

## 🚀 Usage

### Requirements
- Python 3.9+  
- [CadQuery](https://cadquery.readthedocs.io/)  
- NumPy, Pandas  
- MuJoCo (for testing XML models)  

Install dependencies:
```bash
pip install -r requirements.txt
````

---

### Run the pipeline

From the project root:

```bash
python build.py
```

By default:

* Cleans previous outputs
* Generates `Geom_Data_CSV/` (CSV files)
* Generates `meshes/` (STL files)
* Generates `spirob_physics_model.xml`

---

### Options

* Skip cleaning (keep old files):

```bash
python build.py --noclean
```

---

## 📂 Project Structure

```
Organized_Version/
│
├── build.py                 # Main driver script
├── spirob_csv_generator.py  # Generates CSV geometry
├── csv2geom.py              # Generates STL meshes
├── csv2xml.py               # Generates MuJoCo XML
├── helper_functions.py      # Shared math/geometry helpers
├── params.json              # Example input parameters
├── .gitignore               # Ignore rules
└── README.md                # Project description
```

---

## 📝 Notes

* Generated files (`Geom_Data_CSV/`, `meshes/`, and XML) are ignored by Git.
* Use `params.json` to control geometry parameters.
* MuJoCo XML uses STL meshes with consistent naming (`link_001.stl`, …).

---

## 📜 License

[MIT License](LICENSE)

````

---