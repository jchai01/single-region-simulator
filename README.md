# single-region-simulator

### Dataset to gather

**CSO Area, Yield and Production of Crops**
https://data.cso.ie/
https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/AQA04/CSV/1.0/en
PxStat table, AQA04

**CSO Farm Structure Survey / Census of Agriculture** (county-level grassland/rough grazing/tillage hectares)
PxStat table IFS10, Land Utilisation table.
https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/IFS10/CSV/1.0/en

**LPIS parcels**
https://opendata.agriculture.gov.ie/dataset/?tags=LPIS
Download 2025 for now. Download as geopackage.

**Irish county boundaries** — OSi (Ordnance Survey Ireland) admin boundaries.
https://data-osi.opendata.arcgis.com/datasets/a2dd7924915e4c74ad6a417e33c394eb_1/explore?location=53.411151%2C-8.395890%2C7
Download as geopackage

**SIS Soil model**
https://gis.teagasc.ie/soils/downloads.php
Seems to have cert issues when accessing via browser, use:

```bash
wget --no-check-certificate https://gis.teagasc.ie/soils/downloads/INSM250k_ING_1b.zip
```

Check geopackage file with:

```bash
ogrinfo -so -al your_file.gpkg
```

### Training GNN

Dependencies (running with CUDA version 12.9)

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install torch_geometric pandas pyarrow fastparquet

# Installing pyg-lib and torch-sparse
uv pip install pyg-lib torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

