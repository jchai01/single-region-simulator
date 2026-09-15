"""Inspect the raster's class legend before trusting PEATLAND_CLASS_CODES.
Checks rasterio tags/band descriptions for an embedded legend, and failing
that, prints unique pixel values with pixel counts + approximate hectares
so you can at least sanity-check which code is a plausible size for
"residual peatland" specifically (the class comparable to LPIS BOG) vs.
grassland/forestry/industrial-extraction (which would legitimately NOT
overlap with LPIS BOG parcels).
"""
import numpy as np
import rasterio

RASTER = "./data/Peatlands_Landuse_Rasters_Ireland/Landuse_2020.tif"

with rasterio.open(RASTER) as src:
    print("Dataset tags:", src.tags())
    print("Band 1 tags:", src.tags(1))
    print("Band 1 description:", src.descriptions[0])
    print("Colorinterp:", src.colorinterp)

    band = src.read(1)
    pixel_area_ha = abs(src.transform.a * src.transform.e) / 10_000
    vals, counts = np.unique(band, return_counts=True)
    print("\nPixel value : count : approx hectares")
    for v, c in zip(vals, counts):
        print(f"  {v:>6} : {c:>10} : {c * pixel_area_ha:>12,.1f} ha")
