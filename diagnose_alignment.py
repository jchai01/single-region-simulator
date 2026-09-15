"""Run this against your actual files to check for a spatial alignment bug
before trusting the IoU number. Prints the raster's ORIGINAL declared CRS
(before any reprojection) and the post-reprojection bounding boxes of both
layers -- if the two bounding boxes don't substantially overlap, that's
your answer.
"""
import rasterio
import geopandas as gpd

HABIB_CONNOLLY_RASTER = "./data/Peatlands_Landuse_Rasters_Ireland/Landuse_2020.tif"
HC_POLYGONIZED_GPKG = "./data/habib_connolly_peatland_polygons.gpkg"
LPIS_SOURCE = "./data/parcel_carbon_lookup.gpkg"

with rasterio.open(HABIB_CONNOLLY_RASTER) as src:
    print("Raster's ORIGINAL declared CRS (before reprojection):", src.crs)
    print("Raster bounds (native CRS):", src.bounds)

hc = gpd.read_file(HC_POLYGONIZED_GPKG)
print("\nHabib & Connolly polygons, post-reprojection to", hc.crs)
print("  bounds (ITM):", hc.total_bounds)

lpis = gpd.read_file(LPIS_SOURCE)
bog = lpis[lpis["crop"].str.upper() == "BOG"]
print("\nLPIS BOG parcels, CRS:", bog.crs)
print("  bounds (ITM):", bog.total_bounds)

print("\nFor reference, Ireland's extent in ITM (EPSG:2157) is roughly:")
print("  x: 20,000 to 380,000   y: 20,000 to 470,000")
