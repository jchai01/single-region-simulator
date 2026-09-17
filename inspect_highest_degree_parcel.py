"""
Determines whether parcel_id 1537631 (99.84ha, 'Permanent Pasture',
Mayo, degree=1404) is a genuine large/irregular commonage-type parcel,
or a geometry data-quality problem (invalid polygon, scattered
MultiPolygon, digitization noise) -- these look identical from degree
count alone, only the geometry itself tells them apart.
"""
import geopandas as gpd

COMBINED_PATH = "data/parcel_yield_carbon.gpkg"
TARGET_PARCEL_ID = 1537631

gdf = gpd.read_file(COMBINED_PATH).reset_index(drop=True)
gdf["parcel_id"] = gdf.index
row = gdf[gdf["parcel_id"] == TARGET_PARCEL_ID].iloc[0]
geom = row.geometry

print(f"Geometry type: {geom.geom_type}")
print(f"Is valid: {geom.is_valid}")
if not geom.is_valid:
    from shapely.validation import explain_validity
    print(f"Validity issue: {explain_validity(geom)}")

print(f"Area: {geom.area / 10_000:.2f} ha")
print(f"Perimeter/boundary length: {geom.length:.1f} m")
bounds = geom.bounds
bbox_w, bbox_h = bounds[2]-bounds[0], bounds[3]-bounds[1]
print(f"Bounding box: {bbox_w:.1f}m x {bbox_h:.1f}m ({bbox_w*bbox_h/10_000:.1f} ha)")
print(f"Area / bbox area ratio: {geom.area / (bbox_w*bbox_h):.3f} "
      "(low ratio = sprawling/scattered/elongated shape; ~1.0 = compact rectangle-like)")

if geom.geom_type == "MultiPolygon":
    parts = list(geom.geoms)
    print(f"\nMultiPolygon with {len(parts)} separate parts:")
    for i, part in enumerate(parts[:20]):
        print(f"  part {i}: area={part.area/10_000:.3f}ha, valid={part.is_valid}")
    if len(parts) > 20:
        print(f"  ... and {len(parts)-20} more parts")

# Perimeter-to-area ratio compared to a simple compact shape of the same
# area, as an irregularity signal (a circle of the same area is the
# minimum-perimeter reference)
import math
circle_equiv_perimeter = 2 * math.sqrt(math.pi * geom.area)
print(f"\nPerimeter vs. equivalent-area circle: {geom.length:.0f}m actual vs "
      f"{circle_equiv_perimeter:.0f}m for a compact circle of the same area "
      f"(ratio {geom.length/circle_equiv_perimeter:.1f}x -- high ratio = very irregular/winding boundary)")
