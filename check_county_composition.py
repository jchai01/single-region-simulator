"""
Run this before committing to a case-study county/region -- checks the
ACTUAL land-cover-class composition for a given county in the combined
dataset, rather than assuming from general knowledge of the county.
"""
import geopandas as gpd
from carbon_transition_delta import classify_land_cover

COMBINED_PATH = "data/parcel_yield_carbon.gpkg"
CANDIDATE_COUNTIES = ["CORK", "OFFALY"]  # add/remove as needed

def check_composition(county_name, gdf):
    county = gdf[gdf["COUNTY"].str.upper() == county_name.upper()]
    if len(county) == 0:
        print(f"{county_name}: no parcels found -- check COUNTY value spelling/casing in your data")
        return
    counts = county["land_cover_class"].value_counts()
    areas = county.groupby("land_cover_class")["area_ha"].sum()
    print(f"\n=== {county_name}: {len(county):,} parcels, {county['area_ha'].sum():,.0f} ha total ===")
    for cls in counts.index:
        print(f"  {cls}: {counts[cls]:,} parcels, {areas[cls]:,.0f} ha "
              f"({100*counts[cls]/len(county):.1f}% of parcels)")

if __name__ == "__main__":
    gdf = gpd.read_file(COMBINED_PATH)
    gdf["land_cover_class"] = gdf.apply(classify_land_cover, axis=1)
    gdf["area_ha"] = gdf.geometry.area / 10_000
    for county in CANDIDATE_COUNTIES:
        check_composition(county, gdf)
