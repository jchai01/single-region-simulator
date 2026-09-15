"""
Peat-soil area validation: compares your model's grassland/forestry
area on peat-influenced soil against Habib & Connolly (2023)'s
published national totals (357,900 ha grassland-on-peat, 224,700 ha
forest-on-peat), cited via the 2025 ScienceDirect follow-up paper.

Two peat-influence tiers, from the association/rank check:
  DOMINANT: association 01xx -- peat series occupy ranks 1, 2, AND 3,
    so this association's carbon estimate is genuinely peat-driven.
  PARTIAL: associations where a peat series appeared at rank 2 or 3
    (contributed to the estimate, but wasn't dominant) -- 0410b,
    0650a, 0700h, 0760f, 0843b, 0843f, 0900f.
  (Associations where peat only appeared at rank 4+ are excluded --
  confirmed to have had zero influence on that association's carbon
  estimate.)

Also includes LPIS's own explicit 'LOW INPUT PEAT GRASSLAND' class,
which is peat-labelled directly in the crop field regardless of which
soil association it falls in.
"""

import geopandas as gpd

DOMINANT_PEAT_ASSOC = {"01xx"}
PARTIAL_PEAT_ASSOC = {"0410b", "0650a",
                      "0700h", "0760f", "0843b", "0843f", "0900f"}

combined = gpd.read_file("data/parcel_yield_carbon.gpkg")
combined["area_ha"] = combined.geometry.area / 10_000
combined["crop_upper"] = combined["crop"].str.upper()


def area_on_peat(land_use_mask, assoc_set, label):
    subset = combined[land_use_mask & combined["Associat_S"].isin(assoc_set)]
    area = subset["area_ha"].sum()
    print(f"  {label}: {area:,.0f} ha ({len(subset)} parcels)")
    return area


print("=== Grassland on peat-influenced soil ===")
is_grass = combined["is_grassland"]
explicit_peat_grass = combined.loc[
    is_grass & (combined["crop_upper"] ==
                "LOW INPUT PEAT GRASSLAND"), "area_ha"
].sum()
print(
    f"  Explicit 'LOW INPUT PEAT GRASSLAND' (any soil): {explicit_peat_grass:,.0f} ha")

dom_grass = area_on_peat(is_grass, DOMINANT_PEAT_ASSOC,
                         "Dominant-peat association (01xx)")
partial_grass = area_on_peat(
    is_grass, PARTIAL_PEAT_ASSOC, "Partial-peat associations")

total_grass_dom_only = explicit_peat_grass + dom_grass
total_grass_dom_partial = explicit_peat_grass + dom_grass + partial_grass
print(f"  TOTAL (dominant only): {total_grass_dom_only:,.0f} ha")
print(f"  TOTAL (dominant + partial): {total_grass_dom_partial:,.0f} ha")
print(f"  Published (Habib & Connolly-derived): 357,900 ha")

print("\n=== Forestry on peat-influenced soil ===")
is_forest = combined["carbon_source"] == "nfi_blended_average"
dom_forest = area_on_peat(is_forest, DOMINANT_PEAT_ASSOC,
                          "Dominant-peat association (01xx)")
partial_forest = area_on_peat(
    is_forest, PARTIAL_PEAT_ASSOC, "Partial-peat associations")

print(f"  TOTAL (dominant only): {dom_forest:,.0f} ha")
print(f"  TOTAL (dominant + partial): {dom_forest + partial_forest:,.0f} ha")
print(f"  Published (Habib & Connolly-derived): 224,700 ha")
