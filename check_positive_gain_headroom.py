"""
Directly checks how much genuinely-positive-gain material exists in
the region under the corrected hybrid accounting -- distinguishes
"search isn't finding it" from "there genuinely isn't much more to
find" without needing to infer this from search behavior alone.
"""
import pandas as pd
from carbon_transition_delta import hybrid_transition_target_stock
from carbon_time_dynamics import carbon_at_year, YEARS_TO_POLICY_TARGET

REALISTIC_TARGET_CLASSES = ["tillage", "grassland", "forestry"]

def check_headroom(parcel_ids, combined, branch_means):
    region = combined[combined["parcel_id"].isin(parcel_ids)].copy()

    def best_gain(row):
        current = row["land_cover_class"]
        baseline = row["carbon_t_c_per_ha_mean"]
        if pd.isna(baseline):
            return None
        gains = {}
        for cls in REALISTIC_TARGET_CLASSES:
            if cls == current:
                continue
            target_stock = hybrid_transition_target_stock(current, cls, baseline, branch_means)
            realized = carbon_at_year(current, cls, baseline, target_stock, YEARS_TO_POLICY_TARGET)
            gains[cls] = (realized - baseline) * row["area_ha"]  # total t C, not per-ha
        return max(gains.values()) if gains else None

    region["best_gain_t"] = region.apply(best_gain, axis=1)
    positive = region[region["best_gain_t"] > 0]

    print(f"{len(region)} parcels total")
    print(f"{len(positive)} ({100*len(positive)/len(region):.1f}%) have ANY positive-gain transition available")
    print(f"Total positive-gain headroom if ALL were converted: {positive['best_gain_t'].sum():,.0f} t C")
    print(f"Total area of positive-gain parcels: {positive['area_ha'].sum():,.1f} ha "
          f"({100*positive['area_ha'].sum()/region['area_ha'].sum():.1f}% of region)")

if __name__ == "__main__":
    with open("data/offaly_subregion_parcel_ids.txt") as f:
        parcel_ids = [int(line.strip()) for line in f if line.strip()]
    import geopandas as gpd
    from carbon_transition_delta import classify_land_cover
    combined = gpd.read_file("data/parcel_yield_carbon.gpkg").reset_index(drop=True)
    combined["parcel_id"] = combined.index
    combined["area_ha"] = combined.geometry.area / 10_000
    combined["land_cover_class"] = combined.apply(classify_land_cover, axis=1)
    branch_means = combined.groupby("land_cover_class")["carbon_t_c_per_ha_mean"].mean().to_dict()
    from generate_scenarios import filter_reallocatable   # same candidate set every method uses
    parcel_ids = filter_reallocatable(parcel_ids, combined)
    check_headroom(parcel_ids, combined, branch_means)
