"""
Raster-based validation -- avoids the polygon-overlay bottleneck entirely.

Rationale: the Habib & Connolly data is native raster. Rather than
polygonize -> vector overlay (expensive, hung on ~40k x 15k polygons),
rasterize LPIS BOG parcels onto the SAME grid as the H&C raster and
compare pixel arrays directly. This is the standard approach for
raster-vs-raster land-cover agreement (used throughout the remote-sensing
accuracy-assessment literature, including Habib & Connolly's own
accuracy assessment methodology) and is dramatically faster.
"""

import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd

HABIB_CONNOLLY_RASTER = "./data/Peatlands_Landuse_Rasters_Ireland/Landuse_2020.tif"
RESIDUAL_PEATLAND_CODE = 4  # confirmed against Table 5 of the paper

LPIS_SOURCE = "data/parcel_carbon_lookup.gpkg"
BOG_CARBON_T_PER_HA = 705.0

OUT_METRICS = "./data/validation_metrics_raster.csv"


def main():
    with rasterio.open(HABIB_CONNOLLY_RASTER) as src:
        hc_band = src.read(1)
        transform = src.transform
        raster_crs = src.crs
        out_shape = src.shape
        pixel_area_ha = abs(transform.a * transform.e) / 10_000

    print(f"Raster grid: {out_shape}, {pixel_area_ha:.4f} ha/pixel, CRS={raster_crs}")

    lpis = gpd.read_file(LPIS_SOURCE)
    bog = lpis[lpis["crop"].str.upper() == "BOG"].copy()
    if bog.crs != raster_crs:
        bog = bog.to_crs(raster_crs)
    print(f"LPIS BOG parcels: {len(bog)}")

    # Burn BOG parcels onto the SAME grid as the raster -- this is the key
    # step, guarantees pixel-for-pixel alignment, no polygon overlay needed
    print("Rasterizing LPIS BOG parcels onto H&C grid...")
    bog_mask = rasterize(
        [(geom, 1) for geom in bog.geometry if geom is not None and not geom.is_empty],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    hc_mask = (hc_band == RESIDUAL_PEATLAND_CODE)
    bog_mask = bog_mask.astype(bool)

    agreement_px = np.count_nonzero(hc_mask & bog_mask)
    lpis_only_px = np.count_nonzero(bog_mask & ~hc_mask)
    hc_only_px = np.count_nonzero(hc_mask & ~bog_mask)
    union_px = agreement_px + lpis_only_px + hc_only_px

    agreement_ha = agreement_px * pixel_area_ha
    lpis_only_ha = lpis_only_px * pixel_area_ha
    hc_only_ha = hc_only_px * pixel_area_ha

    iou = agreement_ha / (agreement_ha + lpis_only_ha + hc_only_ha)
    dice = 2 * agreement_ha / (2 * agreement_ha + lpis_only_ha + hc_only_ha)

    print(f"\nagreement_ha:  {agreement_ha:,.1f}")
    print(f"lpis_only_ha:  {lpis_only_ha:,.1f}  (overclaim)")
    print(f"hc_only_ha:    {hc_only_ha:,.1f}  (missed)")
    print(f"IoU:           {iou:.4f}")
    print(f"Dice:          {dice:.4f}")
    print(f"agreement_tC:  {agreement_ha * BOG_CARBON_T_PER_HA:,.0f}")
    print(f"overclaim_tC:  {lpis_only_ha * BOG_CARBON_T_PER_HA:,.0f}")
    print(f"missed_tC:     {hc_only_ha * BOG_CARBON_T_PER_HA:,.0f}")

    import pandas as pd
    pd.DataFrame([{
        "agreement_ha": agreement_ha, "lpis_only_ha_overclaim": lpis_only_ha,
        "habib_connolly_only_ha_missed": hc_only_ha, "IoU": iou, "dice": dice,
        "agreement_tC": agreement_ha * BOG_CARBON_T_PER_HA,
        "overclaim_tC": lpis_only_ha * BOG_CARBON_T_PER_HA,
        "missed_tC": hc_only_ha * BOG_CARBON_T_PER_HA,
    }]).to_csv(OUT_METRICS, index=False)
    print(f"\nWrote {OUT_METRICS}")


if __name__ == "__main__":
    main()
