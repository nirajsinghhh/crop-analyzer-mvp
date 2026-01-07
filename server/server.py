from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import Polygon, mapping
from shapely.ops import transform
from pyproj import Transformer
from pystac_client import Client
import planetary_computer as pc

app = Flask(__name__)
CORS(app)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def classify_zone(ndvi, evi, ndwi, msi):
    if ndwi > 0.3 and ndvi < 0.1:
        return "Water body", "Surface water detected (pond, lake, canal, or flooded area)"

    if ndvi < 0.15 and ndwi < -0.2 and evi < 0.1:
        return "Built-up / Urban area", "Non-vegetated built-up surface (buildings, roads, or infrastructure)"

    if ndvi < 0.2 and ndwi > -0.2:
        return "Bare soil / Fallow land", "Exposed soil or recently harvested field"

    if ndvi < 0.35:
        if msi > 1.5:
            return "Vegetation", "Sparse vegetation with high water stress"
        elif ndwi < -0.2:
            return "Vegetation", "Sparse vegetation with moisture deficiency"
        else:
            return "Vegetation", "Early growth stage or sparse vegetation"

    if ndvi < 0.55:
        if msi > 1.8:
            return "Vegetation", "Moderate vegetation with severe water stress"
        elif ndwi < -0.2:
            return "Vegetation", "Moderate vegetation with moisture stress"
        elif evi < ndvi:
            return "Vegetation", "Moderate vegetation with soil background influence"
        else:
            return "Vegetation", "Moderate and stable vegetation health"

    if msi > 1.5:
        return "Vegetation", "Dense vegetation under emerging water stress"
    elif ndwi < -0.1:
        return "Vegetation", "Healthy canopy with declining moisture"
    else:
        return "Vegetation", "Healthy and dense vegetation"


def compute_indices(bands, polygon):
    arrays = {}

    with rasterio.open(bands["B08"]) as ref_src:
        transformer = Transformer.from_crs("EPSG:4326", ref_src.crs, always_xy=True)
        projected_polygon = transform(transformer.transform, polygon)
        ref_img, ref_transform = mask(ref_src, [mapping(projected_polygon)], crop=True)
        ref = ref_img.astype("float32") / 10000.0
        ref_crs = ref_src.crs
        ref_shape = ref.shape
        arrays["B08"] = ref

    for band in ["B02", "B03", "B04", "B11"]:
        with rasterio.open(bands[band]) as src:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            projected_polygon = transform(transformer.transform, polygon)
            img, src_transform = mask(src, [mapping(projected_polygon)], crop=True)
            data = img.astype("float32")
            resampled = np.zeros(ref_shape, dtype="float32")

            reproject(
                source=data,
                destination=resampled,
                src_transform=src_transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )

            arrays[band] = resampled / 10000.0

    nir = arrays["B08"]
    red = arrays["B04"]
    green = arrays["B03"]
    blue = arrays["B02"]
    swir = arrays["B11"]

    ndvi = (nir - red) / (nir + red + 1e-6)
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    ndwi = (green - nir) / (green + nir + 1e-6)

    valid_mask = (nir > 0.1) & (swir > 0.01)
    msi = np.where(valid_mask, swir / nir, np.nan)

    return {
        "NDVI": float(np.nanmean(ndvi)),
        "EVI": float(np.nanmean(evi)),
        "NDWI": float(np.nanmean(ndwi)),
        "MSI": float(np.nanmean(msi)),
        "RED": float(np.nanmean(red)),
        "GREEN": float(np.nanmean(green)),
        "NIR": float(np.nanmean(nir)),
    }


@app.route("/analyze-farm", methods=["POST"])
def analyze_farm():
    try:
        data = request.get_json()
        coords = data["coordinates"][0]
        polygon = Polygon(coords)

        catalog = Client.open(STAC_URL)
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=mapping(polygon),
            query={"eo:cloud_cover": {"lt": 20}},
        )

        items = list(search.get_items())
        if not items:
            return jsonify({"error": "No satellite data found"}), 404

        item = pc.sign(items[0])

        bands = {
            "B02": item.assets["B02"].href,
            "B03": item.assets["B03"].href,
            "B04": item.assets["B04"].href,
            "B08": item.assets["B08"].href,
            "B11": item.assets["B11"].href,
        }

        indices = compute_indices(bands, polygon)

        zone_type, health_status = classify_zone(
            indices["NDVI"],
            indices["EVI"],
            indices["NDWI"],
            indices["MSI"],
        )

        return jsonify({
            "indices": {k: round(v, 4) for k, v in indices.items()},
            "zone_type": zone_type,
            "health_status": health_status
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
