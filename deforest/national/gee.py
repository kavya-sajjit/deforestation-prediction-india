"""All-India 640 m grid from Google Earth Engine, exported to Google Drive.

Earth Engine does the heavy part (reading 30 m Hansen and the other layers for the whole country)
and writes three things to the Drive folder `deforestation-india`:

  india_loss_640m*.tif   band f2000 = share of the square that was forest in 2000 (canopy >= 30%),
                         bands l2001..l2025 = share of the square lost in each year. Stored x10000 as
                         int16, so 2500 means 25%.
  india_cov_640m*.tif    factors that are not the forest itself (see COVARIATES below), float32.
  india_districts.geojson + india_admin_codes.csv   district shapes and names, for the website.

Every square is 640 m x 640 m in an equal-area projection (EPSG:6933), so each one is 40.96 ha
anywhere in India. Large exports are split by Earth Engine into several .tif files; table.py
stitches them back together.

Nothing here is computed in Colab, so a free Colab session is enough. The exports run on Earth
Engine's side after you start them, and you can close the notebook while they run.

    import ee; ee.Authenticate(); ee.Initialize(project="your-project-id")
    from deforest.national import gee
    tasks = gee.start_exports()
    gee.status(tasks)
"""
from __future__ import annotations

import time
from typing import Dict, List

import ee

HANSEN = "UMD/hansen/global_forest_change_2025_v1_13"
LAST_YEAR = 2025
CRS = "EPSG:6933"          # equal-area: every square has the same size everywhere
SCALE = 640                # metres
TREECOVER = 30             # % canopy in 2000 to count as forest (same as the regional pipeline)
FOLDER = "deforestation-india"

# What each covariate band means. All are measured BEFORE 2019 so the 2020-23 forecast cannot
# peek at the future; only the protected-area layer is filtered by designation year for that.
COVARIATES = {
    "state": "GAUL state code (0 = outside India)",
    "district": "GAUL district code",
    "elevation": "m above sea level (SRTM)",
    "slope": "degrees (SRTM)",
    "access_min": "travel time to the nearest city of 50,000+ people, minutes (2015)",
    "pop": "people living in the square (WorldPop 2015)",
    "lights": "median night-time light radiance 2015-2018 (VIIRS)",
    "fire_days": "days with an active fire detected 2013-2018 (MODIS FIRMS)",
    "rain_mm": "mean annual precipitation 2013-2018, mm (ERA5-Land)",
    "protected": "1 if the square centre lies in a protected area designated by 2018 (WDPA)",
}


def india_states() -> ee.FeatureCollection:
    """States and union territories from FAO GAUL 2015.

    GAUL draws some areas (parts of Jammu & Kashmir, Ladakh, Arunachal Pradesh) as separate,
    disputed units, so they are added explicitly. Check `print_units()` once: a map published in
    India must follow the official Survey of India boundary, which GAUL does not.
    """
    g = ee.FeatureCollection("FAO/GAUL/2015/level1")
    return g.filter(ee.Filter.Or(
        ee.Filter.eq("ADM0_NAME", "India"),
        ee.Filter.stringContains("ADM0_NAME", "Kashmir"),
        ee.Filter.eq("ADM0_NAME", "Arunachal Pradesh"),
        ee.Filter.eq("ADM0_NAME", "Aksai Chin"),
    ))


def india_districts() -> ee.FeatureCollection:
    g = ee.FeatureCollection("FAO/GAUL/2015/level2")
    return g.filter(ee.Filter.inList("ADM1_CODE", india_states().aggregate_array("ADM1_CODE")))


def print_units():
    """Which GAUL country names were picked up, and how many states. Run once and read it."""
    s = india_states()
    print("GAUL country names included:", s.aggregate_array("ADM0_NAME").distinct().getInfo())
    print("states / UTs:", s.size().getInfo(), "| districts:", india_districts().size().getInfo())


def _grid(img: ee.Image, reducer=None) -> ee.Image:
    """Average (or sum) fine pixels into 640 m squares."""
    reducer = reducer or ee.Reducer.mean()
    return img.reduceResolution(reducer=reducer, maxPixels=4096).reproject(crs=CRS, scale=SCALE)


def loss_image(last_year: int = LAST_YEAR) -> ee.Image:
    h = ee.Image(HANSEN)
    forest = h.select("treecover2000").gte(TREECOVER)
    ly = h.select("lossyear")
    bands = [forest.rename("f2000")]
    for y in range(1, last_year - 2000 + 1):
        bands.append(forest.And(ly.eq(y)).rename(f"l{2000 + y}"))
    stack = ee.Image.cat(bands).unmask(0).toFloat()
    return _grid(stack).multiply(10000).round().toInt16()


def _with_proj(col: ee.ImageCollection, img: ee.Image) -> ee.Image:
    """Composites lose their projection; give them the collection's native one back."""
    return img.setDefaultProjection(col.first().projection())


def covariate_image() -> ee.Image:
    states, districts = india_states(), india_districts()
    state = ee.Image(0).int().paint(states, "ADM1_CODE").rename("state")
    district = ee.Image(0).int().paint(districts, "ADM2_CODE").rename("district")

    srtm = ee.Image("USGS/SRTMGL1_003")
    elevation = _grid(srtm.select("elevation").toFloat()).rename("elevation")
    slope = _grid(ee.Terrain.slope(srtm)).rename("slope")

    access = ee.Image("Oxford/MAP/accessibility_to_cities_2015_v1_0").select("accessibility").rename("access_min")

    wp = ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq("country", "IND")) \
        .filter(ee.Filter.eq("year", 2015))
    pop = _grid(_with_proj(wp, wp.mosaic()).select("population"), ee.Reducer.sum()).rename("pop")

    vi = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate("2015-01-01", "2019-01-01").select("avg_rad")
    lights = _with_proj(vi, vi.median()).rename("lights")

    fi = ee.ImageCollection("FIRMS").filterDate("2013-01-01", "2019-01-01").select("T21")
    fire = _with_proj(fi, fi.map(lambda i: i.gt(0)).sum()).rename("fire_days")

    era = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate("2013-01-01", "2019-01-01") \
        .select("total_precipitation_sum")
    rain = _with_proj(era, era.mean()).multiply(12 * 1000).rename("rain_mm")      # m/month -> mm/year

    pa = ee.FeatureCollection("WCMC/WDPA/current/polygons").filterBounds(states.geometry().bounds()) \
        .filter(ee.Filter.lte("STATUS_YR", 2018))
    protected = ee.Image(0).byte().paint(pa, 1).rename("protected")

    img = ee.Image.cat([state, district, elevation, slope, access, pop, lights, fire, rain, protected])
    return img.reproject(crs=CRS, scale=SCALE).unmask(0).toFloat().select(list(COVARIATES))


def start_exports(folder: str = FOLDER, last_year: int = LAST_YEAR) -> List[ee.batch.Task]:
    """Queue the exports on Earth Engine. They keep running if you close Colab."""
    region = india_states().geometry().bounds()
    common = dict(folder=folder, region=region, crs=CRS, scale=SCALE, maxPixels=1e11, fileFormat="GeoTIFF")
    districts = india_districts().map(lambda f: f.simplify(500))
    tasks = [
        ee.batch.Export.image.toDrive(image=loss_image(last_year), description="india_loss_640m",
                                      fileNamePrefix="india_loss_640m", **common),
        ee.batch.Export.image.toDrive(image=covariate_image(), description="india_cov_640m",
                                      fileNamePrefix="india_cov_640m", **common),
        ee.batch.Export.table.toDrive(
            collection=districts.select(["ADM1_CODE", "ADM1_NAME", "ADM2_CODE", "ADM2_NAME"]),
            description="india_districts", folder=folder, fileNamePrefix="india_districts", fileFormat="GeoJSON"),
        ee.batch.Export.table.toDrive(
            collection=india_districts().select(["ADM0_NAME", "ADM1_CODE", "ADM1_NAME", "ADM2_CODE", "ADM2_NAME"],
                                                None, False),
            description="india_admin_codes", folder=folder, fileNamePrefix="india_admin_codes", fileFormat="CSV"),
    ]
    for t in tasks:
        t.start()
    print(f"Started {len(tasks)} exports into Drive folder '{folder}'. Watch them with gee.status(tasks) "
          "or in the Tasks tab at https://code.earthengine.google.com")
    return tasks


def recent(n: int = 8):
    """Your latest Earth Engine exports, for when Colab restarted and `tasks` is gone."""
    for t in ee.batch.Task.list()[:n]:
        st = t.status()
        print(f"{st['description']:<22} {st['state']:<10} {st.get('error_message', '')}")


def status(tasks: List[ee.batch.Task], wait: bool = False, every: int = 120) -> Dict[str, str]:
    """Print each export's state; with wait=True, keep checking until all have stopped."""
    while True:
        s = {t.status()["description"]: t.status()["state"] for t in tasks}
        print(time.strftime("%H:%M"), s)
        if not wait or all(v in ("COMPLETED", "FAILED", "CANCELLED") for v in s.values()):
            for t in tasks:
                st = t.status()
                if st["state"] == "FAILED":
                    print(f"  {st['description']} failed: {st.get('error_message')}")
            return s
        time.sleep(every)
