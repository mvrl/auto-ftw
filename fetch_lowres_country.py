import geopandas as gpd
import mgrs
import numpy as np
import requests
import os

from pystac_client import Client
from shapely.geometry import Polygon, Point, box

def get_mrgs_tiles_for_a_country(country):
    
    PRECISION               = 0.1
    COUNTRIES_POLYGON_URL   = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    countries_polygons      = gpd.read_file(COUNTRIES_POLYGON_URL)  
    country_geometry        = countries_polygons[countries_polygons.SOVEREIGNT == country].get_geometry(0).iloc[0]
    
    min_x, min_y, max_x, max_y = country_geometry.bounds

    mrgs_tiles  = set() 
    for x in np.arange(min_x, max_x, PRECISION):
        for y in np.arange(min_y, max_y, PRECISION):
            point = Point(x,y)
            if country_geometry.contains(point):
                mrgs_tile   = mgrs.MGRS().toMGRS(y,x, MGRSPrecision=5)
                mrgs_ref    = f"{mrgs_tile[:5]}"
                mrgs_tiles.add(mrgs_ref)

    return list(mrgs_tiles)



countries = ["Latvia", "Estonia", "Lithuania", "Vietnam", "Cambodia"]
country_mrgs_tiles  = [ set(get_mrgs_tiles_for_a_country(country)) for country in countries  ]
country_mrgs_tiles  = list(set.union(*country_mrgs_tiles))



catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

search = catalog.search(
    collections=["sentinel-2-l2a"],
    datetime="2021-01-01/2021-12-31",
    limit=1,
    query={"s2:mgrs_tile": {"in": country_mrgs_tiles}, "eo:cloud_cover": {"lt": 50}},
)
search_results  = search.get_all_items()
folder_path     = "/data/p.vinh/Auto-FTW/Lowres-Images"
os.makedirs("/data/p.vinh/Auto-FTW/Lowres-Images", exist_ok=True)


for result in search_results:
    image_href = result.assets['rendered_preview'].href
    try:
        response = requests.get(image_href, stream=True)
        response.raise_for_status()
        image_id = result.id
        output_filename = f"{folder_path}/{image_id}.png"
        with open(output_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        
        print(f"Image successfully downloaded as '{output_filename}'")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")

    


