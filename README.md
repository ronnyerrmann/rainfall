Downloads the metoffice rainfall radar images, summarises them daily, and creates time laps videas for selected areas

### Requirements
```
python -m venv rain_venv
. rain_venv/bin/activate
pip install Image requests
pip install opencv-python
```

### Run
```
rain_venv/bin/python rainfall_images.py
```
All data will be stored in the folder of the script.

As data only a few days back is available, the script should be run at least every 24 hours. For courtesy, do not run it more then once per hour.

### Description
For the daily images the same scale as [metoffice uses in their rainfall radar](https://weather.metoffice.gov.uk/maps-and-charts/rainfall-radar-forecast-map#?model=ukmo-ukv&layer=rainfall-rate) is used, but instead of indicating mm/h it's mm/day. Additionally purple for 64 to 128 and white for 128 to 999 where added. If more than that value is reached, black will be shown and you should build am arch.

### Add your own areas
This can be done either by creating a file `subareas.csv` in the script folder, with comma-separated values for
1. (`int`) x-Pixel position in the original file in `image_data`, starting from 0. In the future, that should be derived from lat/lon but for now I used Irfanview to see the pixel position during a click (which is already 0 based). To know which pixel, I looked for rain reaching the position on the metoffice website 
2. (`int`) y-Pixel position in the original file in `image_data`.
3. longitude
4. latitude
5. semi-size of the box around the selected pixel, e.g. 1 leads to a 3x3 array, 10 to 21x21
6. scale of a pixel in the final image, e.g. a scale of 5 on a 3x3 array leads to a 15x15 png
7. name of the place
Alternatively, the script can be edited to add the lines containing `SUBAREAS.append` with the same entries (name needs to be in quotes)
