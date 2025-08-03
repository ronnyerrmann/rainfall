Downloads the metoffice rainfall radar images, summarises them daily, and creates time laps videos for selected areas.\
If `pdflatex` is installed, it will also create pdf with the daily image for the selected areas to provide a nice overview.
In the pdfs the centre of the images is marked with a small white dot.

### Installation/Requirements
```
git clone https://github.com/ronnyerrmann/rainfall
cd rainfall
python -m venv rain_venv
. rain_venv/bin/activate
pip install Image requests opencv-python
```

Add crontab to run automatically every 8 hours:
```
1 */8 * * * /home/<user>/rainfall/rain_venv/bin/python /home/<user>/rainfall/rainfall_images.py
```

### Run
```
rain_venv/bin/python rainfall_images.py
```
All data will be stored in the folder of the script.

As data only a few days back is available, the script should be run at least every 24 hours. For courtesy, do not run it more then once per hour.

### Description
For the daily images the same scale as [metoffice uses in their rainfall radar](https://weather.metoffice.gov.uk/maps-and-charts/rainfall-radar-forecast-map#?model=ukmo-ukv&layer=rainfall-rate) is used, but instead of indicating mm/h it's mm/day. Additionally purple for 64 to 128 and white for 128 to 999 mm/day where added. If more than that value is reached, black will be shown and you should build am arch.

As the publicly available data only provides ranges from `x` to `2x` mm/h (with `x` being the lower rainfall limit for a colour), the resulting summaries for the day can be off by a similar amount. The program uses the value at 33% of the range to reflect the assumed Poisson distribution of rainfall and that the rainfall measurement is done at discrete times. The final values, given in the grey scaled image, have therefore a relative uncertainty of [+50% -25%].

### Add your own areas
This can be done either by creating a file `subareas.csv` in the script folder, with comma-separated values for
1. (`int`) x-Pixel position in the original file in `image_data`, starting from 0. In the future, that should be derived from lat/lon but for now I used Irfanview to see the pixel position during a click (which is already 0 based). To know which pixel, I looked for rain reaching the position on the metoffice website.
2. (`int`) y-Pixel position in the original file in `image_data`.
3. longitude
4. latitude
5. semi-size of the box around the selected pixel, e.g. 1 leads to a 3x3 array, 10 to 21x21
6. scale of a pixel in the final image, e.g. a scale of 5 on a 3x3 array leads to a 15x15 png
7. name of the place

Alternatively, the script can be edited to add the lines containing `SUBAREAS.append` with the same entries (name needs to be in quotes)
