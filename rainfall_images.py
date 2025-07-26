import csv
import os
import re
import requests
import sys
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
from collections import defaultdict
import cv2

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image_data")


class DownloadFiles:
    # Configuration
    BASE_URL = "https://maps.consumer-digital.api.metoffice.gov.uk/wms_ob/single/high-res/rainfall_radar/"
    INTERVAL_MINUTES = 5
    DAYS_AVAILABLE = 7
    TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

    def __init__(self):
        # Create download directory if it doesn't exist
        os.makedirs(IMAGE_DIR, exist_ok=True)
    
    @staticmethod
    def floor_to_nearest_five(dt):
        floored_minute = dt.minute - (dt.minute % 5)
        return dt.replace(minute=floored_minute, second=0, microsecond=0)

    def run(self):
        # Calculate time range
        end_time = self.floor_to_nearest_five(datetime.utcnow())
        start_time = end_time - timedelta(days=self.DAYS_AVAILABLE, minutes=60)  # allow to go back another hour
        delta = timedelta(minutes=self.INTERVAL_MINUTES)

        # Download loop
        current = end_time
        while current >= start_time:
            timestamp = current.strftime(self.TIME_FORMAT)
            filename = f"{timestamp}.png"
            filepath = os.path.join(IMAGE_DIR, filename)
            url = f"{self.BASE_URL}{filename}"

            if os.path.exists(filepath):
                print(f"Already exists: {filename} - will not attempt older files")
                break
            else:
                print(f"Downloading: {url}")
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        with open(filepath, "wb") as f:
                            f.write(response.content)
                    else:
                        print(f"Failed: HTTP {response.status_code}")
                except requests.RequestException as e:
                    print(f"Error: {e}")

            current -= delta


class ProcessImages:
    DAILY_SUM_DIR = os.path.join(SCRIPT_DIR, "image_daily")
    DAILY_SUB_DIR = os.path.join(SCRIPT_DIR, "subframes_daily")
    VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos")
    # Regex to extract datetime from filename
    filename_re = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}\:\d{2}\:00Z)\.png')
    filename_daily_re = re.compile(r'(\d{4}-\d{2}-\d{2})_sum(_cur)?\.png')
    SUBAREAS = [] # x, y (irfan, start at 0), lon, lat, px_box_size/2, scale up, name # Todo: calculate px position from lat/lon
    SUBAREAS.append([2623, 2011, -0.119305, 51.509704, 30, 2, "London"])
    SUBAREAS.append([2363, 2019, -2.585907, 51.458285, 5, 2, "Bristol"])
    SUBAREAS.append([2649, 1915, 0.12291, 52.207607, 5, 2, "Cambridge"])
    SUBAREAS.append([2189, 1376, -4.31428, 55.93901, 1, 5, "Milngavie"])
    SUBAREAS.append([2098, 1240, -5.095253, 56.822303, 1, 5, "Fort William"])
    SUBAREAS_FILENAME = os.path.join(SCRIPT_DIR, "subareas.csv")    # more entries as csv, not included in the python script for privacy reasons
    SUBAREAS_TYPE = [int, int, float, float, int, int, str]
    
    conversions = [ # as 'P', avg, 33%ile mm, min mm, max mm, palette from palette = img.getpalette(); print([tuple(palette[i:i+3]) for i in range(0, len(palette), 3)])
    [ 0,   0,   0,     0,   0.1, (0, 0, 0)],
    [ 1,   0.3, 0.3,   0.1, 0.5, (0, 0, 254)],    # <0.5    blue
    [ 4,   0.7, 0.7,   0.5, 1,   (50, 101, 254)], # 0.5-1   lighter blue
    [ 3,   1.5, 1.3,   1,   2,   (12, 188, 254)], # 1-2     light blue
    [ 2,   3,   2.6,   2,   4,   (0, 163, 0)],    # 2-4     green
    [ 8,   6,   5.3,   4,   8,   (254, 203, 0)],  # 4-8     yellow
    [ 7,  12,  10.6,   8,  16,   (254, 152, 0)],  # 8-16    orange
    [ 6,  24,  21.3,  16,  32,   (254, 0, 0)],    # 16-32   red
    [ 5,  48,  42.6,  32,  64,   (179, 0, 0)],    # >32     dark red
    [ 9,  96,  85.3,  64, 128,   (171, 32, 253)], # 64-128  purple
    [10, 192, 170.6, 128, 999,   (255, 255, 255)],# >128    white
    ]


    def __init__(self):
        os.makedirs(self.DAILY_SUM_DIR, exist_ok=True)
        os.makedirs(self.DAILY_SUB_DIR, exist_ok=True)
        os.makedirs(self.VIDEO_DIR, exist_ok=True)
        if os.path.exists(self.SUBAREAS_FILENAME):
            print(f"Will use sub areas from {self.SUBAREAS_FILENAME}")
            with open(self.SUBAREAS_FILENAME, newline='') as csvfile:
                reader = csv.reader(csvfile, skipinitialspace=True, )
                for row in reader:
                    if len(row) != len(self.SUBAREAS_TYPE):
                        print(f"Skipping row with wrong number of columns: {row}")
                        continue
                    try:
                        parsed_row = [typ(val) for typ, val in zip(self.SUBAREAS_TYPE, row)]
                        self.SUBAREAS.append(parsed_row)
                    except ValueError as e:
                        print(f"Skipping row due to conversion error: {row} ({e})")
        self.palette = [0] * len(self.conversions) * 3
        for conversion in self.conversions:
            for i in range(3):
                self.palette[conversion[0]*3+i] = conversion[5][i]   # palette needs to be in increasing number order, but for my convenience conversions are in the order of increasing values

    @staticmethod
    def parse_datetime(s):
        # Unescape colon and parse datetime
        s = s.replace('\\:', ':')
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")

    def save_video_palette(self, frames, video_path, fps=10):
        number, height, width = frames.shape
        rgb_frames = []
        for i in range(number):
            pal_img = Image.fromarray(frames[i]).convert('P')
            pal_img.putpalette(self.palette)
            rgb_img = pal_img.convert('RGB')  # convert to RGB for OpenCV
            rgb_frames.append(np.array(rgb_img))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=(width, height))
        for frame in rgb_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # OpenCV uses BGR
        out.release()
    
    def run(self):
        # Organize images by day
        images_by_day = defaultdict(list)
        images = self.list_day_sum_images()  # already processed images
        for fname in sorted(os.listdir(IMAGE_DIR)):
            match = self.filename_re.match(fname)
            if not match:
                continue
            dt = self.parse_datetime(match.group(1))
            day = dt.date()
            if f"{day}_sum.png" not in images.values():
                images_by_day[day].append((dt, os.path.join(IMAGE_DIR, fname)))
        self.process_days(images_by_day)
        
        # Create summary video from summed images
        images = self.list_day_sum_images()
        if len(images.keys()):
            self.process_summaries(images)
        
        print(f"{datetime.now().time()} Done.")

    def list_day_sum_images(self):
        images = {}
        for fname in sorted(os.listdir(self.DAILY_SUM_DIR)):
            match = self.filename_daily_re.match(fname)
            if not match:
                continue
            day = match.group(1)
            images[day] = fname
        return images

    def process_days(self, images_by_day):
        # Process each day
        for day, day_files in images_by_day.items():
            print(f"{datetime.now().time()} Processing {day} with {len(day_files)} frames...")

            frames = {}
            accum = None

            for i, (dt, path) in enumerate(sorted(day_files)):
                img = Image.open(path)  # keep as "P" as it's a colour palette. Convert to decimals using .convert("L")
                arr = np.array(img, dtype=np.uint8)
                deci_arr = np.zeros_like(arr, dtype=np.float64)
                for conversion in self.conversions:
                    deci_arr[arr==conversion[0]] = conversion[2]
                deci_arr /= 12  # 5min are a 1/12th of an hour to convert mm/hour into mm
                #print(arr.shape, np.unique(arr,return_counts=True))

                # Sum up images
                if accum is None:
                    accum = np.zeros_like(deci_arr, dtype=np.float64)
                accum += deci_arr
                
                # Extract subsections
                for (x, y, _, _, size, _, name) in self.SUBAREAS:
                    if name not in frames.keys():
                        frames[name] = np.zeros((len(day_files), 2*size+1, 2*size+1), dtype=np.uint8)
                    frames[name][i,:,:] = arr[y-size:y+size+1, x-size:x+size+1]
                    #if name == "Home":
                       # print(name, frames[name][i,:,:]) # check what needs to be done to the values
            
            # Save daily summed image as grey-scale mm and with colour palette
            summed_img = np.clip(accum, 0, 255).astype(np.uint8)
            Image.fromarray(summed_img).save(f"{self.DAILY_SUM_DIR}/greyscale-mm_{day}_sum.png")
            # Save daily summed image as with colour palette
            # accum *= 0.1    # Show the colour scale in cm instead of mm
            summed_img = np.zeros_like(accum, dtype=np.uint8)
            for conversion in self.conversions:
                summed_img[(accum>=conversion[3]) & (accum<conversion[4])] = conversion[0]
            img = Image.fromarray(summed_img).convert('P')
            img.putpalette(self.palette)
            # Mark if it's the current day
            fname_current_day = f"{self.DAILY_SUM_DIR}/{day}_sum_cur.png"
            if os.path.exists(fname_current_day):
                os.remove(fname_current_day)
            img.save(fname_current_day if day == datetime.utcnow().date() else f"{self.DAILY_SUM_DIR}/{day}_sum.png")
            
            # Write daily videos
            for name, subframes in frames.items():
                video_path = f"{self.VIDEO_DIR}/{name.replace(' ', '_')}_{day}.mp4"
                self.save_video_palette(subframes, video_path, fps=12)  # 1h -> 1s

    def process_summaries(self, images):
        print(f"{datetime.now().time()} Processing daily images with {len(images)} frames...")
        frames = {}
        for i, (day, fname) in enumerate(sorted(images.items())):
            arr = np.array(Image.open(os.path.join(self.DAILY_SUM_DIR, fname)), dtype=np.uint8)
                
            for (x, y, _, _, size, scale, name) in self.SUBAREAS:
                arr_sub = arr[y-size:y+size+1, x-size:x+size+1]
                if name not in frames.keys():
                    frames[name] = np.zeros((len(images), 2*size+1, 2*size+1), dtype=np.uint8)
                frames[name][i,:,:] = arr_sub
                # Save the daily subframes with colour palette
                img = Image.fromarray(arr_sub).convert('P')
                img = img.resize((arr_sub.shape[1] * scale, arr_sub.shape[0] * scale), resample=Image.NEAREST)
                img.putpalette(self.palette)
                img.save(f"{self.DAILY_SUB_DIR}/{name.replace(' ', '_')}_{day}.png")

        for name, subframes in frames.items():
            video_path = f"{self.VIDEO_DIR}/{name.replace(' ', '_')}.mp4"
            self.save_video_palette(subframes, video_path)

DownloadFiles().run()
ProcessImages().run()
