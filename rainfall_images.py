import csv
from collections import defaultdict
import cv2
from datetime import datetime, timedelta
import json
from multiprocessing import Pool
import os
import re
import requests
import numpy as np
from PIL import Image
import psutil

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image_data")
DAILY_SUB_DIR = os.path.join(SCRIPT_DIR, "subframes_daily")
CONVERSIONS = [
    # as 'P', avg, 33%ile mm, min mm, max mm, palette from palette = img.getpalette(); print([tuple(palette[i:i+3]) for i in range(0, len(palette), 3)]), legend text
    [0, 0, 0, 0, 0.1, (0, 0, 0)],
    [1, 0.3, 0.3, 0.1, 0.5, (0, 0, 254), "<0.5"],  # <0.5    blue
    [4, 0.7, 0.7, 0.5, 1, (50, 101, 254), "0.5-1"],  # 0.5-1   lighter blue
    [3, 1.5, 1.3, 1, 2, (12, 188, 254), "1-2"],  # 1-2     light blue
    [2, 3, 2.6, 2, 4, (0, 163, 0), "2-4"],  # 2-4     green
    [8, 6, 5.3, 4, 8, (254, 203, 0), "4-8"],  # 4-8     yellow
    [7, 12, 10.6, 8, 16, (254, 152, 0), "8-16"],  # 8-16    orange
    [6, 24, 21.3, 16, 32, (254, 0, 0), "16-32"],  # 16-32   red
    [5, 48, 42.6, 32, 64, (179, 0, 0), "32-64"],  # >32     dark red
    [9, 96, 85.3, 64, 128, (171, 32, 253), "64-128"],  # 64-128  purple
    [10, 192, 170.6, 128, 999, (255, 255, 255), ">128"],  # >128    white
]
STAT_INFORMATION = os.path.join(SCRIPT_DIR, "statistics.json")


def read_image_statistics():
    stats = {}
    if os.path.exists(STAT_INFORMATION):
        try:
            with open(STAT_INFORMATION, "r") as f:
                stats = json.load(f)
        except json.JSONDecodeError as e:
            print("Invalid JSON:", e)
    return stats

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

def _process_day(args):
    # to allow multiprocessing to pickle it
    processImagesInstance, day, day_files = args
    return processImagesInstance.process_day(day, day_files)

class ProcessImages:
    DAILY_SUM_DIR = os.path.join(SCRIPT_DIR, "image_daily")
    STAT_INFORMATION = os.path.join(SCRIPT_DIR, "statistics.json")
    VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos")
    # Regex to extract datetime from filename
    filename_re = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}\:\d{2}\:00Z)\.png')
    filename_daily_re = re.compile(r'(\d{4}-\d{2}-\d{2})_sum\.png')
    SUBAREAS = []  # x, y (irfan, start at 0), lon, lat, px_box_size/2, scale up, name # Todo: calculate px position from lat/lon
    SUBAREAS.append([2623, 2011, -0.119305, 51.509704, 30, 2, "London"])
    SUBAREAS.append([2363, 2019, -2.585907, 51.458285, 5, 2, "Bristol"])
    SUBAREAS.append([2649, 1915, 0.12291, 52.207607, 5, 2, "Cambridge"])
    SUBAREAS.append([2300, 1373, -3.188438, 55.950257, 10, 1, "Edinburgh"])
    SUBAREAS.append([2189, 1376, -4.31428, 55.93901, 1, 5, "Milngavie"])
    SUBAREAS.append([2098, 1240, -5.095253, 56.822303, 1, 5, "Fort William"])
    SUBAREAS_FILENAME = os.path.join(SCRIPT_DIR, "subareas.csv")    # more entries in a csv file,
                                                            # not included in the python script for privacy reasons
    SUBAREAS_TYPE = [int, int, float, float, int, int, str]

    def __init__(self):
        os.makedirs(self.DAILY_SUM_DIR, exist_ok=True)
        os.makedirs(DAILY_SUB_DIR, exist_ok=True)
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
        self.palette = [0] * len(CONVERSIONS) * 3
        for conversion in CONVERSIONS:
            for i in range(3):
                self.palette[conversion[0] * 3 + i] = conversion[5][i]  # palette needs to be in increasing number order
                                            # , but for my convenience CONVERSIONS are in the order of increasing values
        self.use_cores = psutil.cpu_count(logical=False)    # only use physical cores
        self.stats = read_image_statistics()

    def write_image_statistics(self):
        with open(STAT_INFORMATION, "w") as f:
            f.write(json.dumps(self.stats, indent=2))

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
        for fname in sorted(os.listdir(IMAGE_DIR)):
            match = self.filename_re.match(fname)
            if not match:
                continue
            dt = self.parse_datetime(match.group(1))
            day = dt.date()
            images_by_day[day].append((dt, os.path.join(IMAGE_DIR, fname)))
        for day, data in list(images_by_day.items()):
            day_str = day.strftime("%Y-%m-%d")
            if self.stats.get(day_str, 0) == len(data):
                del images_by_day[day]  # do not reprocess
            else:
                self.stats[day_str] = len(data)
        self.process_days(images_by_day)
        self.write_image_statistics()

        # Create summary video from summed images and weekly/monthly summaries
        images = self.list_day_sum_images()
        if len(images.keys()):
            self.process_summaries(images)

        print(f"{datetime.now().time()} Done.")

    def list_day_sum_images(self):
        # Already processed images
        images = {}
        for fname in sorted(os.listdir(self.DAILY_SUM_DIR)):
            match = self.filename_daily_re.match(fname)
            if not match:
                continue
            day = match.group(1)
            images[day] = fname
        return images

    def process_day(self, day, day_files):
        frames = {}
        accum = None

        for i, (dt, path) in enumerate(sorted(day_files)):
            img = Image.open(path)  # keep as "P" as it's a colour palette. Convert to decimals using .convert("L")
            arr = np.array(img, dtype=np.uint8)
            deci_arr = np.zeros_like(arr, dtype=np.float64)
            for conversion in CONVERSIONS:
                deci_arr[arr == conversion[0]] = conversion[2]
            deci_arr /= 12  # 5min are a 1/12th of an hour to convert mm/hour into mm
            # print(arr.shape, np.unique(arr,return_counts=True))

            # Sum up images
            if accum is None:
                accum = np.zeros_like(deci_arr, dtype=np.float64)
            accum += deci_arr

            # Extract subsections
            for (x, y, _, _, size, _, name) in self.SUBAREAS:
                if name not in frames.keys():
                    frames[name] = np.zeros((len(day_files), 2 * size + 1, 2 * size + 1), dtype=np.uint8)
                frames[name][i, :, :] = arr[y - size:y + size + 1, x - size:x + size + 1]

        # Save daily summed image as grey-scale mm and with colour palette
        summed_img = np.clip(accum, 0, 255).astype(np.uint8)
        Image.fromarray(summed_img).save(f"{self.DAILY_SUM_DIR}/greyscale-mm_{day}_sum.png")
        # Save daily summed image as with colour palette
        # accum *= 0.1    # Show the colour scale in cm instead of mm
        summed_img = np.zeros_like(accum, dtype=np.uint8)
        for conversion in CONVERSIONS:
            summed_img[(accum >= conversion[3]) & (accum < conversion[4])] = conversion[0]
        img = Image.fromarray(summed_img).convert('P')
        img.putpalette(self.palette)
        img.save(f"{self.DAILY_SUM_DIR}/{day}_sum.png")

        # Write daily videos
        for name, subframes in frames.items():
            video_path = f"{self.VIDEO_DIR}/{name.replace(' ', '_')}_{day}.mp4"
            self.save_video_palette(subframes, video_path, fps=12)  # 1h -> 1s

        return {'day': day, 'number_files': len(day_files)}

    def process_days(self, images_by_day):
        tasks = [(self, day, day_files) for day, day_files in images_by_day.items()]
        cores = min(self.use_cores, len(tasks))
        print(f"{datetime.now().time()} Processing {len(tasks)} days on {cores} cores")
        with Pool(processes=cores) as pool:
            for result in pool.imap(_process_day, tasks):
                print(f"{datetime.now().time()} Finished {result['day']} with {result['number_files']} frames...")
            # results = pool.map(_process_day, tasks)
            # return results  # only executed the pool now

    def process_summaries(self, images):
        print(f"{datetime.now().time()} Processing daily images with {len(images)} frames...")
        frames = {}
        for i, (day, fname) in enumerate(sorted(images.items())):
            arr = np.array(Image.open(os.path.join(self.DAILY_SUM_DIR, fname)), dtype=np.uint8)

            for (x, y, _, _, size, scale, name) in self.SUBAREAS:
                arr_sub = arr[y - size:y + size + 1, x - size:x + size + 1]
                if name not in frames.keys():
                    frames[name] = np.zeros((len(images), 2 * size + 1, 2 * size + 1), dtype=np.uint8)
                frames[name][i, :, :] = arr_sub
                # Save the daily subframes with colour palette
                img = Image.fromarray(arr_sub).convert('P')
                img = img.resize((arr_sub.shape[1] * scale, arr_sub.shape[0] * scale), resample=Image.NEAREST)
                img.putpalette(self.palette)
                img.save(f"{DAILY_SUB_DIR}/{name.replace(' ', '_')}_{day}.png")

        for name, subframes in frames.items():
            video_path = f"{self.VIDEO_DIR}/{name.replace(' ', '_')}.mp4"
            self.save_video_palette(subframes, video_path)


class MakePdf:
    texFileNameBase = "rainfall.tex"
    framesPerRow = 7
    rowsPerPage = 9
    filename_daily_re = re.compile(r'(.*)_(\d{4}-\d{2}-\d{2})\.png')
    TEX_DIR = os.path.join(SCRIPT_DIR, "tex")

    def __init__(self):
        os.makedirs(self.TEX_DIR, exist_ok=True)
        os.chdir(self.TEX_DIR)
        self.legend_items = [(conv[5], conv[6].replace("<", "$<$").replace(">", "$>$")) for conv in CONVERSIONS if conv[0] > 0]
        self.stats = read_image_statistics()

    def list_day_sub_images(self):
        images = {}
        for fname in sorted(os.listdir(DAILY_SUB_DIR)):
            match = self.filename_daily_re.match(fname)
            if not match:
                continue
            name = match.group(1)
            day = match.group(2)
            if name not in images.keys():
                images[name] = {}
            images[name][day] = os.path.join(DAILY_SUB_DIR, fname)
        return images

    def write_end_and_legend(self, f):
        f.write(r"\end{tabular}\end{figure}" + "\n"
                r"\noindent\begin{tikzpicture}[x=1cm, y=1cm]" + "\n")
        for i, (color, label) in enumerate(self.legend_items):
            if i == len(self.legend_items) - 1:
                label += " mm/d"    # add the physical property to the last lable
            x = i * 1.7
            f.write(f"  \\definecolor{{c{i}}}{{RGB}}{{{color[0]}, {color[1]}, {color[2]}}}\n"
                    f"  \\filldraw[fill=c{i}, draw=black] ({x:.1f}, 0) rectangle ({(x + 0.5):.1f}, 0.5);\n"
                    f"  \\node[right] at ({(x + 0.6):.1f}, 0.25) {{\\small {label}}};\n")
        f.write(r"\end{tikzpicture}" + "\n")

    def create_latex(self):
        images = self.list_day_sub_images()
        for name, img in images.items():
            texName = f"{name}_{self.texFileNameBase}"
            with open(texName, "w") as f:
                f.write(r"""\pdfinfo{
   /Author (%s)
   /Title  (Daily rainfall for %s)
   /CreationDate (D:%s)
}
\documentclass[a4paper]{article}
\usepackage[margin=0.5in]{geometry}
\usepackage{graphicx}
\usepackage{tikz}
\pagestyle{empty}

\newcommand{\subf}[2]{
  {\small\begin{tabular}[t]{@{}c@{}}
  #1\\#2
  \end{tabular}}
}
\newcommand{\addDot}[1]{
  \begin{tikzpicture}
    \node[inner sep=0pt] (img) at (0,0) {#1};
    \fill[white] (img.center) circle[radius=0.5pt];
  \end{tikzpicture}
}

\begin{document}
""" % (os.path.basename(__file__), name, datetime.now().strftime("%y%m%d%H%M%S")) )
                rows = 0
                for i, (day, fname) in enumerate(img.items()):
                    posOnRow = i % self.framesPerRow
                    if posOnRow == 0 and i != 0:
                        # f.write(r"\vspace{0.2cm}\newline" + "\n")  # new row
                        rows += 1
                        if rows == self.rowsPerPage:
                            # new page needed
                            self.write_end_and_legend(f)
                            f.write(r"\newpage" + "\n")
                            rows = 0
                    if posOnRow == 0 and rows == 0:  # this is a new page
                        # f.write(r"" + name + "\n")    # write place name
                        f.write(r"\noindent\begin{figure}[t!]\centering"
                                r"\begin{tabular}{" + "c" * self.framesPerRow + "}\n")
                    frame_number_text = ""
                    if self.stats.get(day, 288) != 288:
                        frame_number_text = f" ({self.stats[day]})"
                    f.write(r"  \subf{\addDot{\includegraphics[width=0.12\linewidth]{\detokenize{" + fname + r"}}}}" +
                            "{" + day + frame_number_text + "}" +  # subtitle
                            ("&" if posOnRow < self.framesPerRow - 1 else r"\\") + "\n")  # middle or last entry
                self.write_end_and_legend(f)

                f.write(r"\end{document}")
            self.compile_latex(texName)

    @staticmethod
    def compile_latex(texName):
        os.system(f"pdflatex -interaction=batchmode {texName}")


if __name__ == "__main__":
    DownloadFiles().run()
    ProcessImages().run()
    toPdf = MakePdf()
    toPdf.create_latex()
