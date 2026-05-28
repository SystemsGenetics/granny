"""
This module performs color extraction/evaluation calculation on pear image files.
The analysis is conducted as follows:
    1. It loads input images from a specified directory.
    2. Removes surrounding purple from apples using YCrCb color space.
    3. Calculates mean values of green and yellow in CIELAB color space, normalized to L = 50.
    4. Calculates distance from normalized LAB to each bin color.
    5. Calculates distance to the least-mean-square line in LAB color space.

date: July 12, 2024
author: Nhan H. Nguyen

--- ANNOTATION OF CHANGES (Loren Honaas, May 2026) ---
This file was modified on 2026-05-27/28 during color analysis of WA 64 blush apples.
Changes fall into two categories:
  (A) Bug fix — submitted as PR to SystemsGenetics/granny, branch fix-peelcolor-divide-by-zero
  (B) New features — added to loren-honaas/granny fork only
All changes are marked inline with # [CHANGE ...] comments.
The original scoring pipeline and CSV output are completely unchanged.
"""

import os
from datetime import datetime
from multiprocessing import Pool
from typing import Any, List, Tuple, cast

import cv2
import numpy as np
from Granny.Analyses.Analysis import Analysis
from Granny.Models.Images.Image import Image
from Granny.Models.Images.RGBImage import RGBImage
from Granny.Models.IO.ImageIO import ImageIO
from Granny.Models.IO.RGBImageFile import RGBImageFile
from Granny.Models.Values.FloatValue import FloatValue
from Granny.Models.Values.ImageListValue import ImageListValue
from Granny.Models.Values.IntValue import IntValue
from Granny.Models.Values.MetaDataValue import MetaDataValue
from numpy.typing import NDArray


class PeelColor(Analysis):
    """
    Analysis class for evaluating peel color characteristics of images.

    Attributes:
        __analysis_name__ (str): Name of the analysis.
        input_images (ImageListValue): Input images directory.
        output_images (ImageListValue): Output images directory for analyzed images.
        output_results (MetaDataValue): Output directory for analysis results.
        MEAN_VALUES_A (List[float]): Mean values for A component in color card.
        MEAN_VALUES_B (List[float]): Mean values for B component in color card.
        SCORE (List[float]): Scores corresponding to color bins.
        LINE_POINT_1 (NDArray[np.float16]): First point of the reference line in LAB color space.
        LINE_POINT_2 (NDArray[np.float16]): Second point of the reference line in LAB color space.
    """

    __analysis_name__ = "color"

    def __init__(self):
        super().__init__()
        # sets up input and output directory
        self.input_images = ImageListValue(
            "input", "input", "The directory where input images are located."
        )
        self.input_images.setIsRequired(True)

        # NOTE: The following CLI parameters (purple_threshold through normalize_lightness)
        # were added in the upstream fork (not by Loren). They expose the previously
        # hardcoded LAB threshold values as configurable parameters, which partially
        # addresses the blush apple limitation described in GitHub issue #__.
        self.purple_threshold = IntValue(
            "purple_threshold",
            "purple_threshold",
            "Threshold for removing purple background/tray pixels using YCrCb color space. "
            + "Pixels with Cb channel <= this value are kept. Range is 0 to 255, default is 126.",
        )
        self.purple_threshold.setMin(0)
        self.purple_threshold.setMax(255)
        self.purple_threshold.setValue(126)
        self.purple_threshold.setIsRequired(False)

        self.lightness_min = IntValue(
            "lightness_min",
            "lightness_min",
            "Minimum lightness value for peel color detection in LAB color space. "
            + "Range is 0 to 255, default is 0.",
        )
        self.lightness_min.setMin(0)
        self.lightness_min.setMax(255)
        self.lightness_min.setValue(0)
        self.lightness_min.setIsRequired(False)

        self.lightness_max = IntValue(
            "lightness_max",
            "lightness_max",
            "Maximum lightness value for peel color detection in LAB color space. "
            + "Range is 0 to 255, default is 255.",
        )
        self.lightness_max.setMin(0)
        self.lightness_max.setMax(255)
        self.lightness_max.setValue(255)
        self.lightness_max.setIsRequired(False)

        self.green_min = IntValue(
            "green_min",
            "green_min",
            "Minimum green channel value for peel color detection in LAB color space. "
            + "Range is 0 to 255, default is 0.",
        )
        self.green_min.setMin(0)
        self.green_min.setMax(255)
        self.green_min.setValue(0)
        self.green_min.setIsRequired(False)

        # NOTE: green_max default of 128 corresponds to the neutral point of the
        # OpenCV LAB a* channel (0–255 scale, 128 = neutral). Values below 128
        # indicate green; values above 128 indicate red/blush. This is the threshold
        # that causes fully-blush sun-side apple images to return no scoreable pixels.
        # For green/yellow pear analysis (the intended use case) this is correct.
        self.green_max = IntValue(
            "green_max",
            "green_max",
            "Maximum green channel value for peel color detection in LAB color space. "
            + "Range is 0 to 255, default is 128.",
        )
        self.green_max.setMin(0)
        self.green_max.setMax(255)
        self.green_max.setValue(128)
        self.green_max.setIsRequired(False)

        self.yellow_min = IntValue(
            "yellow_min",
            "yellow_min",
            "Minimum yellow channel value for peel color detection in LAB color space. "
            + "Range is 0 to 255, default is 128.",
        )
        self.yellow_min.setMin(0)
        self.yellow_min.setMax(255)
        self.yellow_min.setValue(128)
        self.yellow_min.setIsRequired(False)

        self.yellow_max = IntValue(
            "yellow_max",
            "yellow_max",
            "Maximum yellow channel value for peel color detection in LAB color space. "
            + "Range is 0 to 255, default is 255.",
        )
        self.yellow_max.setMin(0)
        self.yellow_max.setMax(255)
        self.yellow_max.setValue(255)
        self.yellow_max.setIsRequired(False)

        self.normalize_lightness = IntValue(
            "normalize_lightness",
            "normalize_lightness",
            "Target lightness value for color normalization in LAB space. "
            + "Range is 0 to 100, default is 50.",
        )
        self.normalize_lightness.setMin(0)
        self.normalize_lightness.setMax(100)
        self.normalize_lightness.setValue(50)
        self.normalize_lightness.setIsRequired(False)

        self.output_images = ImageListValue(
            "output",
            "output",
            "The output directory where analysis' images are written.",
        )
        result_dir = os.path.join(
            os.curdir,
            "results",
            self.__analysis_name__,
            datetime.now().strftime("%Y-%m-%d-%H-%M"),
        )
        self.output_images.setValue(result_dir)
        self.addInParam(
            self.input_images,
            self.purple_threshold,
            self.lightness_min,
            self.lightness_max,
            self.green_min,
            self.green_max,
            self.yellow_min,
            self.yellow_max,
            self.normalize_lightness,
        )

        # sets up output result directory
        self.output_results = MetaDataValue(
            "results",
            "results",
            "The output directory where analysis' results are written.",
        )
        self.output_results.setValue(result_dir)

        # Color card reference values normalized to the LMS line.
        # These 10 bins span the green-to-yellow spectrum of pear/apple background peel.
        # MEAN_VALUES_A and MEAN_VALUES_B are all negative/positive respectively,
        # confirming these are calibrated for green/yellow peel only.
        self.MEAN_VALUES_A: List[float] = [
            -36.64082458, -35.82390694, -29.47956688, -24.68504792, -21.51960279,
            -21.49440178, -19.49577289, -16.92159296, -13.70076143, -13.34873991,
        ]
        self.MEAN_VALUES_B: List[float] = [
            57.4946451,  58.6671866,  67.77337014, 74.65505828, 79.19849765,
            79.23466925, 82.10334927, 85.79813151, 90.42106829, 90.92633324,
        ]
        self.SCORE: List[float] = [
            0.5192001330394723, 0.5233446426876467, 0.5838859128997311,
            0.6529992071684837, 0.6934065834794143, 0.7210722038415041,
            0.7302285740121869, 0.7652909029091124, 0.8066780913327652,
            0.8106974639404376,
        ]
        self.LINE_POINT_1: NDArray[np.float16] = np.array([-76.69774, 0.0], dtype=np.float16)
        self.LINE_POINT_2: NDArray[np.float16] = np.array([0.0, 110.0861], dtype=np.float16)

    # ──────────────────────────────────────────────────────────────────────────
    # ORIGINAL METHODS (unchanged from upstream)
    # ──────────────────────────────────────────────────────────────────────────

    def remove_purple(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Remove surrounding purple from individual apples using YCrCb color space.
        UNCHANGED from upstream.
        """
        new_img = img.copy()
        ycc_img = cast(NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
        threshold_1 = np.logical_and((ycc_img[:, :, 0] >= 0), (ycc_img[:, :, 0] <= 255))
        threshold_2 = np.logical_and((ycc_img[:, :, 1] >= 0), (ycc_img[:, :, 1] <= 255))
        threshold_3 = np.logical_and((ycc_img[:, :, 2] >= 0), (ycc_img[:, :, 2] <= self.purple_threshold.getValue()))
        th123 = np.logical_and(np.logical_and(threshold_1, threshold_2), threshold_3).astype(np.uint8)
        for i in range(3):
            new_img[:, :, i] = new_img[:, :, i] * th123
        return new_img

    def get_green_yellow_values(self, img: NDArray[np.uint8]) -> Tuple[float, float, float]:
        """
        Get mean pixel values representing green and yellow in CIELAB color space,
        normalized to L = 50.

        [CHANGE A-1 — BUG FIX — submitted as PR]
        Problem: the original code divided by np.count_nonzero() of each individual
        channel after th123 was applied. If no pixels passed the mask, count was 0,
        causing RuntimeWarning: invalid value encountered in divide. The nan result
        then propagated silently through downstream scoring.

        Also: using a separate count per channel meant the three means could in
        principle use different denominators if the mask produced different zero
        patterns per channel — internally inconsistent.

        Fix: compute a single shared pixel_count = np.count_nonzero(th123) and use
        it for all three channels. Add an explicit early return of (nan, nan, nan)
        when pixel_count == 0, so callers receive a clear signal rather than silent nan.

        [CHANGE A-2 — THRESHOLD NOTE]
        The three threshold conditions below are the direct cause of the blush apple
        limitation. threshold_2 requires a < green_max (default 128) which is the
        neutral point of OpenCV's LAB a* channel — only green-side pixels pass.
        threshold_3 requires b > yellow_min (default 128) — only yellow-side pixels pass.
        Sun-side blush apple images have predominantly positive a* values (red side)
        and fail threshold_2 entirely, producing an empty th123 mask.
        This is correct behaviour for the green/yellow pear use case; the fix is the
        graceful handling of the empty mask, not the threshold itself.
        See GitHub issue #__ for a proposed configurable threshold enhancement.
        """
        # convert from BGR to Lab color space
        lab_img = cast(NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_BGR2LAB))

        # create binary matrices
        # NOTE: these thresholds are now driven by CLI parameters (lightness_min/max,
        # green_min/max, yellow_min/max) rather than hardcoded values, since the
        # upstream fork exposed them as configurable parameters.
        threshold_1 = np.logical_and(
            (lab_img[:, :, 0] > self.lightness_min.getValue()),
            (lab_img[:, :, 0] < self.lightness_max.getValue())
        )
        threshold_2 = np.logical_and(
            (lab_img[:, :, 1] > self.green_min.getValue()),
            (lab_img[:, :, 1] < self.green_max.getValue())   # default 128 = green side only
        )
        threshold_3 = np.logical_and(
            (lab_img[:, :, 2] > self.yellow_min.getValue()),  # default 128 = yellow side only
            (lab_img[:, :, 2] < self.yellow_max.getValue())
        )

        # combine to one matrix
        th123 = np.logical_and(
            np.logical_and(threshold_1, threshold_2), threshold_3
        ).astype(np.uint8)

        # apply the binary mask on the image
        for i in range(3):
            lab_img[:, :, i] = lab_img[:, :, i] * th123

        # [CHANGE A-1] Shared pixel count + zero guard
        # ORIGINAL CODE used np.count_nonzero(lab_img[:,:,channel]) for each channel
        # independently, which (a) could divide by zero and (b) used inconsistent
        # denominators. Now uses a single count from the mask itself.
        pixel_count = np.count_nonzero(th123)
        if pixel_count == 0:
            # No pixels survived the green/yellow threshold (e.g. fully blush/overcolor
            # fruit). Return nan so _processImage() can handle it explicitly rather than
            # propagating a silent RuntimeWarning through downstream computation.
            return (float('nan'), float('nan'), float('nan'))

        mean_l = np.sum(lab_img[:, :, 0]) / pixel_count * 100 / 255
        mean_a = np.sum(lab_img[:, :, 1]) / pixel_count - 128
        mean_b = np.sum(lab_img[:, :, 2]) / pixel_count - 128

        # normalize by shifting point in the spherical coordinates
        # scaled_l is now driven by the normalize_lightness CLI parameter (default 50).
        # All fruit are projected to the same L value so scoring reflects hue only,
        # removing lighting variation between images.
        radius = np.sqrt(mean_l**2 + mean_a**2 + mean_b**2)
        scaled_l = self.normalize_lightness.getValue()
        scaled_a = np.sign(mean_a) * np.sqrt(
            np.abs(radius**2 - scaled_l**2) / (1 + (mean_b / mean_a) ** 2)
        )
        scaled_b = np.sign(mean_b) * mean_b / mean_a * scaled_a

        return (scaled_l, scaled_a, scaled_b)

    def calculate_bin_distance(
        self, color_list: List[float], method: str = "Euclidean"
    ) -> Tuple[int, NDArray[np.float16]]:
        """
        Calculate distance from normalized LAB color to each bin color.
        UNCHANGED from upstream.
        """
        bin_num = 0
        dist: NDArray[np.float16]
        if method == "Euclidean":
            dist_a = color_list[0] - np.array(self.MEAN_VALUES_A)
            dist_b = color_list[1] - np.array(self.MEAN_VALUES_B)
            dist = np.sqrt(
                (dist_a / np.linalg.norm(dist_a)) ** 2
                + (dist_b / np.linalg.norm(dist_b)) ** 2
            )
            bin_num = np.argmin(dist) + 1
        if method == "X-component":
            dist_a = color_list[0] - np.array(self.MEAN_VALUES_A)
            dist_b = color_list[1] - np.array(self.MEAN_VALUES_B)
            dist = np.sqrt((dist_a / np.linalg.norm(dist_a)) ** 2)
            bin_num = np.argmin(dist) + 1
        if method == "Y-component":
            dist_a = color_list[0] - np.array(self.MEAN_VALUES_A)
            dist_b = color_list[1] - np.array(self.MEAN_VALUES_B)
            dist = np.sqrt((dist_b / np.linalg.norm(dist_b)) ** 2)
            bin_num = np.argmin(dist) + 1
        if method == "Score":
            dist = color_list[0] - np.array(self.SCORE)
            dist = np.abs(dist)
            bin_num = np.argmin(dist) + 1
        return bin_num, dist

    def _calculate_score_distance(
        self, color_list: List[float]
    ) -> Tuple[Tuple[float, float], float, float, float]:
        """
        Calculate distance to least-mean-square line in LAB color space.
        UNCHANGED from upstream.
        """
        def calculate_intersection(line1, line2):
            xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
            ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
            def det(a, b):
                return a[0] * b[1] - a[1] * b[0]
            div = det(xdiff, ydiff)
            d = (det(*line1), det(*line2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            return (x, y)

        score = 0
        distance = 0
        point = 0
        color_point = np.array([color_list[1], color_list[2]]).astype(dtype=float)
        n = self.LINE_POINT_2 - self.LINE_POINT_1
        n /= np.linalg.norm(n)
        projection = calculate_intersection(
            ((0.0, 0.0), (color_list[1], color_list[2])),
            (self.LINE_POINT_1, self.LINE_POINT_2),
        )
        score = cast(
            float,
            np.linalg.norm(projection - self.LINE_POINT_1)
            / np.linalg.norm(self.LINE_POINT_2 - self.LINE_POINT_1),
        )
        distance = cast(
            float,
            np.linalg.norm(
                np.cross(
                    self.LINE_POINT_2 - self.LINE_POINT_1,
                    color_point - self.LINE_POINT_1,
                )
            )
            / np.linalg.norm(self.LINE_POINT_2 - self.LINE_POINT_1),
        )
        point = np.sign(color_point[1] - projection[1])
        if score < 0:
            score = float(0)
        elif score > 1:
            score = float(1.0)
        return projection, score, distance, point

    def _processImage(self, image_instance: Image) -> Image:
        """
        Load, process, and score a single image instance.

        [CHANGE A-3 — BUG FIX — submitted as PR]
        Added a nan guard after calling get_green_yellow_values(). Without this guard,
        when l/a/b are nan (from the zero return added in CHANGE A-1), the downstream
        call to np.argmin() inside calculate_bin_distance() receives an all-nan array
        and raises a ValueError in newer numpy versions (>= 1.25). In Python 3.14 with
        numpy 2.4.6 this crashes the multiprocessing worker, causing pool.map() to
        return empty results and the entire analysis to fail with KeyError: 'Name'.

        The guard sets all output values to nan and returns the image instance early,
        bypassing all scoring. These images appear in results.csv with empty score/bin
        fields, clearly indicating they could not be scored rather than silently
        receiving a misleading bin=1 assignment (the original behaviour).
        """
        self.image_io.setFilePath(image_instance.getFilePath())
        image_instance.loadImage(image_io=self.image_io)
        img = image_instance.getImage()

        # remove surrounding purple
        img = self.remove_purple(img)

        # image smoothing
        img = cast(NDArray[np.uint8], cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0))

        # get image values — may return (nan, nan, nan) for overcolor/blush images
        l, a, b = self.get_green_yellow_values(img)

        # [CHANGE A-3] Nan guard: if no pixels passed the green/yellow threshold,
        # return early with all output values set to nan.
        # This prevents np.argmin() from receiving an all-nan array in
        # calculate_bin_distance(), which would crash the multiprocessing worker.
        import math
        if math.isnan(l):
            nan_fields = [
                ("bin",      "bin_num",  "Granny sorted bin number."),
                ("score",    "score",    "Granny calculated rating of the peel color."),
                ("distance", "distance", "Granny calculated distance from the LMS best-fit line."),
                ("location", "location", "Granny calculated location wrt. the LMS best-fit line."),
                ("l",        "L",        "Granny calculated L value."),
                ("a",        "A",        "Granny calculated A value."),
                ("b",        "B",        "Granny calculated B value."),
            ]
            for key, label, help_text in nan_fields:
                v = FloatValue(key, label, help_text)
                v.setValue(float('nan'))
                image_instance.addValue(v)
            return image_instance  # early return — no scoring performed

        # calculate distance to the least-mean-square line
        projection, score, orth_distance, point = self._calculate_score_distance([l, a, b])

        # bin number according to the color card
        bin_num, _ = self.calculate_bin_distance([score], method="Score")
        bin_value = FloatValue("bin", "bin_num",
            "Granny sorted bin number of the peel color, according to the color card.")
        bin_value.setMin(0.0)
        bin_value.setMax(1.0)
        bin_value.setValue(bin_num)

        score_value = FloatValue("score", "score", "Granny calculated rating of the peel color.")
        score_value.setMin(0.0)
        score_value.setMax(1.0)
        score_value.setValue(score)

        distance_value = FloatValue("distance", "distance",
            "Granny calculated distance from the LMS best-fit line.")
        distance_value.setValue(orth_distance)

        location_value = FloatValue("location", "location",
            "Granny calculated location wrt. the LMS best-fit line.")
        location_value.setValue(point)

        l_value = IntValue("l", "L", "Granny calculated L value of the image in the LAB space.")
        l_value.setValue(l)
        a_value = IntValue("a", "A", "Granny calculated A value of the image in the LAB space.")
        a_value.setValue(a)
        b_value = IntValue("b", "B", "Granny calculated B value of the image in the LAB space.")
        b_value.setValue(b)

        image_instance.addValue(
            bin_value, score_value, distance_value, location_value,
            l_value, a_value, b_value,
        )
        return image_instance

    # ──────────────────────────────────────────────────────────────────────────
    # [CHANGE B-1 — NEW METHOD — fork only, not submitted as PR]
    # _generateColorspacePlot(): generates two self-contained HTML visualization
    # files per run (colorspace_shade.html and colorspace_sun.html).
    #
    # Design principle: the original scoring pipeline (CSV output) is completely
    # unchanged. This method reads the already-written results.csv and output
    # images after the run completes, computes additional information for
    # visualization only, and writes HTML files alongside the existing outputs.
    #
    # Key design decisions:
    # - True fruit colors are computed from ALL non-background pixels (no threshold
    #   mask), so overcolor/blush fruit display their actual red color rather than
    #   the color of the few green pixels that passed the threshold.
    # - Overcolor (unscored) fruits are still plotted using their raw a*/b* as
    #   position, shown below y=0 in the box plot.
    # - The tray mean score is pulled from tray_summary.csv (already written by
    #   Granny) rather than recomputed, ensuring consistency with the CSV output.
    # - Statistical analysis uses scipy.stats (available in granny's pipx venv
    #   via ultralytics dependency); statsmodels is NOT required.
    # ──────────────────────────────────────────────────────────────────────────

    def _generateColorspacePlot(self, result_dir: str) -> None:
        """
        Generate self-contained HTML color space plots for shade (B) and sun (A) sides.
        Each file has a LAB scatter chart and a box-and-whisker tray score chart.
        Dot colors reflect true fruit color from raw image pixels.

        [CHANGE B-1] Entirely new method. Not present in upstream Granny.
        """
        import csv
        import json
        import statistics
        import re as _re

        results_path = os.path.join(result_dir, "results.csv")
        if not os.path.exists(results_path):
            return

        # ── Step 1: Read results.csv and compute true fruit colors from images ──
        # For each image, we compute raw LAB means from ALL non-background pixels
        # (bg_mask: any BGR channel > 0). This bypasses the green/yellow threshold
        # so blush/overcolor fruit display their actual red color in the plots.
        # raw_l, raw_a, raw_b are used ONLY for dot/bar color — not for scoring.
        points = []
        with open(results_path, newline="") as f:
            for row in csv.DictReader(f):
                name = row.get("Name", "")
                a = row.get("a", "")
                b = row.get("b", "")
                scored = bool(a and b)  # True if image produced a valid score

                # Read the output image to compute true average fruit color
                img_path = os.path.join(result_dir, name)
                raw_l = raw_a = raw_b = None
                if os.path.exists(img_path):
                    img_arr = cv2.imread(img_path)
                    if img_arr is not None:
                        lab_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2LAB)
                        # Background mask: include only pixels where at least one
                        # BGR channel is > 0 (i.e. not pure black background)
                        bg_mask = np.any(img_arr > 0, axis=2)
                        n = int(np.count_nonzero(bg_mask))
                        if n > 0:
                            # Convert from OpenCV LAB encoding to standard LAB:
                            # L: 0–255 in OpenCV → 0–100 standard (multiply by 100/255)
                            # a, b: 0–255 in OpenCV → -128 to 127 standard (subtract 128)
                            raw_l = float(np.sum(lab_arr[:, :, 0][bg_mask])) / n * 100 / 255
                            raw_a = float(np.sum(lab_arr[:, :, 1][bg_mask])) / n - 128
                            raw_b = float(np.sum(lab_arr[:, :, 2][bg_mask])) / n - 128

                if raw_l is not None:
                    try:
                        # Plot position: scored fruits use normalized a*/b* (from CSV);
                        # overcolor fruits use raw a*/b* (their actual color position)
                        pos_a = float(a) if scored else raw_a
                        pos_b = float(b) if scored else raw_b
                        m = _re.match(r'^(.*?)(?:_\d+)?\.(?:png|jpg|jpeg)$', name)
                        tray_name = m.group(1) if m else name
                        side = "A" if " A_fruit" in name else "B" if " B_fruit" in name else "?"
                        points.append({
                            "name": name.replace(".png", ""),
                            "tray": tray_name,
                            "x": pos_a,   # Chart.js scatter requires x/y keys
                            "y": pos_b,
                            "a": pos_a,   # kept for tooltip display
                            "b": pos_b,
                            "raw_l": raw_l,   # used for dot color only
                            "raw_a": raw_a,
                            "raw_b": raw_b,
                            "scored": scored,
                            "side": side,
                            "score": row.get("score", ""),
                            "bin": row.get("bin", ""),
                        })
                    except ValueError:
                        pass

        # ── Step 2: Box plot statistics helpers ──────────────────────────────

        def quartile(data, q):
            """Compute quantile q (0–1) of sorted data using linear interpolation."""
            if not data:
                return None
            s = sorted(data)
            n = len(s)
            idx = (n - 1) * q
            lo, hi = int(idx), min(int(idx) + 1, n - 1)
            return s[lo] + (s[hi] - s[lo]) * (idx - lo)

        def compute_tray_stats(side_points):
            """
            Group fruits by tray and compute:
            - Box plot statistics (min, Q1, median, Q3, max) from scored fruits
            - Per-fruit data for individual dot plotting (with deterministic x-jitter)
            - Tray-average true color (mean raw_l/a/b across ALL fruits including overcolor)
            - summary_mean placeholder (populated later from tray_summary.csv)
            """
            trays = {}
            order = []
            for p in side_points:
                t = p["tray"]
                if t not in trays:
                    trays[t] = {"fruits": []}
                    order.append(t)
                trays[t]["fruits"].append(p)

            stats = []
            for i, tray_name in enumerate(sorted(order)):
                fruits = trays[tray_name]["fruits"]
                scores = []
                for p in fruits:
                    if p["scored"] and p["score"]:
                        try:
                            scores.append(float(p["score"]))
                        except ValueError:
                            pass

                min_s = min(scores) if scores else None
                q1 = quartile(scores, 0.25)
                med = quartile(scores, 0.5)
                q3 = quartile(scores, 0.75)
                max_s = max(scores) if scores else None

                # Tray average color uses ALL fruits (including overcolor) so the
                # box color in the plot reflects the true average appearance of the tray
                all_raw_l = [p["raw_l"] for p in fruits]
                all_raw_a = [p["raw_a"] for p in fruits]
                all_raw_b = [p["raw_b"] for p in fruits]
                mean_raw_l = sum(all_raw_l) / len(all_raw_l)
                mean_raw_a = sum(all_raw_a) / len(all_raw_a)
                mean_raw_b = sum(all_raw_b) / len(all_raw_b)

                # Strip "_fruit" suffix and trailing whitespace for display label
                # e.g. "WA 64 Tray 1 B_fruit" → "WA 64 Tray 1 B"
                label = _re.sub(r'_fruit$', '', tray_name).strip()

                # Build per-fruit list with deterministic x-jitter so dots don't
                # overlap. Jitter spans ±0.15 units evenly across fruits in the tray.
                n_fruits = len(fruits)
                fruit_list = []
                for j, p in enumerate(fruits):
                    jitter = (j / max(n_fruits - 1, 1) - 0.5) * 0.3 if n_fruits > 1 else 0
                    score_val = None
                    if p["scored"] and p["score"]:
                        try:
                            score_val = float(p["score"])
                        except ValueError:
                            pass
                    fruit_list.append({
                        "name": p["name"],
                        "jitter": jitter,
                        "score": score_val,
                        "scored": p["scored"],
                        "raw_l": p["raw_l"],
                        "raw_a": p["raw_a"],
                        "raw_b": p["raw_b"],
                    })

                stats.append({
                    "tray": label,
                    "x": i,             # numeric x position for scatter-based box plot
                    "min_score": min_s,
                    "q1": q1,
                    "median": med,
                    "q3": q3,
                    "max_score": max_s,
                    "raw_l": mean_raw_l,
                    "raw_a": mean_raw_a,
                    "raw_b": mean_raw_b,
                    "fruits": fruit_list,
                    "summary_mean": None,  # populated below from tray_summary.csv
                })
            return stats

        # ── Step 3: Build HTML ────────────────────────────────────────────────
        # build_html() assembles a self-contained HTML file with two Chart.js panels:
        # Left: LAB scatter plot (a* vs b*), Right: box-and-whisker score distribution.
        # All data is embedded as JSON; no server required to open the file.
        #
        # The box-and-whisker is implemented as a Chart.js scatter chart with a custom
        # plugin (boxplotPlugin) that draws the boxes, whiskers, tray labels, and mean
        # score text using the Canvas 2D API. Chart.js does not natively support box plots.
        #
        # LAB→RGB color conversion is performed in JavaScript (labToRgb()) using the
        # standard CIE LAB → XYZ → sRGB pipeline with D65 illuminant, so each dot and
        # box is colored with the actual measured fruit color.

        def build_html(side_points, tray_stats, title):
            js_pts = json.dumps(side_points)
            js_tr = json.dumps(tray_stats)

            css = """
body{font-family:sans-serif;margin:2rem;background:#fafafa;color:#333}
h1{font-size:1.2rem;font-weight:500;margin-bottom:0.25rem}
p{font-size:0.85rem;color:#666;margin-bottom:1rem}
.legend{display:flex;gap:1.5rem;font-size:0.8rem;color:#555;margin-bottom:1rem}
.legend span{display:flex;align-items:center;gap:6px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.diamond{width:10px;height:10px;transform:rotate(45deg);display:inline-block}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:2rem;align-items:start}
.chart-wrap{position:relative;width:100%;height:460px}
h2{font-size:1rem;font-weight:500;margin:0 0 0.5rem}
"""

            # JavaScript: standard CIE LAB → XYZ → sRGB conversion.
            # Input: L (0–100), a* (-128 to 127), b* (-128 to 127) in standard LAB.
            # Output: [R, G, B] each 0–255, clamped and rounded.
            # mkColor(d) is a convenience wrapper that reads raw_l/raw_a/raw_b
            # from a data point object and returns an rgb() CSS color string.
            lab_to_rgb_js = """
function labToRgb(L,a,b){
  let fy=(L+16)/116,fx=a/500+fy,fz=fy-b/200;
  let x=(fx>0.2069?fx*fx*fx:(fx-16/116)/7.787)*0.95047;
  let y=(fy>0.2069?fy*fy*fy:(fy-16/116)/7.787)*1.00000;
  let z=(fz>0.2069?fz*fz*fz:(fz-16/116)/7.787)*1.08883;
  let r=x*3.2406+y*-1.5372+z*-0.4986;
  let g=x*-0.9689+y*1.8758+z*0.0415;
  let bv=x*0.0557+y*-0.2040+z*1.0570;
  r=r>0.0031308?1.055*Math.pow(r,1/2.4)-0.055:12.92*r;
  g=g>0.0031308?1.055*Math.pow(g,1/2.4)-0.055:12.92*g;
  bv=bv>0.0031308?1.055*Math.pow(bv,1/2.4)-0.055:12.92*bv;
  return [Math.min(255,Math.max(0,Math.round(r*255))),
          Math.min(255,Math.max(0,Math.round(g*255))),
          Math.min(255,Math.max(0,Math.round(bv*255)))];
}
function mkColor(d){
  const[r,g,b]=labToRgb(d.raw_l,d.raw_a,d.raw_b);
  return 'rgb('+r+','+g+','+b+')';
}
function mkColorRaw(rl,ra,rb){
  const[r,g,b]=labToRgb(rl,ra,rb);
  return 'rgb('+r+','+g+','+b+')';
}
"""

            # Left panel: LAB scatter plot.
            # x = normalized a* (scoring-based, from CSV), y = normalized b*.
            # For overcolor fruits: x = raw_a, y = raw_b (no normalized values available).
            # Scored fruits → circles; overcolor fruits → diamonds (rectRot pointStyle).
            scatter_js = (
                "const pts=" + js_pts + ";\n"
                """const scored_pts=pts.filter(d=>d.scored);
const unscored_pts=pts.filter(d=>!d.scored);
new Chart(document.getElementById('scatter'),{
  type:'scatter',
  data:{datasets:[
    {label:'Scored',data:scored_pts,pointStyle:'circle',pointRadius:8,pointHoverRadius:10,
      backgroundColor:scored_pts.map(mkColor),borderColor:scored_pts.map(mkColor),borderWidth:2},
    {label:'Overcolor',data:unscored_pts,pointStyle:'rectRot',pointRadius:8,pointHoverRadius:10,
      backgroundColor:unscored_pts.map(mkColor),borderColor:unscored_pts.map(mkColor),borderWidth:2},
  ]},
  options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:false},tooltip:{callbacks:{
      title:i=>i[0].raw.name,
      label:ctx=>[
        'a*: '+ctx.raw.x.toFixed(2),
        'b*: '+ctx.raw.y.toFixed(2),
        ctx.raw.scored?'score: '+ctx.raw.score:'overcolor (no score)'
      ]
    }}},
    scales:{
      x:{title:{display:true,text:'a* (green \u2190 \u2192 red)'},grid:{color:'rgba(0,0,0,0.05)'}},
      y:{title:{display:true,text:'b* (blue \u2190 \u2192 yellow)'},grid:{color:'rgba(0,0,0,0.05)'}}
    }
  }
});
"""
            )

            # Right panel: box-and-whisker implemented as a Chart.js scatter chart
            # with a custom Canvas plugin (boxplotPlugin).
            #
            # boxplotPlugin.afterDatasetsDraw() is called by Chart.js after the scatter
            # dots are drawn. It iterates over trayData (passed via chart options) and
            # manually draws each box, whisker, tray name label, and mean score label
            # using the Canvas 2D context.
            #
            # Individual fruit scores are the scatter dataset dots (colored by true
            # fruit color with deterministic x-jitter). Overcolor fruits appear at
            # y = -0.07, below the y=0 line, labelled 'overcolor' on the y-axis.
            boxplot_plugin_js = """
const boxplotPlugin={
  id:'boxplot',
  afterDatasetsDraw(chart){
    const ctx=chart.ctx;
    const trays=chart.config.options.trayData;
    if(!trays)return;
    const yScale=chart.scales.y;
    const xScale=chart.scales.x;
    // Box width scales with available space, clamped to 20–40 px
    const bw=Math.max(20, Math.min(40, (xScale.width/trays.length)*0.35));
    trays.forEach(t=>{
      if(t.min_score==null)return;
      const cx=xScale.getPixelForValue(t.x);
      const yMin=yScale.getPixelForValue(t.min_score);
      const yQ1=yScale.getPixelForValue(t.q1);
      const yMed=yScale.getPixelForValue(t.median);
      const yQ3=yScale.getPixelForValue(t.q3);
      const yMax=yScale.getPixelForValue(t.max_score);
      // Box fill color = tray average true color (from all fruits including overcolor)
      const [r,g,b]=labToRgb(t.raw_l,t.raw_a,t.raw_b);
      ctx.save();
      // Draw IQR box (Q1 to Q3)
      ctx.fillStyle='rgba('+r+','+g+','+b+',0.55)';
      ctx.strokeStyle='rgba(0,0,0,0.55)';
      ctx.lineWidth=1.5;
      ctx.fillRect(cx-bw/2,yQ3,bw,yQ1-yQ3);
      ctx.strokeRect(cx-bw/2,yQ3,bw,yQ1-yQ3);
      // Draw median line
      ctx.beginPath();
      ctx.moveTo(cx-bw/2,yMed);ctx.lineTo(cx+bw/2,yMed);
      ctx.strokeStyle='rgba(0,0,0,0.85)';ctx.lineWidth=2;ctx.stroke();
      // Draw whiskers (min to Q1, Q3 to max) with end caps
      ctx.strokeStyle='rgba(0,0,0,0.55)';ctx.lineWidth=1.5;
      ctx.beginPath();
      ctx.moveTo(cx,yQ1);ctx.lineTo(cx,yMin);
      ctx.moveTo(cx-bw/4,yMin);ctx.lineTo(cx+bw/4,yMin);
      ctx.moveTo(cx,yQ3);ctx.lineTo(cx,yMax);
      ctx.moveTo(cx-bw/4,yMax);ctx.lineTo(cx+bw/4,yMax);
      ctx.stroke();
      // Draw tray name label at the top of the chart area
      ctx.font='11px sans-serif';
      ctx.fillStyle='rgba(0,0,0,0.75)';
      ctx.textAlign='center';
      ctx.fillText(t.tray, cx, chart.chartArea.top+14);
      // Draw mean score from tray_summary.csv below the bottom whisker
      if(t.summary_mean!=null){
        ctx.font='10px sans-serif';
        ctx.fillStyle='rgba(0,0,0,0.6)';
        ctx.fillText('mean: '+t.summary_mean, cx, yMin+16);
      }
      ctx.restore();
    });
  }
};
"""

            boxplot_js = (
                "const trays=" + js_tr + ";\n"
                """const fruitPts=[];
trays.forEach((t,ti)=>{
  t.fruits.forEach(f=>{
    fruitPts.push({
      x: t.x + f.jitter,
      y: f.scored ? f.score : -0.07,  // overcolor dots shown below y=0
      raw_l:f.raw_l, raw_a:f.raw_a, raw_b:f.raw_b,
      name:f.name, scored:f.scored
    });
  });
});
const trayLabels=trays.map(t=>t.tray);
new Chart(document.getElementById('boxplot'),{
  type:'scatter',
  plugins:[boxplotPlugin],
  data:{datasets:[{
    data:fruitPts,
    pointStyle:fruitPts.map(d=>d.scored?'circle':'rectRot'),
    pointRadius:6,
    pointHoverRadius:8,
    backgroundColor:fruitPts.map(mkColor),
    borderColor:fruitPts.map(mkColor),
    borderWidth:1.5,
  }]},
  options:{
    responsive:true,
    maintainAspectRatio:false,
    trayData:trays,   // passed to boxplotPlugin via chart.config.options
    plugins:{
      legend:{display:false},
      tooltip:{callbacks:{
        title:i=>i[0].raw.name,
        label:ctx=>ctx.raw.scored
          ?('score: '+ctx.raw.y.toFixed(3))
          :'overcolor (no score)'
      }}
    },
    scales:{
      x:{
        min:-0.5,
        max:trays.length-0.5,
        ticks:{display:false},   // tray names drawn by plugin instead
        grid:{color:'rgba(0,0,0,0.05)'}
      },
      y:{
        min:-0.12,   // extra space below 0 for overcolor dots at y=-0.07
        max:1.0,
        ticks:{
          callback:(v)=>v<0?'overcolor':v.toFixed(1)
        },
        title:{display:true,text:'Score'},
        grid:{color:'rgba(0,0,0,0.05)'}
      }
    }
  }
});
"""
            )

            parts = [
                '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n',
                '<title>Granny \u2014 ' + title + '</title>\n',
                '<style>' + css + '</style>\n</head>\n<body>\n',
                '<h1>Peel color space \u2014 ' + title + '</h1>\n',
                '<p>Dot color = true fruit color. '
                'Scored: position = normalized a*/b*. '
                'Overcolor (no score): position = raw a*/b*.</p>\n',
                '<div class="legend">'
                '<span><span class="dot" style="background:#888;border:1px solid #555"></span>Scored</span>'
                '<span><span class="diamond" style="background:#888;border:1px solid #555"></span>Overcolor</span>'
                '</div>\n',
                '<div class="charts">\n',
                '  <div><h2>Color space (a* vs b*)</h2>',
                '  <div class="chart-wrap"><canvas id="scatter" role="img" aria-label="LAB scatter"></canvas></div></div>\n',
                '  <div><h2>Tray score distribution</h2>',
                '  <div class="chart-wrap"><canvas id="boxplot" role="img" aria-label="Box and whisker tray scores"></canvas></div></div>\n',
                '</div>\n',
                '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>\n',
                '<script>\n',
                lab_to_rgb_js,
                scatter_js,
                boxplot_plugin_js,
                boxplot_js,
                '</script>\n</body>\n</html>',
            ]
            return ''.join(parts)

        # ── Step 4: Generate one HTML file per side ───────────────────────────
        for side_label, side_name, title in [
            ("B", "shade", "Shade side (B)"),
            ("A", "sun",   "Sun side (A)"),
        ]:
            side_points = [p for p in points if p["side"] == side_label]
            if not side_points:
                continue
            tray_stats = compute_tray_stats(side_points)

            # Populate summary_mean from tray_summary.csv.
            # This file is written by Granny's MetaDataValue.writeValue() immediately
            # before _generateColorspacePlot() is called, so it is guaranteed to exist.
            # Using the summary file (rather than recomputing from points) ensures
            # the displayed mean matches exactly what is in tray_summary.csv.
            summary_path = os.path.join(result_dir, "tray_summary.csv")
            if os.path.exists(summary_path):
                with open(summary_path, newline="") as sf:
                    for srow in csv.DictReader(sf):
                        # TrayName in summary: "WA 64 Tray 1 B_fruit"
                        # After stripping: "WA 64 Tray 1 B" — matches tray_stats label
                        tname = _re.sub(r'_fruit$', '', srow.get('TrayName', '')).strip()
                        sc = srow.get('score', '')
                        if tname and sc:
                            try:
                                smean = round(float(sc), 3)
                                for s in tray_stats:
                                    if s['tray'] == tname:
                                        s['summary_mean'] = smean
                            except ValueError:
                                pass

            html = build_html(side_points, tray_stats, title)
            out_path = os.path.join(result_dir, "colorspace_" + side_name + ".html")
            with open(out_path, "w") as f:
                f.write(html)
            print("Plot written to " + out_path)

    # ──────────────────────────────────────────────────────────────────────────
    # ORIGINAL METHODS (unchanged from upstream)
    # ──────────────────────────────────────────────────────────────────────────

    def _preRun(self):
        """
        {@inheritdoc}
        UNCHANGED from upstream.
        """
        self.image_io: ImageIO = RGBImageFile()

    def _postRun(self, results):
        """
        {@inheritdoc}

        [CHANGE B-2 — fork only]
        Added call to self._generateColorspacePlot() at the end, after results.csv
        and tray_summary.csv have been written. This ensures the plot has access to
        both output files and all processed images.
        """
        # adds the result list to self.output_images then writes the resulting images to folder
        self.output_images.setImageList(results)
        self.output_images.writeValue()

        # adds the result list to self.output_results then writes the resulting results to folder
        self.output_results.setImageList(results)
        self.output_results.writeValue()

        self.addRetValue(self.output_images)

        # [CHANGE B-2] Generate HTML color space plots after all CSVs are written.
        # Not present in upstream Granny.
        self._generateColorspacePlot(self.output_results.getValue())

        return self.output_images.getImageList()
