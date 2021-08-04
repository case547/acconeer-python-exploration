import warnings
from copy import copy
from enum import Enum

import numpy as np

import acconeer.exptool as et
from argparse import ArgumentParser

import json
json_data = '{"ip_a": "10.54.14.197", "downsampling_factor": 1, "gain": 0.2, "hw_accelerated_average_samples": 10, "maximize_signal_attenuation": false, "noise_level_normalization": true, "profile": "et.configs.EnvelopeServiceConfig.Profile.PROFILE_1", "range_interval": [0.1, 0.8], "repetition_mode": "et.configs.EnvelopeServiceConfig.RepetitionMode.SENSOR_DRIVEN", "running_average_factor": 0, "tx_disable": false, "update_rate": 45}'
json_as_py = json.loads(json_data)
processor_json = '{"nbr_average" = 5.0, "threshold_type" = "ThresholdType.FIXED", "fixed_threshold" = 1800}'

PEAK_MERGE_LIMIT_M = 0.005


def main():    
    client = et.SocketClient(json_as_py['ip_a']) # Raspberry Pi uses socket client
    
    config = et.configs.EnvelopeServiceConfig() # picking envelope service

    sensor_config = get_sensor_config(config)
    processing_config = get_processing_config()
    sensor_config.sensor = [1]

    # Set up session with created config
    session_info = client.setup_session(sensor_config) # also calls connect()
    print("Session info:\n", session_info, "\n")

    client.start_session() # call will block until sensor confirms its start

    processor = Processor(sensor_config, processing_config, session_info)

    for _ in range(5):
        info, sweep = client.get_next()
        plot_data = processor.process(sweep, info)

        if plot_data["found_peaks"]:
            peaks = np.take(processor.r, plot_data["found_peaks"]) * 100.0
            print(info, "\n", "{:.2f} cm".format(peaks[0]), "\n")

    print("Disconnecting...")
    client.disconnect()


def get_sensor_config(config):
    """Define default sensor config and service to use."""
    for k, v in json_as_py.items():
        if hasattr(config, k):
            try:
                setattr(config, k, eval(v))
            except:
                setattr(config, k, v)

    # downsampling_factor - must be 1, 2, or 4
    # hw_accelerated_average_samples - number of samples taken for single point in data, [1,63]
    # range_interval - measurement range (metres)
    # running_average_factor - keep as 0; use averaging in detector instead of in API
    # tx_disable - enable/disable radio transmitter
    # update_rate - target measurement rate (Hz)

    return config


class Processor:
    """Detector class, which does all the processing."""
    def __init__(self, sensor_config, processing_config, session_info):
        self.session_info = session_info

        self.f = sensor_config.update_rate

        num_depths = self.session_info["data_length"]

        self.current_mean_sweep = np.zeros(num_depths)
        self.last_mean_sweep = np.full(num_depths, np.nan)
        self.sweeps_since_mean = 0

        self.history_length_s = processing_config.history_length_s
        self.main_peak_hist_sweep_idx = []
        self.main_peak_hist_dist = []
        self.minor_peaks_hist_sweep_idx = []
        self.minor_peaks_hist_dist = []
        self.above_thres_hist_sweep_idx = []
        self.above_thres_hist_dist = []

        self.r = et.utils.get_range_depths(sensor_config, session_info)
        self.dr = self.r[1] - self.r[0]
        self.sweep_index = 0

        self.update_processing_config(processing_config)

    def update_processing_config(self, processing_config):
        """Called when sliders or values for the detector are changed in the GUI."""
        self.nbr_average = processing_config.nbr_average
        self.threshold_type = processing_config.threshold_type
        self.peak_sorting_method = processing_config.peak_sorting_type

        self.fixed_threshold_level = processing_config.fixed_threshold

        # self.idx_cfar_pts = np.round(
        #     (
        #         processing_config.cfar_guard_cm / 100.0 / 2.0 / self.dr
        #         + np.arange(processing_config.cfar_window_cm / 100.0 / self.dr)
        #     )
        # )

        # self.cfar_one_sided = processing_config.cfar_one_sided
        # self.cfar_sensitivity = processing_config.cfar_sensitivity

        self.history_length_s = processing_config.history_length_s

    # def calculate_cfar_threshold(self, sweep, idx_cfar_pts, alpha, one_side):

    #     threshold = np.full(sweep.shape, np.nan)

    #     start_idx = np.max(idx_cfar_pts)
    #     if one_side:
    #         rel_indexes = -idx_cfar_pts
    #         end_idx = sweep.size
    #     else:
    #         rel_indexes = np.concatenate((-idx_cfar_pts, +idx_cfar_pts), axis=0)
    #         end_idx = sweep.size - start_idx

    #     for idx in np.arange(start_idx, end_idx):
    #         threshold[int(idx)] = (
    #             1.0 / (alpha + 1e-10) * np.mean(sweep[(idx + rel_indexes).astype(int)])
    #         )

    #     return threshold

    def find_first_point_above_threshold(self, sweep, threshold):

        if threshold is None or np.all(np.isnan(threshold)):
            return None

        points_above = sweep > threshold

        if not np.any(points_above):
            return None

        return np.argmax(points_above)

    def find_peaks(self, sweep, threshold):
        #  Not written for optimal speed.

        if threshold is None or np.all(np.isnan(threshold)):
            return []

        found_peaks = []

        # Note: at least 3 samples above threshold are required to form a peak

        d = 1
        N = len(sweep)
        while d < (N - 1):
            # Skip to when threshold starts, applicable only for CFAR
            if np.isnan(threshold[d - 1]):
                d += 1
                continue

            # Break when threshold ends, applicable only for CFAR
            if np.isnan(threshold[d + 1]):
                break

            # At this point, threshold is defined (not NaN)

            # If the current point is not over threshold, the next will not be a peak
            if sweep[d] <= threshold[d]:
                d += 2
                continue

            # Continue if previous point is not over threshold
            if sweep[d - 1] <= threshold[d - 1]:
                d += 1
                continue

            # Continue if this point isn't larger than the previous
            if sweep[d - 1] >= sweep[d]:
                d += 1
                continue

            # A peak is either a single point or a plateau consisting of several equal points,
            # all over their threshold. The closest neighboring points on each side of the
            # point/plateau must have a lower value and be over their threshold.
            # Now, decide if the following point(s) are a peak:

            d_upper = d + 1
            while True:
                if (d_upper) >= (N - 1):  # If out of range or on last point
                    break

                if np.isnan(threshold[d_upper]):
                    break

                if sweep[d_upper] <= threshold[d_upper]:
                    break

                if sweep[d_upper] > sweep[d]:
                    break
                elif sweep[d_upper] < sweep[d]:
                    delta = d_upper - d
                    found_peaks.append(d + int(np.ceil((delta - 1) / 2.0)))
                    break
                else:  # equal
                    d_upper += 1

            d = d_upper

        return found_peaks

    def merge_peaks(self, peak_indexes, merge_max_range):
        merged_peaks = copy(peak_indexes)

        while True:
            num_neighbors = np.zeros(len(merged_peaks))  # number of neighbors
            for i, p in enumerate(merged_peaks):
                num_neighbors[i] = np.sum(np.abs(np.array(merged_peaks) - p) < merge_max_range)

            # First peak with max number of neighbors
            i_peak = np.argmax(num_neighbors)  # returns arg of first max

            if num_neighbors[i_peak] <= 1:
                break

            peak = merged_peaks[i_peak]

            remove_mask = np.abs(np.array(merged_peaks) - peak) < merge_max_range
            peaks_to_remove = np.array(merged_peaks)[remove_mask]

            for p in peaks_to_remove:
                merged_peaks.remove(p)

            # Add back mean peak
            merged_peaks.append(int(round(np.mean(peaks_to_remove))))

            merged_peaks.sort()

        return merged_peaks

    def sort_peaks(self, peak_indexes, sweep):
        amp = np.array([sweep[int(i)] for i in peak_indexes])
        r = np.array([self.r[int(i)] for i in peak_indexes])

        PeakSorting = ProcessingConfiguration.PeakSorting
        if self.peak_sorting_method == PeakSorting.CLOSEST:
            quantity_to_sort = r
        elif self.peak_sorting_method == PeakSorting.STRONGEST:
            quantity_to_sort = -amp
        elif self.peak_sorting_method == PeakSorting.STRONGEST_REFLECTOR:
            quantity_to_sort = -amp * r ** 2
        elif self.peak_sorting_method == PeakSorting.STRONGEST_FLAT_REFLECTOR:
            quantity_to_sort = -amp * r
        else:
            raise Exception("Unknown peak sorting method")

        return [peak_indexes[i] for i in quantity_to_sort.argsort()]

    def process(self, data, data_info=None):
        """Function is called every frame and should return the struct out_data.
        
        This struct contains all processed data needed for graphs and plots.
        """
        if data_info is None:
            warnings.warn(
                "To leave out data_info or set to None is deprecated",
                DeprecationWarning,
                stacklevel=2,
            )

        sweep = data

        # Average envelope sweeps, written to handle varying nbr_average
        weight = 1.0 / (1.0 + self.sweeps_since_mean)
        self.current_mean_sweep = weight * sweep + (1.0 - weight) * self.current_mean_sweep
        self.sweeps_since_mean += 1

        # Determining threshold
        if self.threshold_type is ProcessingConfiguration.ThresholdType.FIXED:
            threshold = self.fixed_threshold_level * np.ones(sweep.size)
        elif self.threshold_type is ProcessingConfiguration.ThresholdType.CFAR:
            threshold = self.calculate_cfar_threshold(
                self.current_mean_sweep,
                self.idx_cfar_pts,
                self.cfar_sensitivity,
                self.cfar_one_sided,
            )
        else:
            print("Unknown thresholding method")

        found_peaks = None

        # If a new averaged sweep is ready for processing
        if self.sweeps_since_mean >= self.nbr_average:
            self.sweeps_since_mean = 0
            self.last_mean_sweep = self.current_mean_sweep.copy()
            self.current_mean_sweep *= 0

            # Find the first delay over threshold. Used in tank-level when monitoring changes
            # in the direct leakage.
            first_point_above_threshold = self.find_first_point_above_threshold(
                self.last_mean_sweep, threshold
            )

            # First peak-finding, then peak-merging, finally peak sorting.
            found_peaks = self.find_peaks(self.last_mean_sweep, threshold)
            if len(found_peaks) > 1:
                found_peaks = self.merge_peaks(found_peaks, np.round(PEAK_MERGE_LIMIT_M / self.dr))
                found_peaks = self.sort_peaks(found_peaks, self.last_mean_sweep)

            # Adding main peak to history
            if len(found_peaks) > 0:
                self.main_peak_hist_sweep_idx.append(self.sweep_index)
                self.main_peak_hist_dist.append(self.r[found_peaks[0]])

            # Adding minor peaks to history
            for i in range(1, len(found_peaks)):
                self.minor_peaks_hist_sweep_idx.append(self.sweep_index)
                self.minor_peaks_hist_dist.append(self.r[found_peaks[i]])

            # Adding first distance above threshold to history
            if first_point_above_threshold is not None:
                self.above_thres_hist_sweep_idx.append(self.sweep_index)
                self.above_thres_hist_dist.append(self.r[first_point_above_threshold])

            # Removing old main peaks from history
            while (
                len(self.main_peak_hist_sweep_idx) > 0
                and self.sweep_index - self.main_peak_hist_sweep_idx[0]
                > self.history_length_s * self.f
            ):
                self.main_peak_hist_sweep_idx.pop(0)
                self.main_peak_hist_dist.pop(0)

            # Removing old minor peaks from history
            while (
                len(self.minor_peaks_hist_sweep_idx) > 0
                and self.sweep_index - self.minor_peaks_hist_sweep_idx[0]
                > self.history_length_s * self.f
            ):
                self.minor_peaks_hist_sweep_idx.pop(0)
                self.minor_peaks_hist_dist.pop(0)

            # Removing old first distance above threshold from history
            while (
                len(self.above_thres_hist_sweep_idx) > 0
                and self.sweep_index - self.above_thres_hist_sweep_idx[0]
                > self.history_length_s * self.f
            ):
                self.above_thres_hist_sweep_idx.pop(0)
                self.above_thres_hist_dist.pop(0)

        out_data = {
            "sweep": sweep,
            "last_mean_sweep": self.last_mean_sweep,
            "threshold": threshold,
            "main_peak_hist_sweep_s": (
                (np.array(self.main_peak_hist_sweep_idx) - self.sweep_index) / self.f
            ),
            "main_peak_hist_dist": np.array(self.main_peak_hist_dist),
            "minor_peaks_hist_sweep_s": (
                (np.array(self.minor_peaks_hist_sweep_idx) - self.sweep_index) / self.f
            ),
            "minor_peaks_hist_dist": np.array(self.minor_peaks_hist_dist),
            "above_thres_hist_sweep_s": (
                (np.array(self.above_thres_hist_sweep_idx) - self.sweep_index) / self.f
            ),
            "above_thres_hist_dist": np.array(self.above_thres_hist_dist),
            "sweep_index": self.sweep_index,
            "found_peaks": found_peaks,
        }

        self.sweep_index += 1

        return out_data


class ProcessingConfiguration(et.configbase.ProcessingConfig):
    """Define configuration options for detector."""
    class ThresholdType(Enum):
        FIXED = "Fixed"
        RECORDED = "Recorded"
        CFAR = "CFAR"

    class PeakSorting(Enum):
        STRONGEST = "Strongest signal"
        CLOSEST = "Closest signal"
        STRONGEST_REFLECTOR = "Strongest reflector"
        STRONGEST_FLAT_REFLECTOR = "Strongest flat reflector"

    VERSION = 1

    nbr_average = et.configbase.FloatParameter(
        label="Sweep averaging",
        default_value=5,
        limits=(1, 100),
        logscale=True,
        decimals=0,
        updateable=True,
        order=0,
        visible=True,
        help=(
            "The number of envelope sweeps to be average into one then used for"
            " distance detection."
        ),
    )

    threshold_type = et.configbase.EnumParameter(
        label="Threshold type",
        default_value=ThresholdType.FIXED,
        enum=ThresholdType,
        updateable=True,
        order=5,
        help="Setting the type of threshold",
    )

    fixed_threshold = et.configbase.FloatParameter(
        label="Fixed threshold level",
        default_value=800,
        limits=(1, 20000),
        decimals=0,
        updateable=True,
        order=10,
        visible=lambda conf: conf.threshold_type == conf.ThresholdType.FIXED,
        help=(
            "Sets the value of fixed threshold. The threshold has this constant value over"
            " the full sweep."
        ),
    )

    # cfar_sensitivity = et.configbase.FloatParameter(
    #     label="CFAR sensitivity",
    #     default_value=0.5,
    #     limits=(0.01, 1),
    #     logscale=True,
    #     visible=lambda conf: conf.threshold_type == conf.ThresholdType.CFAR,
    #     decimals=4,
    #     updateable=True,
    #     order=40,
    #     help=(
    #         "Value between 0 and 1 that sets the threshold. A low sensitivity will set a "
    #         "high threshold, resulting in only few false alarms but might result in "
    #         "missed detections."
    #     ),
    # )

    # cfar_guard_cm = et.configbase.FloatParameter(
    #     label="CFAR guard",
    #     default_value=12,
    #     limits=(1, 20),
    #     unit="cm",
    #     decimals=1,
    #     visible=lambda conf: conf.threshold_type == conf.ThresholdType.CFAR,
    #     updateable=True,
    #     order=41,
    #     help=(
    #         "Range around the distance of interest that is omitted when calculating "
    #         "CFAR threshold. Can be low, ~4 cm, for Profile 1, and should be "
    #         "increased for higher Profiles."
    #     ),
    # )

    # cfar_window_cm = et.configbase.FloatParameter(
    #     label="CFAR window",
    #     default_value=3,
    #     limits=(0.1, 20),
    #     unit="cm",
    #     decimals=1,
    #     visible=lambda conf: conf.threshold_type == conf.ThresholdType.CFAR,
    #     updateable=True,
    #     order=42,
    #     help="Range next to the CFAR guard from which the threshold level will be calculated.",
    # )

    # cfar_one_sided = et.configbase.BoolParameter(
    #     label="Use only lower distance to set threshold",
    #     default_value=False,
    #     visible=lambda conf: conf.threshold_type == conf.ThresholdType.CFAR,
    #     updateable=True,
    #     order=43,
    #     help=(
    #         "Instead of determining the CFAR threshold from sweep amplitudes from "
    #         "distances both closer and a farther, use only closer. Helpful e.g. for "
    #         "fluid level in small tanks, where many multipath signal can apprear "
    #         "just after the main peak."
    #     ),
    # )

    peak_sorting_type = et.configbase.EnumParameter(
        label="Peak sorting",
        default_value=PeakSorting.STRONGEST,
        enum=PeakSorting,
        updateable=True,
        order=100,
        help="Setting the type of peak sorting method.",
    )

    history_length_s = et.configbase.FloatParameter(
        default_value=10,
        limits=(3, 1000),
        updateable=True,
        logscale=True,
        unit="s",
        label="History length",
        order=198,
        help="Length of time history for plotting.",
    )

    show_first_above_threshold = et.configbase.BoolParameter(
        label="Show first distance above threshold",
        default_value=False,
        updateable=True,
        order=199,
        help=(
            "When detecting the presence of object very close to the sensor, the "
            "strong direct leakage might cause that no well shaped peaks are detected, "
            "even though the envelope signal is above the threshold. Therefore the "
            "first distace where the signal is above the threshold can be used as an "
            "alternative to peak detection."
        ),
    )

    # Configure parameters here
    nbr_average = 5.0
    threshold_type = ThresholdType.FIXED
    fixed_threshold = 1800

    # processor_params = json.loads(processor_json)
    # for k, v in processor_params.items():
    #     try:
    #         eval(k) = eval(v)
    #     except:
    #         eval(k) = v

    def check_sensor_config(self, sensor_config):
        alerts = {
            "processing": [],
            "sensor": [],
        }
        if sensor_config.update_rate is None:
            alerts["sensor"].append(et.configbase.Error("update_rate", "Must be set"))

        if not sensor_config.noise_level_normalization:
            if self.threshold_type == self.ThresholdType.FIXED:
                alerts["sensor"].append(
                    et.configbase.Warning(
                        "noise_level_normalization",
                        (
                            "Enabling noise level normalization is\n"
                            "recommended with Fixed threshold"
                        ),
                    )
                )

        return alerts


get_processing_config = ProcessingConfiguration

if __name__ == "__main__":
    main()
