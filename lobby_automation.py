import os
import time

import cv2
from lobby_ocr import DeepSeekOCRv2, LobbyOCRError, extract_text_and_positions, lobby_ocr_settings
from utils import (
    count_hsv_pixels,
    load_toml_as_dict, config_bool, load_brawlers_info,
    normalize_brawler_filename,
)


class LobbyAutomation:

    def __init__(self, window_controller, ocr_reader=None):
        self.gray_pixels_treshold = load_toml_as_dict("./cfg/bot_config.toml").get('idle_pixels_minimum', 500)
        self.idle_reconnect_coords = load_toml_as_dict("cfg/buttons_config.toml")["idle_reconnect"]
        if self.idle_reconnect_coords and isinstance(self.idle_reconnect_coords[0], (int, float)):
            self.idle_reconnect_coords = [self.idle_reconnect_coords]
        self.window_controller = window_controller
        self.verbose_debug = config_bool(load_toml_as_dict("cfg/debug_settings.toml").get('verbose_debug'), False)
        self.idle_disconnect_hsv_high_bounds = load_toml_as_dict("cfg/lobby_config.toml").get("hsv_bounds", {}).get("idle_reconnect_high_bounds", [[10, 22, 42], [10, 22, 90], [118, 66, 46]])
        ocr_settings = lobby_ocr_settings()
        self.ocr_reader = ocr_reader or DeepSeekOCRv2.from_lobby_config()
        self.ocr_scale_down_factor = ocr_settings["scale_down_factor"]
        self.ocr_scale_up_factor = 1 / self.ocr_scale_down_factor

    def check_for_idle(self, frame):
        wr = self.window_controller.width_ratio
        hr = self.window_controller.height_ratio
        x_start, x_end = int(460 * wr), int(1460 * wr)
        y_start, y_end = int(400 * hr), int(675 * hr)
        if self.verbose_debug:
            print(f"gray pixels (if > {self.gray_pixels_treshold} then bot will try to unidle)")
        for idle_disconnect_hsv_high_bound in self.idle_disconnect_hsv_high_bounds:
            gray_pixels = count_hsv_pixels(frame[y_start:y_end, x_start:x_end], (0, 0, 0), tuple(idle_disconnect_hsv_high_bound), self.window_controller)
            if self.verbose_debug:
                try:
                    cv2.imwrite(f"./debug_frames/idle_detection_{gray_pixels}_{len(os.listdir('./debug_frames'))}.png", cv2.cvtColor(frame[y_start:y_end, x_start:x_end], cv2.COLOR_BGR2RGB))
                except Exception:
                    pass
            if gray_pixels > self.gray_pixels_treshold:
                print("Idle detected, clicking to unidle")
                for idle_reconnect_coord in self.idle_reconnect_coords:
                    self.window_controller.click(idle_reconnect_coord[0], idle_reconnect_coord[1], already_include_ratio=False)


    @staticmethod
    def _should_interrupt(runtime_control=None, stop_event=None):
        if runtime_control and (runtime_control.should_stop() or runtime_control.should_pause()):
            return True
        return stop_event is not None and stop_event.is_set()

    @staticmethod
    def _sleep_interruptible(duration, runtime_control=None, stop_event=None, poll_interval=0.1):
        end_time = time.time() + duration
        while time.time() < end_time:
            if LobbyAutomation._should_interrupt(runtime_control, stop_event):
                return True
            time.sleep(min(poll_interval, max(end_time - time.time(), 0)))
        return False

    def select_brawler(self, brawler, get_latest_state, stop_event=None, runtime_control=None):
        self.window_controller.screenshot()
        wr = self.window_controller.width_ratio
        hr = self.window_controller.height_ratio
        brawler = str(brawler).lower().strip()
        normalized_brawler = normalize_brawler_filename(brawler)
        brawler_info = load_brawlers_info().get(normalized_brawler, {})
        target_names = {normalized_brawler}
        actual_name = brawler_info.get("actual_name")
        if actual_name:
            target_names.add(normalize_brawler_filename(actual_name))

        x, y = load_toml_as_dict("cfg/buttons_config.toml")["brawlers_menu"]
        self.window_controller.click(x, y, already_include_ratio=False)
        time.sleep(0.5)
        print("Automatic brawler selection started for", actual_name or normalized_brawler)
        scrolled_once = False
        shop_counter = 0
        for _i in range(100):
            if self._should_interrupt(runtime_control, stop_event):
                print("Brawler selection aborted by user.")
                return "aborted"
            screenshot = self.window_controller.screenshot()
            screenshot = cv2.resize(
                screenshot,
                (
                    int(screenshot.shape[1] * self.ocr_scale_down_factor),
                    int(screenshot.shape[0] * self.ocr_scale_down_factor),
                ),
                interpolation=cv2.INTER_AREA,
            )

            print("Extracting text on current screen with DeepSeek OCR v2...")
            try:
                if isinstance(self.ocr_reader, DeepSeekOCRv2):
                    document = self.ocr_reader.read_brawler_sync(screenshot, target_names)
                    results = {}
                    for item in document["menu_text"]:
                        bbox = item["bbox"]
                        results[item["text"].lower()] = {
                            "center": ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                        }
                else:
                    results = extract_text_and_positions(screenshot, self.ocr_reader)
            except LobbyOCRError as exc:
                print(f"WARNING: Automatic brawler selection could not start OCR: {exc}")
                print("The bot will continue without changing the currently selected brawler.")
                return "error"
            except Exception as exc:
                print(f"WARNING: Automatic brawler selection could not read this screen with OCR: {exc}")
                print("The bot will continue without changing the currently selected brawler.")
                return "error"

            clean_results = {}
            for key, value in results.items():
                if len(key) < 2:
                    continue
                clean_results[normalize_brawler_filename(key)] = value

            current_state = get_latest_state()
            if "shop" in clean_results:
                print("Latest screenshot is still of the lobby, waiting for the frame to update...")
                shop_counter += 1
                if shop_counter > 5:
                    print("WARNING: The bot has been waiting for the lobby screen to update for a long time. It's possible that the game is stuck or the OCR is having trouble reading the screen. The bot will continue without changing the currently selected brawler.")
                    return "stuck"
                continue
            if current_state != "brawler_selection":
                print("Latest screenshot is no longer of the lobby, aborting brawler selection...")
                return "stuck"

            matched_key = next((name for name in clean_results if name in target_names), None)
            if self.verbose_debug:
                print("DeepSeek OCR v2 detected:", list(clean_results.keys()))
            if matched_key:
                x, y = clean_results[matched_key]["center"]
                y -= 50 * self.ocr_scale_down_factor
                click_x = int(x * self.ocr_scale_up_factor)
                click_y = int(y * self.ocr_scale_up_factor)
                self.window_controller.click(click_x, click_y)
                print(f"Found brawler {normalized_brawler} ({matched_key}) clicking on its icon at {click_x} {click_y}")
                if self._sleep_interruptible(1, runtime_control, stop_event):
                    print("Brawler selection aborted by user.")
                    return "aborted"
                select_x, select_y = load_toml_as_dict("cfg/buttons_config.toml")["select_brawler"]
                self.window_controller.click(select_x, select_y, already_include_ratio=False)
                if self._sleep_interruptible(1.5, runtime_control, stop_event):
                    print("Brawler selection aborted by user.")
                    return "aborted"
                self.window_controller.screenshot()
                print("Selected brawler ", normalized_brawler)
                return "success"

            print("Brawler name not found on screen, scrolling down to load more brawlers...")
            if not scrolled_once:
                self.window_controller.swipe(int(1700 * wr), int(900 * hr), int(1700 * wr), int(850 * hr), duration=0.5)
                scrolled_once = True
            else:
                self.window_controller.swipe(int(1700 * wr), int(900 * hr), int(1700 * wr), int(650 * hr), duration=0.5)
            if self._sleep_interruptible(3, runtime_control, stop_event):
                print("Brawler selection aborted by user.")
                return "aborted"

        print(f"WARNING: Brawler '{brawler}' was not found after 100 scroll attempts.")
        return "failed"
