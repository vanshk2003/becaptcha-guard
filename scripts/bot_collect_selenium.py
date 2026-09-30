"""Collect *real* automation telemetry by driving the collection UI with Selenium.

This is how you gather genuine bot-family data (rather than the synthetic model):
launch a real browser under automation, load the collector in auto mode with a
label, and let Selenium acquire the targets and type. The browser emits real
DOM events, so the captured timing/motion is authentic automation behaviour —
exactly the adversary the detector must catch (cf. Plesner et al., 'Breaking
reCAPTCHAv2', COMPSAC 2024).

Requires:  pip install selenium   (and a matching chromedriver on PATH)
API must be running.

Run:
    python scripts/bot_collect_selenium.py --url http://127.0.0.1:8000 \
        --label selenium_linear --trials 20
"""
import argparse
import time

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.chrome.options import Options
except ImportError:
    raise SystemExit("Install selenium first:  pip install selenium")


def run(url, label, trials):
    opts = Options()
    opts.add_argument("--window-size=1280,900")
    driver = webdriver.Chrome(options=opts)
    try:
        for n in range(trials):
            driver.get(f"{url}/collect-ui?label={label}&auto=1")
            time.sleep(0.6)
            # Task 1: acquire each target as it appears (naive linear move = Selenium tell)
            for _ in range(8):
                try:
                    target = driver.find_element(By.CLASS_NAME, "target")
                except Exception:
                    break
                ActionChains(driver).move_to_element(target).click().perform()
                time.sleep(0.15)
            # Task 2: type the phrase, then store
            try:
                box = driver.find_element(By.ID, "ta")
                box.click()
                box.send_keys("the quick brown fox jumps")
                time.sleep(0.2)
                driver.find_element(By.ID, "store").click()
                time.sleep(0.3)
            except Exception:
                pass
            print(f"  trial {n+1}/{trials} submitted as label={label}")
    finally:
        driver.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", default="selenium_linear",
                    choices=["selenium_linear", "puppeteer_step", "bezier_bot", "minjerk_adv"])
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()
    run(args.url, args.label, args.trials)
