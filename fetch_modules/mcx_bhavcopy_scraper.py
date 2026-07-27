"""
MCX India BhavCopy Scraper – CRUDEOIL FUTCOM, all expiries
===========================================================
Navigates to the Commodity Wise tab, sets Instrument=FUTCOM,
Symbol=CRUDEOIL (exact match), checks "All dates", then downloads
one CSV per expiry in chronological order (oldest → newest) into
./data/com_futures/crudeoil/

Key design decisions
--------------------
* Symbol is matched with :text-is() (exact) so "BRCRUDEOIL" and other
  symbols that merely contain "CRUDEOIL" are never picked by mistake.
* Expiry list is reversed after collection so iteration goes oldest→newest
  (05APR2005 … 21SEP2026).
* All form interactions use real UI clicks so ASP.NET / Telerik AJAX
  event chains fire and populate dependent dropdowns correctly.
* CSV download uses a direct .click() on #lnkExportToCSV instead of
  page.evaluate(__doPostBack) — the latter fails in Playwright's strict
  mode because the site's legacy WebForms code accesses `arguments`.
* `page` is passed explicitly to every helper that needs wait_for_load_state
  – there is no frame.page() call anywhere.

Usage
-----
    python mcx_bhavcopy_scraper.py                   # headless, resume
    python mcx_bhavcopy_scraper.py --headless false  # watch the browser
    python mcx_bhavcopy_scraper.py --no-resume       # force re-download
    python mcx_bhavcopy_scraper.py --verbose         # debug logging

Install
-------
    pip install playwright
    playwright install chromium
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Union
from playwright.async_api import (
    async_playwright,
    Page,
    Frame,
    TimeoutError as PWTimeout,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL         = "https://www.mcxindia.com/market-data/bhavcopy"
INSTRUMENT       = "FUTCOM"
SYMBOL           = "CRUDEOIL"          # must be an exact match in the dropdown
DATA_DIR         = Path("data/com_futures/crudeoil")
TIMEOUT          = 45_000   # ms – element waits
NAV_TIMEOUT      = 90_000   # ms – page / network-idle
DOWNLOAD_TIMEOUT = 90_000   # ms – file download
RETRY_LIMIT      = 3        # per-expiry retries
PAUSE_S          = 2.0      # seconds between expiries

AnyFrame = Union[Page, Frame]

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("mcx_scraper")

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("mcx_scraper.log", mode="w", encoding="utf-8"),
        ],
    )

# ── Network settle helper ─────────────────────────────────────────────────────
async def wait_for_network(page: Page, timeout: int = 20_000) -> None:
    """Wait for networkidle; silently swallow timeout so callers don't crash."""
    try:
        await page.wait_for_load_state("networkidle", timeout=timeout)
    except PWTimeout:
        pass

# ── Frame resolution ──────────────────────────────────────────────────────────
async def get_working_frame(page: Page) -> AnyFrame:
    """Return the frame/page that hosts the bhavcopy UI."""
    try:
        await page.wait_for_selector(".maToggle", timeout=8_000)
        log.debug("Bhavcopy UI found in main page frame.")
        return page
    except PWTimeout:
        pass
    for frame in page.frames:
        try:
            if await frame.query_selector(".maToggle"):
                log.debug(f"Bhavcopy UI found in child frame: {frame.url}")
                return frame
        except Exception:
            continue
    log.warning(".maToggle not found in any frame – defaulting to main page.")
    return page

# ── Tab switching ─────────────────────────────────────────────────────────────
async def switch_to_commodity_wise(page: Page, frame: AnyFrame) -> None:
    """Click the 'Commodity Wise' toggle and wait for its panel."""
    log.info("Switching to Commodity Wise tab …")
    clicked = False
    for sel in [".maToggle .all", ".bhavcopytopsec .all", ".maToggle div.all",
                "text=Commodity Wise"]:
        try:
            loc = frame.locator(sel).first
            await loc.wait_for(state="visible", timeout=6_000)
            await loc.scroll_into_view_if_needed()
            await loc.click()
            clicked = True
            log.debug(f"Tab clicked via '{sel}'.")
            break
        except Exception as exc:
            log.debug(f"Selector '{sel}' failed: {exc}")
    if not clicked:
        log.warning("CSS selectors failed – JS click fallback.")
        await frame.evaluate("""
            () => {
                var el = document.querySelector('.maToggle .all')
                    || document.querySelector('.bhavcopytopsec .all');
                if (!el) {
                    for (var d of document.querySelectorAll('div'))
                        if (d.textContent.trim() === 'Commodity Wise') { el = d; break; }
                }
                if (el) el.click();
            }
        """)
    try:
        await frame.wait_for_selector("#commoditywise", state="visible", timeout=TIMEOUT)
        log.info("Commodity Wise panel is visible ✓")
    except PWTimeout:
        log.warning("Panel still hidden – forcing display via JS.")
        await frame.evaluate("""
            () => {
                var cw = document.getElementById('commoditywise');
                var dw = document.getElementById('datewise');
                if (cw) { cw.style.display = 'block'; cw.style.visibility = 'visible'; }
                if (dw) dw.style.display = 'none';
            }
        """)
    await asyncio.sleep(0.8)

# ── Instrument ────────────────────────────────────────────────────────────────
async def set_instrument(page: Page, frame: AnyFrame, instrument: str) -> None:
    """Select instrument via the real <select> so the onChange postback fires."""
    log.info(f"Setting instrument → {instrument}")
    await frame.wait_for_selector("#ddlInstrument", state="visible", timeout=TIMEOUT)
    current = await frame.eval_on_selector("#ddlInstrument", "el => el.value")
    if current.strip() == instrument:
        log.info(f"Instrument already {instrument} – no change needed.")
        return
    await frame.select_option("#ddlInstrument", value=instrument)
    await wait_for_network(page, timeout=20_000)
    await asyncio.sleep(1.5)
    log.info(f"Instrument set to {instrument} ✓")

# ── Symbol – exact UI match ───────────────────────────────────────────────────
async def set_symbol_ui(page: Page, frame: AnyFrame, symbol: str) -> None:
    """
    Select the symbol via genuine UI interaction with the Telerik RadComboBox
    using EXACT text matching so that e.g. 'BRCRUDEOIL' is never chosen when
    'CRUDEOIL' is requested.

    Waterfall:
      1. Open arrow → wait for list → click item with :text-is('<symbol>')
      2. Type-to-filter → click exact match
      3. Last resort: press Enter (warns if used)
    """
    log.info(f"Setting symbol → {symbol} (exact match)")
    await frame.wait_for_selector("#ddlSymbols_Input", state="visible", timeout=TIMEOUT)

    # ── Open the dropdown ─────────────────────────────────────────────────────
    opened = False
    for arrow_sel in ["#ddlSymbols_Arrow", "a[id='ddlSymbols_Arrow']", ".rcbArrowCell a"]:
        try:
            loc = frame.locator(arrow_sel).first
            await loc.wait_for(state="visible", timeout=5_000)
            await loc.click()
            opened = True
            break
        except Exception:
            continue
    if not opened:
        try:
            await frame.click("#ddlSymbols_Input")
        except Exception:
            pass
    await asyncio.sleep(0.8)

    # ── Wait for the list container ───────────────────────────────────────────
    list_visible = False
    for list_sel in [".rcbList", "ul.rcbList", ".RadComboBoxDropDown"]:
        try:
            await frame.wait_for_selector(list_sel, state="visible", timeout=5_000)
            list_visible = True
            break
        except Exception:
            continue

    # ── EXACT match selectors (:text-is prevents partial matches) ─────────────
    exact_selectors = [
        f".rcbList li:text-is('{symbol}')",
        f"ul.rcbList li:text-is('{symbol}')",
        f".RadComboBoxDropDown li:text-is('{symbol}')",
    ]
    if list_visible:
        for sel in exact_selectors:
            try:
                await frame.wait_for_selector(sel, state="visible", timeout=5_000)
                await frame.click(sel)
                log.info(f"Symbol '{symbol}' selected via exact list click ✓")
                await asyncio.sleep(0.5)
                return
            except Exception:
                continue

    # ── Fallback: type the symbol to filter the list, then exact-click ────────
    log.warning("Direct list click failed – type-to-filter fallback.")
    try:
        inp = frame.locator("#ddlSymbols_Input").first
        await inp.triple_click()
        await inp.fill("")
        await inp.type(symbol, delay=80)
        await asyncio.sleep(1.2)
        for sel in exact_selectors:
            try:
                await frame.wait_for_selector(sel, state="visible", timeout=4_000)
                await frame.click(sel)
                log.info(f"Symbol '{symbol}' selected via type-to-filter exact click ✓")
                await asyncio.sleep(0.5)
                return
            except Exception:
                continue
        try:
            visible_items = await frame.eval_on_selector_all(
                ".rcbList li",
                "nodes => nodes.map(n => n.textContent.trim()).slice(0, 10)"
            )
            log.warning(f"Visible list items: {visible_items}")
        except Exception:
            pass
        await frame.press("#ddlSymbols_Input", "Enter")
        log.warning("Pressed Enter as last resort – verify correct symbol was selected.")
        await asyncio.sleep(0.5)
    except Exception as exc:
        log.error(f"Could not set symbol: {exc}")
        raise

# ── Poll expiry dropdown ──────────────────────────────────────────────────────
async def wait_for_expiry_populated(
    frame: AnyFrame,
    min_count: int = 1,
    timeout_s: float = 40.0,
) -> list:
    """Poll until #ddlExpiry has ≥ min_count real values. Returns value list."""
    log.info("Waiting for expiry dropdown to populate …")
    loop     = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        try:
            options = await frame.eval_on_selector_all(
                "#ddlExpiry option",
                "nodes => nodes.map(o => o.value).filter(v => v && v !== 'Select')",
            )
            if len(options) >= min_count:
                log.info(f"Expiry dropdown populated: {len(options)} dates ✓")
                return options
        except Exception:
            pass
        await asyncio.sleep(0.5)
    log.error(f"Expiry dropdown still empty after {timeout_s}s.")
    return []

# ── All-dates checkbox ────────────────────────────────────────────────────────
async def check_all_dates(frame: AnyFrame) -> None:
    log.info("Enabling 'All dates' checkbox …")
    try:
        chk = frame.locator("#chkAll").first
        await chk.wait_for(state="visible", timeout=TIMEOUT)
        if not await chk.is_checked():
            await chk.click()
            await asyncio.sleep(0.4)
        log.info("'All dates' is checked ✓")
    except Exception as exc:
        log.warning(f"Checkbox UI click failed ({exc}) – JS fallback.")
        await frame.evaluate("""
            () => {
                var c = document.getElementById('chkAll');
                if (c && !c.checked) {
                    c.checked = true;
                    c.dispatchEvent(new Event('change', {bubbles: true}));
                }
            }
        """)

# ── Full form setup ───────────────────────────────────────────────────────────
async def configure_form(page: Page, frame: AnyFrame) -> list:
    """
    Instrument → Symbol (real UI exact-click) → wait for expiry AJAX → All dates.
    Returns expiry list sorted oldest → newest.
    """
    await set_instrument(page, frame, INSTRUMENT)
    await set_symbol_ui(page, frame, SYMBOL)
    await wait_for_network(page, timeout=20_000)
    await asyncio.sleep(1.5)
    expiries = await wait_for_expiry_populated(frame, min_count=1, timeout_s=40.0)
    expiries = list(reversed(expiries))
    await check_all_dates(frame)
    return expiries

# ── Per-expiry download ───────────────────────────────────────────────────────
async def download_expiry(
    page: Page,
    frame: AnyFrame,
    expiry: str,
    out_path: Path,
) -> tuple:
    """
    Select expiry, click Show, download CSV.
    Returns (success: bool, frame: AnyFrame).

    FIX: The CSV export anchor (#lnkExportToCSV) is clicked directly via
    Playwright's .click() instead of calling __doPostBack() through
    page.evaluate().  Playwright's evaluate context runs in strict mode,
    which raises a TypeError when the legacy ASP.NET WebForms runtime tries
    to access the `arguments` object inside _doPostBack.  A native browser
    click on the <a href="javascript:__doPostBack(...)"> element bypasses
    that restriction entirely because the browser executes the href in its
    own non-strict context.
    """
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            log.info(f"    attempt {attempt}/{RETRY_LIMIT} – {expiry}")

            # ── Guard: re-configure if form was reset ─────────────────────────
            instr_val = await frame.eval_on_selector(
                "#ddlInstrument", "el => el.value"
            )
            sym_val = ""
            try:
                sym_val = await frame.input_value("#ddlSymbols_Input")
            except Exception:
                pass
            if (
                instr_val.strip() != INSTRUMENT
                or sym_val.strip().upper() != SYMBOL.upper()
            ):
                log.debug("    Form state reset – reconfiguring.")
                await configure_form(page, frame)

            # ── Select expiry ─────────────────────────────────────────────────
            await frame.wait_for_selector(
                "#ddlExpiry", state="visible", timeout=TIMEOUT
            )
            await frame.select_option("#ddlExpiry", value=expiry)
            await asyncio.sleep(0.5)

            # ── Click Show ────────────────────────────────────────────────────
            show_clicked = False
            for show_sel in ["#btnShowCommoditywise", "a[id='btnShowCommoditywise']"]:
                try:
                    loc = frame.locator(show_sel).first
                    await loc.wait_for(state="visible", timeout=6_000)
                    await loc.scroll_into_view_if_needed()
                    await loc.click()
                    show_clicked = True
                    break
                except Exception:
                    continue
            if not show_clicked:
                await frame.evaluate(
                    "() => { var a = document.getElementById('btnShowCommoditywise');"
                    " if (a) a.click(); }"
                )
            await wait_for_network(page, timeout=NAV_TIMEOUT)
            await asyncio.sleep(1.0)

            # ── Skip if no data ───────────────────────────────────────────────
            try:
                body = (await frame.inner_text("body")).lower()
                if "no records" in body or "no data found" in body:
                    log.warning(f"    No data for {expiry} – skipping.")
                    return True, frame
            except Exception:
                pass

            # ── Download CSV ──────────────────────────────────────────────────
            # Clicking the anchor directly lets the browser execute the
            # href="javascript:__doPostBack(...)" in its own (non-strict) context,
            # avoiding the strict-mode TypeError that occurs when the same call is
            # made through Playwright's page.evaluate().
            csv_btn_sel = "#lnkExportToCSV"
            await frame.wait_for_selector(csv_btn_sel, state="visible", timeout=TIMEOUT)
            csv_btn = frame.locator(csv_btn_sel).first
            await csv_btn.scroll_into_view_if_needed()

            async with page.expect_download(timeout=DOWNLOAD_TIMEOUT) as dl_info:
                await csv_btn.click()

            dl = await dl_info.value
            await dl.save_as(out_path)
            size = out_path.stat().st_size
            if size < 20:
                out_path.unlink(missing_ok=True)
                raise ValueError(f"File too small ({size} B) – likely an error page.")

            log.info(f"    ✓  {out_path.name}  ({size:,} bytes)")
            return True, frame

        except PWTimeout as exc:
            log.warning(f"    ✗  Timeout on attempt {attempt}: {exc}")
        except Exception as exc:
            log.warning(f"    ✗  Error on attempt {attempt}: {exc}")

        if attempt < RETRY_LIMIT:
            wait_s = 4 * attempt
            log.info(f"    Backing off {wait_s}s …")
            await asyncio.sleep(wait_s)
            try:
                await frame.wait_for_selector(
                    "#commoditywise", state="visible", timeout=5_000
                )
            except PWTimeout:
                log.info("    Panel gone – reloading and reconfiguring …")
                await page.goto(BASE_URL, wait_until="networkidle", timeout=NAV_TIMEOUT)
                await asyncio.sleep(2)
                frame = await get_working_frame(page)
                await switch_to_commodity_wise(page, frame)
                await configure_form(page, frame)

    log.error(f"    ✗✗  All {RETRY_LIMIT} attempts failed for {expiry}.")
    return False, frame

# ── Main ──────────────────────────────────────────────────────────────────────
async def run(headless: bool = True, resume: bool = True) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            accept_downloads=True,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1366, "height": 900},
        )
        context.set_default_timeout(TIMEOUT)
        page = await context.new_page()
        page.set_default_timeout(TIMEOUT)

        # ── Navigate ──────────────────────────────────────────────────────────
        log.info(f"Opening {BASE_URL} …")
        try:
            await page.goto(BASE_URL, wait_until="networkidle", timeout=NAV_TIMEOUT)
        except PWTimeout:
            log.warning("networkidle timed out – falling back to domcontentloaded.")
            await page.goto(
                BASE_URL, wait_until="domcontentloaded", timeout=NAV_TIMEOUT
            )
            await asyncio.sleep(5)
        log.info("Page loaded.")

        frame    = await get_working_frame(page)
        await switch_to_commodity_wise(page, frame)
        expiries = await configure_form(page, frame)

        if not expiries:
            log.error(
                "Expiry dropdown is still empty.\n"
                "  • Run with --headless false to watch and diagnose.\n"
                "  • The site may be presenting a CAPTCHA or blocking automation."
            )
            await browser.close()
            return

        log.info(
            f"Starting download: {len(expiries)} expiries, "
            f"{expiries[0]} → {expiries[-1]}  |  output → {DATA_DIR.resolve()}"
        )

        succeeded, skipped, failed = [], [], []
        for idx, expiry in enumerate(expiries, start=1):
            out_path = DATA_DIR / f"CRUDEOIL_FUTCOM_{expiry}.csv"
            log.info(f"[{idx:>3}/{len(expiries)}]  {expiry}")

            if resume and out_path.exists() and out_path.stat().st_size > 20:
                log.info("    ↷  Already downloaded – skipping.")
                skipped.append(expiry)
                continue

            ok, frame = await download_expiry(page, frame, expiry, out_path)
            (succeeded if ok else failed).append(expiry)
            await asyncio.sleep(PAUSE_S)

        # ── Summary ───────────────────────────────────────────────────────────
        log.info("=" * 60)
        log.info(
            f"DONE —  ✓ {len(succeeded)} downloaded  |  "
            f"↷ {len(skipped)} skipped  |  ✗ {len(failed)} failed"
        )
        if failed:
            log.warning(f"Failed expiries: {failed}")
            Path("failed_expiries.txt").write_text(
                "\n".join(failed), encoding="utf-8"
            )
            log.info("Saved failed_expiries.txt — re-run with --no-resume to retry.")

        await browser.close()

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MCX BhavCopy Scraper – FUTCOM/CRUDEOIL CSV per expiry, oldest first."
    )
    p.add_argument(
        "--headless",
        default="true",
        choices=["true", "false"],
        help="Run headlessly (default: true). Use 'false' to watch.",
    )
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip already-downloaded files (default: on).",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Re-download everything.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    setup_logging(verbose=args.verbose)
    asyncio.run(
        run(
            headless=(args.headless.lower() == "true"),
            resume=args.resume,
        )
    )