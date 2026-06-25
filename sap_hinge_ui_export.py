"""Pilot SAP2000 UI automation for exporting real hinge-result tables.

SAP2000 v22 exposes nonlinear saved-state settings through OAPI but does not
expose DatabaseTables or FrameHinge result readers in its local COM type
library. This utility opens one analyzed .sdb model, opens Show Tables through
the desktop UI, and records the visible controls. The resulting probe JSON is
used to lock down a reliable batch exporter for this installation.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


SAP_EXE = Path(r"C:\Program Files\Computers and Structures\SAP2000 22\SAP2000.exe")
TREE_VERTICAL_OFFSET_PX = 137
PUSHOVER_OUTPUT_CASES = ("PUSHOVER_X", "PUSHOVER_Y")


def open_and_probe(model_path: Path, output_path: Path, wait_seconds: float = 8.0) -> dict[str, Any]:
    """Open one SAP2000 model, invoke Show Tables, and write UI control inventory."""
    if not SAP_EXE.exists():
        raise FileNotFoundError(f"SAP2000 executable not found: {SAP_EXE}")
    if not model_path.exists():
        raise FileNotFoundError(f"SAP2000 model not found: {model_path}")
    try:
        from pywinauto import Application, Desktop, keyboard
    except ImportError as exc:
        raise RuntimeError("pywinauto is required. Install with: python -m pip install --user pywinauto") from exc

    process = subprocess.Popen([str(SAP_EXE)], close_fds=True)
    deadline = time.time() + wait_seconds
    main_window = None
    while time.time() < deadline:
        windows = Desktop(backend="win32").windows(process=process.pid, visible_only=True)
        ready_windows = [window for window in windows if window.window_text() == "SAP2000" or window.window_text().startswith("SAP2000 v")]
        if ready_windows:
            main_window = ready_windows[0]
            break
        time.sleep(0.5)
    if main_window is None:
        raise RuntimeError("SAP2000 main window was not found.")

    _open_model_through_com(model_path)
    app = Application(backend="win32").connect(handle=main_window.handle)
    window = app.window(handle=main_window.handle)
    _safe_focus(window)
    # SAP2000 v22 uses a WinForms MenuStrip which pywinauto cannot traverse as
    # a classic native menu. The built-in Ctrl+T accelerator is stable and is
    # shown next to Display > Show Tables in the desktop UI.
    keyboard.send_keys("^t")
    time.sleep(1.5)

    controls = []
    top_windows = Desktop(backend="win32").windows(process=window.process_id(), visible_only=True)
    for top in top_windows:
        controls.append(_control_tree(top))
    result = {
        "model_path": str(model_path.resolve()),
        "sap_exe": str(SAP_EXE),
        "main_window_title": window.window_text(),
        "show_tables_shortcut": "Ctrl+T",
        "visible_windows": controls,
        "note": "Use this probe JSON to map the local SAP2000 v22 Show Tables dialog before enabling unattended batch export.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _open_model_through_com(model_path: Path) -> None:
    """Open an .sdb through COM because SAP2000 v22 ignores CLI model paths."""
    try:
        import comtypes.client
    except ImportError as exc:
        raise RuntimeError("comtypes is required for opening the SAP2000 model.") from exc
    deadline = time.time() + 30.0
    sap_object = None
    while time.time() < deadline:
        try:
            sap_object = comtypes.client.GetActiveObject("CSI.SAP2000.API.SapObject")
            break
        except OSError:
            time.sleep(0.5)
    if sap_object is None:
        raise RuntimeError("SAP2000 COM object was not available after startup.")
    ret = sap_object.SapModel.File.OpenFile(str(model_path.resolve()))
    if int(ret) != 0:
        raise RuntimeError(f"SAP2000 File.OpenFile failed for {model_path}. Return code: {ret}")
    time.sleep(2.0)


def _rerun_pushover_analysis_through_com(output_cases: tuple[str, ...] = PUSHOVER_OUTPUT_CASES) -> dict[str, Any]:
    """Rerun X/Y pushover cases before export because v22 may reopen .sdb without hinge rows."""
    try:
        import comtypes.client
    except ImportError as exc:
        raise RuntimeError("comtypes is required for rerunning SAP2000 pushover cases.") from exc
    sap_object = comtypes.client.GetActiveObject("CSI.SAP2000.API.SapObject")
    sap_model = sap_object.SapModel
    unlock_ret = sap_model.SetModelIsLocked(False)
    if int(unlock_ret) != 0:
        raise RuntimeError(f"SAP2000 SetModelIsLocked(False) failed before worker reanalysis: {unlock_ret}")
    ret = sap_model.Analyze.SetRunCaseFlag("", False, True)
    if int(ret) != 0:
        raise RuntimeError(f"SAP2000 Analyze.SetRunCaseFlag(clear all) failed: {ret}")
    for case_name in output_cases:
        ret = sap_model.Analyze.SetRunCaseFlag(case_name, True, False)
        if int(ret) != 0:
            raise RuntimeError(f"SAP2000 Analyze.SetRunCaseFlag({case_name}) failed: {ret}")
    ret = sap_model.Analyze.RunAnalysis()
    if int(ret) != 0:
        raise RuntimeError(f"SAP2000 Analyze.RunAnalysis failed: {ret}")
    save_ret = sap_model.File.Save("")
    if int(save_ret) != 0:
        raise RuntimeError(f"SAP2000 File.Save(current model) failed after worker reanalysis: {save_ret}")
    status_result = sap_model.Analyze.GetCaseStatus(0, [], [])
    time.sleep(2.0)
    return {
        "requested_cases": list(output_cases),
        "case_status_raw": repr(status_result),
        "message": "Cached results cleared by unlocking the model; pushover cases rerun inside the foreground export worker before reading Frame Hinge States.",
    }


def export_model_hinge_result_tables(
    model_path: Path,
    output_dir: Path,
    wait_seconds: float = 120.0,
    output_cases: tuple[str, ...] = PUSHOVER_OUTPUT_CASES,
    export_stem: str | None = None,
) -> dict[str, Any]:
    """Export one analyzed model in an isolated disposable SAP2000 process."""
    if not SAP_EXE.exists():
        raise FileNotFoundError(f"SAP2000 executable not found: {SAP_EXE}")
    if not model_path.exists():
        raise FileNotFoundError(f"SAP2000 model not found: {model_path}")

    try:
        from pywinauto import Desktop
    except ImportError as exc:
        raise RuntimeError("pywinauto is required for exact SAP hinge export.") from exc

    existing_excel_processes = _windows_process_ids("EXCEL.EXE")
    process = subprocess.Popen([str(SAP_EXE)], close_fds=True)
    try:
        _find_active_sap_main_window(Desktop, wait_seconds, process.pid)
        _open_model_through_com(model_path)
        analysis_status = _rerun_pushover_analysis_through_com(output_cases)
        result = export_active_hinge_result_tables(output_dir, export_stem or model_path.stem, wait_seconds, process.pid, output_cases)
        result["worker_reanalysis"] = analysis_status
        return result
    finally:
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        for excel_process_id in _windows_process_ids("EXCEL.EXE") - existing_excel_processes:
            subprocess.run(
                ["taskkill", "/PID", str(excel_process_id), "/F"],
                capture_output=True,
                text=True,
                check=False,
            )


def export_active_hinge_result_tables(
    output_dir: Path,
    model_stem: str,
    wait_seconds: float = 20.0,
    process_id: int | None = None,
    output_cases: tuple[str, ...] = PUSHOVER_OUTPUT_CASES,
) -> dict[str, Any]:
    """Export exact SAP hinge-result tables from the active solved SAP session.

    SAP2000 v22 does not expose its result-table API through the installed COM
    type library. This function uses the desktop Show Tables form immediately
    after analysis, then asks SAP to export each hinge-result table to Excel.
    Excel COM writes deterministic UTF-8 CSV files for downstream calibration.
    """
    try:
        from pywinauto import Desktop, keyboard
    except ImportError as exc:
        raise RuntimeError("pywinauto is required for exact SAP hinge export.") from exc

    raw_dir = output_dir / "hinge_validation" / "raw_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    main_window = _find_active_sap_main_window(Desktop, wait_seconds, process_id)
    dialog = _open_show_tables_dialog(Desktop, keyboard, main_window, wait_seconds)
    width = dialog.rectangle().width()
    height = dialog.rectangle().height()
    dialog.move_window(x=0, y=0, width=width, height=height, repaint=True)
    time.sleep(0.5)
    _select_output_cases(Desktop, dialog, main_window.process_id(), output_cases, wait_seconds)
    _select_nonlinear_step_by_step(Desktop, dialog, main_window.process_id(), wait_seconds)

    tree = next(child for child in dialog.children() if "SysTreeView32" in child.class_name())
    inventory = _table_tree_inventory(tree)
    inventory_path = raw_dir / f"{model_stem}_table_tree.json"
    inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    hinge_tables = [
        row for row in inventory
        if row["is_table"]
        and "hinge" in row["title"].lower()
        and row["root"].strip().upper().startswith("ANALYSIS RESULTS")
    ]
    if not hinge_tables:
        _click_named_button(dialog, "Cancel")
        return {
            "available": False,
            "message": "SAP Show Tables did not expose an ANALYSIS RESULTS hinge table in the active session.",
            "table_tree_inventory": str(inventory_path),
            "exported_files": [],
        }

    selected_titles: list[str] = []
    for table in hinge_tables:
        leaf = _find_tree_item(tree, table["title"])
        if leaf is None:
            continue
        _ensure_analysis_table_checked(tree, leaf)
        selected_titles.append(table["title"].removeprefix("Table:  "))
    _click_named_button(dialog, "OK")
    viewer = _wait_for_table_viewer(Desktop, main_window.process_id(), wait_seconds)
    viewer = _ensure_viewer_tables(Desktop, main_window.process_id(), viewer, selected_titles, wait_seconds)
    exported_files = _export_selected_tables_to_csv(viewer, selected_titles, raw_dir, model_stem, wait_seconds)
    return {
        "available": bool(exported_files),
        "message": "Exact SAP hinge tables exported through Show Tables and Excel COM. SAP v22 table viewer was left open because closing its step-by-step hinge view can terminate SAP2000." if exported_files else "No selected SAP hinge table was exported.",
        "table_tree_inventory": str(inventory_path),
        "output_cases": list(output_cases),
        "nonlinear_result_mode": "Step-by-Step",
        "selected_tables": selected_titles,
        "exported_files": [str(path) for path in exported_files],
    }


def _find_active_sap_main_window(desktop: Any, wait_seconds: float, process_id: int | None = None) -> Any:
    """Return the visible SAP2000 main window for the active COM session."""
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        matches = [
            window for window in desktop(backend="win32").windows(process=process_id, visible_only=True)
            if window.window_text() == "SAP2000" or window.window_text().startswith("SAP2000 v")
        ]
        if matches:
            return matches[0]
        if process_id is not None and not _windows_process_exists(process_id):
            raise RuntimeError("SAP2000 process exited before its main window appeared.")
        time.sleep(0.5)
    raise RuntimeError("SAP2000 main window was not found for exact hinge export.")


def _wait_for_window(desktop: Any, process_id: int, title: str, wait_seconds: float) -> Any:
    """Wait for one visible SAP-owned window with an exact title."""
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        matches = [
            window for window in desktop(backend="win32").windows(process=process_id, visible_only=True)
            if window.window_text() == title
        ]
        if matches:
            return matches[0]
        if not _windows_process_exists(process_id):
            raise RuntimeError(f"SAP2000 process exited while waiting for window: {title}")
        time.sleep(0.25)
    raise RuntimeError(f"SAP2000 window did not appear: {title}")


def _open_show_tables_dialog(desktop: Any, keyboard: Any, main_window: Any, wait_seconds: float) -> Any:
    """Retry SAP's Ctrl+T accelerator until the Show Tables dialog is visible."""
    deadline = time.time() + wait_seconds
    process_id = main_window.process_id()
    while time.time() < deadline:
        matches = [
            window for window in desktop(backend="win32").windows(process=process_id, visible_only=True)
            if window.window_text() == "Choose Tables for Display"
        ]
        if matches:
            return matches[0]
        if not _windows_process_exists(process_id):
            raise RuntimeError("SAP2000 process exited while opening Show Tables.")
        _safe_focus(main_window)
        _focus_sap_workspace(main_window)
        keyboard.send_keys("{ESC}")
        keyboard.send_keys("^t")
        time.sleep(2.0)
    raise RuntimeError("SAP2000 window did not appear: Choose Tables for Display")


def _wait_for_table_viewer(desktop: Any, process_id: int, wait_seconds: float) -> Any:
    """Wait for SAP2000's database-table viewer form."""
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        for window in desktop(backend="win32").windows(process=process_id, visible_only=True):
            titles = {child.window_text() for child in window.children()}
            if "Add Tables..." in titles and "Done" in titles:
                combo_boxes = [child for child in window.children() if "COMBOBOX" in child.class_name()]
                if (
                    window.window_text() != "Please Wait While Table Display Initializes"
                    and combo_boxes
                    and combo_boxes[0].is_enabled()
                    and combo_boxes[0].item_texts()
                ):
                    return window
        if not _windows_process_exists(process_id):
            raise RuntimeError("SAP2000 process exited while its database-table viewer was loading.")
        time.sleep(0.25)
    raise RuntimeError("SAP2000 database-table viewer did not appear.")


def _ensure_viewer_tables(desktop: Any, process_id: int, viewer: Any, expected_titles: list[str], wait_seconds: float) -> Any:
    """Repair persisted Show Tables checkbox state when a target was toggled off."""
    combo = next(child for child in viewer.children() if "COMBOBOX" in child.class_name())
    item_texts = _wait_for_expected_combo_items(combo, expected_titles, min(wait_seconds, 15.0))
    missing_titles = [title for title in expected_titles if _match_table_viewer_item(title, item_texts) is None]
    if not missing_titles:
        return viewer

    _click_named_button(viewer, "Add Tables...")
    dialog = _wait_for_window(desktop, process_id, "Choose Tables for Display", wait_seconds)
    width = dialog.rectangle().width()
    height = dialog.rectangle().height()
    dialog.move_window(x=0, y=0, width=width, height=height, repaint=True)
    time.sleep(0.5)
    tree = next(child for child in dialog.children() if "SysTreeView32" in child.class_name())
    for title in missing_titles:
        leaf = _find_tree_item(tree, f"Table:  {title}")
        if leaf is None:
            raise RuntimeError(f"SAP Show Tables tree does not contain: {title}")
        _ensure_analysis_table_checked(tree, leaf)
    _click_named_button(dialog, "OK")
    viewer = _wait_for_table_viewer(desktop, process_id, wait_seconds)
    combo = next(child for child in viewer.children() if "COMBOBOX" in child.class_name())
    item_texts = _wait_for_expected_combo_items(combo, expected_titles, min(wait_seconds, 15.0))
    still_missing = [title for title in expected_titles if _match_table_viewer_item(title, item_texts) is None]
    if still_missing:
        raise RuntimeError(
            f"SAP table viewer is missing required tables: {', '.join(still_missing)}. "
            f"Visible tables: {item_texts}"
        )
    return viewer


def _wait_for_combo_items(combo: Any, wait_seconds: float) -> list[str]:
    """Wait for a SAP table-viewer combo box to finish initialization."""
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        item_texts = list(combo.item_texts())
        if item_texts:
            return item_texts
        time.sleep(0.25)
    return []


def _wait_for_expected_combo_items(combo: Any, expected_titles: list[str], wait_seconds: float) -> list[str]:
    """Wait briefly for SAP's asynchronously populated viewer table list."""
    deadline = time.time() + wait_seconds
    item_texts: list[str] = []
    while time.time() < deadline:
        item_texts = list(combo.item_texts())
        if all(_match_table_viewer_item(title, item_texts) is not None for title in expected_titles):
            return item_texts
        time.sleep(0.5)
    return item_texts


def _select_output_cases(desktop: Any, dialog: Any, process_id: int, case_names: tuple[str, ...], wait_seconds: float) -> None:
    """Leave exactly the requested pushover cases selected in SAP's custom list."""
    import win32api
    import win32con

    _click_named_button(dialog, "Select Load Cases...")
    case_dialog = _wait_for_window(desktop, process_id, "Select Output Cases", wait_seconds)
    list_boxes = [
        child for child in case_dialog.descendants()
        if "LISTBOX" in child.class_name() and child.is_visible()
    ]
    if not list_boxes:
        raise RuntimeError("SAP Select Output Cases list was not found.")
    list_box = list_boxes[0]
    item_texts = list(list_box.item_texts())
    missing = [case_name for case_name in case_names if case_name not in item_texts]
    if missing:
        raise RuntimeError(f"SAP output-case list does not contain: {', '.join(missing)}")

    # SAP keeps a tall empty area below its compact WinForms list rows.
    # Deriving the row height from the whole list rectangle clicks outside
    # the second item when only two pushover cases exist.
    row_height = 18
    selected = set(list_box.selected_indices())
    for index in sorted(selected):
        if item_texts[index] in case_names:
            continue
        win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
        try:
            list_box.click_input(coords=(30, row_height * index + row_height // 2))
        finally:
            win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
        selected = set(list_box.selected_indices())
    for case_name in case_names:
        index = item_texts.index(case_name)
        if index in selected:
            continue
        win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
        try:
            list_box.click_input(coords=(30, row_height * index + row_height // 2))
        finally:
            win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
        selected = set(list_box.selected_indices())
    expected = {item_texts.index(case_name) for case_name in case_names}
    if selected != expected:
        raise RuntimeError(f"SAP output cases could not be selected exactly: {case_names}. Selected indices: {sorted(selected)}")
    _click_named_button(case_dialog, "OK")


def _select_nonlinear_step_by_step(desktop: Any, dialog: Any, process_id: int, wait_seconds: float) -> None:
    """Use every saved nonlinear-static state instead of envelope-only rows."""
    _click_named_button(dialog, "Modify/Show Options...")
    options = _wait_for_window(desktop, process_id, "Output Options", wait_seconds)
    step_buttons = [
        child for child in options.descendants()
        if child.window_text() == "Step-by-Step" and child.is_visible() and child.is_enabled()
    ]
    if len(step_buttons) != 1:
        raise RuntimeError(f"Expected one enabled nonlinear Step-by-Step option, found {len(step_buttons)}.")
    step_buttons[0].click_input()
    _click_named_button(options, "OK")


def _table_tree_inventory(tree: Any) -> list[dict[str, Any]]:
    """Return flattened Show Tables tree rows with root ownership."""
    rows: list[dict[str, Any]] = []

    def walk(node: Any, root: str, path: list[str]) -> None:
        title = str(node.text())
        clean_path = [*path, title]
        rows.append({"title": title, "root": root, "path": clean_path, "is_table": title.startswith("Table:")})
        try:
            node.expand()
            children = node.children()
        except Exception:  # noqa: BLE001 - leaf nodes reject expansion.
            children = []
        for child in children:
            walk(child, root, clean_path)

    for root_node in tree.roots():
        walk(root_node, str(root_node.text()), [])
    return rows


def _find_tree_item(tree: Any, title: str) -> Any | None:
    """Return one tree item by exact title."""
    def walk(node: Any) -> Any | None:
        try:
            if str(node.text()) == title:
                return node
            node.expand()
            for child in node.children():
                result = walk(child)
                if result is not None:
                    return result
        except Exception:  # noqa: BLE001 - tolerate leaf and redraw timing.
            return None
        return None

    for root in tree.roots():
        result = walk(root)
        if result is not None:
            return result
    return None


def _ensure_analysis_table_checked(tree: Any, leaf: Any) -> None:
    """Leave one SAP analysis table checked by observing its selected-table counter."""
    from pywinauto import mouse

    leaf.ensure_visible()
    selected_before = _analysis_result_selected_count(tree)
    tree_rect = tree.rectangle()
    leaf_rect = leaf.client_rect()
    if leaf_rect is None:
        raise RuntimeError(f"SAP table row is not visible: {leaf.text()}")
    coords = (
        tree_rect.left + leaf_rect.left + 11,
        tree_rect.top + (leaf_rect.top + leaf_rect.bottom) // 2 + TREE_VERTICAL_OFFSET_PX,
    )
    mouse.click(coords=coords)
    time.sleep(0.5)
    selected_after = _analysis_result_selected_count(tree)
    if selected_after > selected_before:
        return
    if selected_after < selected_before:
        mouse.click(coords=coords)
        time.sleep(0.5)
        selected_restored = _analysis_result_selected_count(tree)
        if selected_restored == selected_before:
            return
    raise RuntimeError(f"SAP required table checkbox could not be selected: {leaf.text()}")


def _analysis_result_selected_count(tree: Any) -> int:
    """Return SAP's ANALYSIS RESULTS selected-table counter."""
    for root in tree.roots():
        match = re.search(r"ANALYSIS RESULTS\s+\((\d+)\s+of\s+\d+", str(root.text()), re.IGNORECASE)
        if match:
            return int(match.group(1))
    raise RuntimeError("SAP ANALYSIS RESULTS selected-table counter was not found.")


def _safe_focus(window: Any) -> None:
    """Focus a desktop window even when Windows rejects SetForegroundWindow."""
    try:
        window.set_focus()
        return
    except Exception:  # noqa: BLE001 - Windows foreground restrictions are intermittent.
        pass
    rect = window.rectangle()
    try:
        _force_foreground_window(int(window.handle))
        from pywinauto import mouse

        mouse.click(coords=(rect.left + min(80, max(10, rect.width() // 3)), rect.top + 12))
    except Exception as exc:  # noqa: BLE001 - report a meaningful desktop automation error.
        raise RuntimeError(f"Could not focus SAP2000 window: {window.window_text()}") from exc
    time.sleep(0.4)


def _force_foreground_window(handle: int) -> None:
    """Bring one SAP-owned native window forward across worker thread boundaries."""
    user32 = ctypes.windll.user32
    foreground_handle = int(user32.GetForegroundWindow())
    current_thread = int(ctypes.windll.kernel32.GetCurrentThreadId())
    target_thread = int(user32.GetWindowThreadProcessId(handle, None))
    foreground_thread = int(user32.GetWindowThreadProcessId(foreground_handle, None)) if foreground_handle else 0
    attached_foreground = foreground_thread and foreground_thread != current_thread
    attached_target = target_thread and target_thread != current_thread and target_thread != foreground_thread
    try:
        if attached_foreground:
            user32.AttachThreadInput(current_thread, foreground_thread, True)
        if attached_target:
            user32.AttachThreadInput(current_thread, target_thread, True)
        user32.ShowWindow(handle, 9)
        user32.BringWindowToTop(handle)
        user32.SetForegroundWindow(handle)
        user32.SetFocus(handle)
    finally:
        if attached_target:
            user32.AttachThreadInput(current_thread, target_thread, False)
        if attached_foreground:
            user32.AttachThreadInput(current_thread, foreground_thread, False)


def _focus_sap_workspace(window: Any) -> None:
    """Click SAP's central workspace so the Show Tables accelerator reaches SAP."""
    from pywinauto import mouse

    rect = window.rectangle()
    mouse.click(coords=(rect.left + max(160, rect.width() // 2), rect.top + max(180, rect.height() // 3)))
    time.sleep(0.5)


def _export_selected_tables_to_csv(viewer: Any, selected_titles: list[str], raw_dir: Path, model_stem: str, wait_seconds: float) -> list[Path]:
    """Export selected SAP table-viewer entries to separate UTF-8 CSV files."""
    from pywinauto import mouse

    viewer_rect = viewer.rectangle()
    if viewer_rect.left < 0 or viewer_rect.top < 0:
        viewer.move_window(x=100, y=100, width=viewer_rect.width(), height=viewer_rect.height(), repaint=True)
        time.sleep(0.5)
    combo = next(child for child in viewer.children() if "COMBOBOX" in child.class_name())
    item_texts = _wait_for_combo_items(combo, wait_seconds)
    exported: list[Path] = []
    for title in selected_titles:
        selected_item = _match_table_viewer_item(title, item_texts)
        if selected_item is None:
            continue
        combo.select(selected_item)
        _wait_for_viewer_table(viewer, selected_item, wait_seconds)
        viewer_rect = viewer.rectangle()
        # SAP's WinForms menu accelerator is unreliable when the table viewer
        # restores itself off-screen. Click the stable File > Export Current
        # Table > To Excel menu path after bringing the viewer to the primary
        # display.
        mouse.click(coords=(viewer_rect.left + 42, viewer_rect.top + 52))
        time.sleep(0.2)
        mouse.move(coords=(viewer_rect.left + 180, viewer_rect.top + 80))
        time.sleep(0.2)
        mouse.click(coords=(viewer_rect.left + 500, viewer_rect.top + 80))
        workbook, excel = _wait_for_excel_workbook(wait_seconds, title)
        safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_").lower()
        target = raw_dir / f"{model_stem}__{safe_title}.csv"
        excel.DisplayAlerts = False
        workbook.SaveAs(str(target), FileFormat=62)
        workbook.Close(SaveChanges=False)
        excel.Quit()
        exported.append(target)
        time.sleep(0.5)
    return exported


def _match_table_viewer_item(title: str, item_texts: list[str]) -> str | None:
    """Match SAP tree and viewer captions despite abbreviated table titles."""
    normalized_title = re.sub(r"[^a-z0-9]+", "", title.lower())
    for item in item_texts:
        normalized_item = re.sub(r"[^a-z0-9]+", "", str(item).lower())
        if normalized_item == normalized_title or normalized_item in normalized_title or normalized_title in normalized_item:
            return item
    return None


def _wait_for_viewer_table(viewer: Any, expected_title: str, wait_seconds: float) -> None:
    """Wait until SAP's table viewer has switched to the requested table."""
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if viewer.window_text().strip() == expected_title.strip():
            time.sleep(0.5)
            return
        time.sleep(0.25)
    raise RuntimeError(f"SAP table viewer did not switch to: {expected_title}")


def _wait_for_excel_workbook(wait_seconds: float, expected_title: str) -> tuple[Any, Any]:
    """Wait for SAP2000's populated Excel export and return its workbook."""
    deadline = time.time() + wait_seconds
    last_error: Exception | None = None
    last_first_cell = ""
    while time.time() < deadline:
        try:
            try:
                import win32com.client
                excel = win32com.client.GetActiveObject("Excel.Application")
            except ImportError:
                import comtypes.client
                excel = comtypes.client.GetActiveObject("Excel.Application")
            workbook = excel.ActiveWorkbook
            first_cell = str(workbook.ActiveSheet.Cells(1, 1).Value or "") if workbook is not None else ""
            last_first_cell = first_cell
            if workbook is not None and workbook.ActiveSheet.UsedRange.Rows.Count > 1 and expected_title.lower() in first_cell.lower():
                return workbook, excel
        except Exception as exc:  # noqa: BLE001 - Excel startup is asynchronous.
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(
        f"Expected Excel table '{expected_title}' did not appear after SAP export. "
        f"Last first cell: {last_first_cell!r}. Last error: {last_error}"
    )


def _click_named_button(window: Any, title: str) -> None:
    """Click a visible button by exact caption."""
    buttons = [child for child in window.children() if child.window_text() == title and child.is_visible()]
    if not buttons:
        raise RuntimeError(f"Button not found in {window.window_text()}: {title}")
    buttons[0].click_input()
    time.sleep(0.5)


def _windows_process_ids(image_name: str) -> set[int]:
    """Return Windows process IDs for one executable image name."""
    completed = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {image_name}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    process_ids: set[int] = set()
    for row in csv.reader(completed.stdout.splitlines()):
        if len(row) < 2 or row[0].lower() != image_name.lower():
            continue
        try:
            process_ids.add(int(row[1]))
        except ValueError:
            continue
    return process_ids


def _windows_process_exists(process_id: int) -> bool:
    """Return whether one Windows process ID is still running."""
    completed = subprocess.run(
        ["tasklist", "/FI", f"PID eq {process_id}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    for row in csv.reader(completed.stdout.splitlines()):
        if len(row) < 2:
            continue
        try:
            if int(row[1]) == process_id:
                return True
        except ValueError:
            continue
    return False


def _control_tree(control: Any, depth: int = 0, max_depth: int = 4) -> dict[str, Any]:
    """Return a compact pywinauto control tree."""
    data = {
        "title": control.window_text(),
        "class_name": control.class_name(),
        "control_id": control.control_id(),
        "handle": int(control.handle),
        "children": [],
    }
    if depth >= max_depth:
        return data
    try:
        children = control.children()
    except Exception:  # noqa: BLE001 - some native controls reject enumeration.
        children = []
    data["children"] = [_control_tree(child, depth + 1, max_depth) for child in children]
    return data


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path, help="Analyzed SAP2000 .sdb model used for the pilot.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"D:\sap2000_generated_models\hinge_validation\ui_probe.json"),
        help="Probe JSON output path.",
    )
    parser.add_argument("--wait-seconds", type=float, default=120.0)
    return parser


def main() -> None:
    """Run pilot UI probe."""
    args = build_parser().parse_args()
    result = open_and_probe(args.model_path, args.output, args.wait_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
