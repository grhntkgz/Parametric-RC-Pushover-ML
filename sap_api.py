"""SAP2000 OAPI / COM adapter.

The wrapper keeps SAP2000 calls centralized and checks return codes after every
API call. SAP2000 OAPI method signatures can vary slightly by product version;
where practical, this module uses small compatibility fallbacks.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from config import Section
from design_rules import CONCRETE_FCK_MPA, concrete_elastic_modulus_mpa


class SapApiError(RuntimeError):
    """Raised when SAP2000 cannot start or an OAPI call fails."""


@dataclass(frozen=True)
class Point3D:
    """Cartesian point in SAP2000 model coordinates."""

    x: float
    y: float
    z: float


@dataclass
class HingeAssignmentSummary:
    """Summary of plastic hinge assignment attempts."""

    attempted: bool = False
    assigned_count: int = 0
    failed_count: int = 0
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Initialize warning storage."""
        if self.warnings is None:
            self.warnings = []


class Sap2000Api:
    """Thin checked wrapper around SAP2000 COM OAPI."""

    def __init__(self, visible: bool = True) -> None:
        """Create a wrapper instance; call start() before modeling."""
        self.visible = visible
        self.sap_object = None
        self.sap_model = None
        self._point_names: dict[tuple[float, float, float], str] = {}

    def start(self) -> None:
        """Start SAP2000 through COM using helper first, then direct ProgID fallback."""
        try:
            import comtypes.client
        except ImportError as exc:
            raise SapApiError("comtypes is required. Install it with: pip install comtypes") from exc

        last_error: Exception | None = None
        for prog_id in ("SAP2000v1.Helper", "CSI.SAP2000.API.Helper"):
            try:
                helper = comtypes.client.CreateObject(prog_id)
                helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)
                self.sap_object = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")
                self._start_sap_application()
                self.sap_model = self.sap_object.SapModel
                return
            except Exception as exc:  # noqa: BLE001 - COM startup needs broad fallback.
                last_error = exc

        try:
            self.sap_object = comtypes.client.CreateObject("CSI.SAP2000.API.SapObject")
            self._start_sap_application()
            self.sap_model = self.sap_object.SapModel
        except Exception as exc:  # noqa: BLE001 - COM reports many startup failures as generic exceptions.
            detail = last_error or exc
            raise SapApiError(f"SAP2000 could not be started through COM: {detail}") from exc

    def initialize_new_model(self) -> None:
        """Initialize and clear a new model in kN-m-C units."""
        self._require_model()
        self._check(self.sap_model.InitializeNewModel(6), "InitializeNewModel(kN_m_C)")
        self._check(self.sap_model.File.NewBlank(), "File.NewBlank")
        self._point_names.clear()
        self.set_units_kn_m_c()

    def _start_sap_application(self) -> None:
        """Start SAP2000 with version-tolerant visible-window signatures."""
        candidates = (
            lambda: self.sap_object.ApplicationStart("", self.visible, ""),
            lambda: self.sap_object.ApplicationStart(self.visible),
            lambda: self.sap_object.ApplicationStart(),
        )
        self._checked_first_success(candidates, "ApplicationStart")

    def set_units_kn_m_c(self) -> None:
        """Set present units to kN, m, C."""
        self._require_model()
        self._check(self.sap_model.SetPresentUnits(6), "SetPresentUnits(kN_m_C)")

    def define_concrete_material(self, name: str) -> None:
        """Define isotropic concrete material with approximate TS-class properties."""
        self._require_model()
        fck = CONCRETE_FCK_MPA[name]
        elastic_modulus_kn_m2 = concrete_elastic_modulus_mpa(name) * 1000.0
        self._check(self.sap_model.PropMaterial.SetMaterial(name, 2), f"PropMaterial.SetMaterial({name})")
        self._check(self.sap_model.PropMaterial.SetMPIsotropic(name, elastic_modulus_kn_m2, 0.20, 1.0e-5), "PropMaterial.SetMPIsotropic")
        try:
            self._check(self.sap_model.PropMaterial.SetOConcrete(name, fck * 1000.0, False, 0, 2, 4, 0.0022, 0.0052, -0.1), "PropMaterial.SetOConcrete")
        except SapApiError:
            # Some SAP2000 versions expose shorter material-overwrite signatures.
            pass

    def define_rebar_material(self, name: str, fy_mpa: float = 420.0) -> None:
        """Define reinforcing steel material for future design metadata."""
        self._require_model()
        self._check(self.sap_model.PropMaterial.SetMaterial(name, 6), f"PropMaterial.SetMaterial({name})")
        self._check(self.sap_model.PropMaterial.SetMPIsotropic(name, 200_000_000.0, 0.30, 1.2e-5), "PropMaterial.SetMPIsotropic")
        try:
            self._check(self.sap_model.PropMaterial.SetORebar(name, fy_mpa * 1000.0, 1.15 * fy_mpa * 1000.0, fy_mpa * 1000.0, 1.15 * fy_mpa * 1000.0, 1, 1, 0.01, 0.09, False), "PropMaterial.SetORebar")
        except SapApiError:
            pass

    def define_rectangular_frame_section(self, name: str, material: str, section: Section) -> None:
        """Define a rectangular frame section."""
        self._require_model()
        self._check(self.sap_model.PropFrame.SetRectangle(name, material, section.depth, section.width), f"PropFrame.SetRectangle({name})")

    def define_raft_shell_section(self, name: str, material: str, thickness_m: float) -> None:
        """Define a concrete shell area section for preliminary raft modeling."""
        self._require_model()
        candidates = (
            lambda: self.sap_model.PropArea.SetShell(name, 1, material, 0.0, thickness_m, thickness_m),
            lambda: self.sap_model.PropArea.SetShell(name, 1, material, thickness_m),
            lambda: self.sap_model.PropArea.SetShell(name, material, thickness_m),
        )
        self._checked_first_success(candidates, f"PropArea.SetShell({name})")

    def define_wall_shell_section(self, name: str, material: str, thickness_m: float) -> None:
        """Define a concrete shell area section for vertical shear wall modeling."""
        self.define_raft_shell_section(name, material, thickness_m)

    def define_slab_shell_section(self, name: str, material: str, thickness_m: float) -> None:
        """Define a concrete shell area section for beam-supported floor slabs."""
        self.define_raft_shell_section(name, material, thickness_m)

    def add_raft_area(self, corner_points: Sequence[Point3D], section_name: str, user_name: str = "RAFT") -> str:
        """Create one raft shell area at foundation level and return its SAP object name."""
        self._require_model()
        if len(corner_points) < 3:
            raise SapApiError("Raft area requires at least three corner points.")
        x_values = [point.x for point in corner_points]
        y_values = [point.y for point in corner_points]
        z_values = [point.z for point in corner_points]
        candidates = (
            lambda: self.sap_model.AreaObj.AddByCoord(len(corner_points), x_values, y_values, z_values, user_name, section_name, user_name, "Global"),
            lambda: self.sap_model.AreaObj.AddByCoord(len(corner_points), x_values, y_values, z_values, user_name, section_name),
            lambda: self.sap_model.AreaObj.AddByCoord(len(corner_points), x_values, y_values, z_values),
        )
        name = user_name
        errors: list[str] = []
        for callback in candidates:
            try:
                result = callback()
                ret, obj_name = self._parse_ret_name(result, name)
                self._check(ret, f"AreaObj.AddByCoord({user_name})")
                name = obj_name or name
                self._checked_optional_call(lambda: self.sap_model.AreaObj.SetProperty(name, section_name), f"AreaObj.SetProperty({name})")
                return name
            except Exception as exc:  # noqa: BLE001 - SAP2000 OAPI version compatibility.
                errors.append(str(exc))
        raise SapApiError(f"AreaObj.AddByCoord({user_name}) failed for all known signatures: {' | '.join(errors)}")

    def add_wall_area(self, corner_points: Sequence[Point3D], section_name: str, user_name: str) -> str:
        """Create one vertical shear wall shell area and return its SAP object name."""
        return self.add_raft_area(corner_points, section_name, user_name)

    def add_slab_area(self, corner_points: Sequence[Point3D], section_name: str, user_name: str) -> str:
        """Create one horizontal floor-slab shell panel and return its SAP object name."""
        return self.add_raft_area(corner_points, section_name, user_name)

    def add_frame(self, point_i: Point3D, point_j: Point3D, section_name: str, user_name: str = "") -> str:
        """Create a frame object and return the SAP2000 object name."""
        self._require_model()
        coord_errors: list[str] = []
        obj_name = self._add_frame_by_coord(point_i, point_j, section_name, user_name, coord_errors)
        if obj_name:
            self._cache_frame_end_points(obj_name, point_i, point_j)
            return obj_name

        point_i_name = self._get_point_name(point_i)
        point_j_name = self._get_point_name(point_j)
        result = self.sap_model.FrameObj.AddByPoint(point_i_name, point_j_name, user_name, section_name, user_name)
        ret, obj_name = self._parse_ret_name(result, user_name)
        try:
            self._check(ret, f"FrameObj.AddByPoint({user_name})")
        except SapApiError as exc:
            joined_errors = " | ".join(coord_errors)
            raise SapApiError(f"{exc}; FrameObj.AddByCoord fallback errors: {joined_errors}") from exc
        self._cache_frame_end_points(obj_name, point_i, point_j)
        return obj_name

    def set_base_restraints(self, base_points: Iterable[Point3D]) -> None:
        """Assign fixed supports to base joint coordinates."""
        for point in base_points:
            point_name = self._get_point_name(point)
            self._check(self.sap_model.PointObj.SetRestraint(point_name, [True, True, True, True, True, True]), "PointObj.SetRestraint")

    def set_base_restraints_for_soil_springs(self, base_points: Iterable[Point3D]) -> None:
        """Assign base restraints compatible with vertical Winkler springs."""
        for point in base_points:
            point_name = self._get_point_name(point)
            self._check(self.sap_model.PointObj.SetRestraint(point_name, [True, True, False, True, True, True]), "PointObj.SetRestraint(soil spring base)")

    def assign_vertical_soil_springs(self, point_stiffness: dict[Point3D, float]) -> None:
        """Assign vertical point springs to base joints using tributary-area stiffness."""
        for point, stiffness_kn_m in point_stiffness.items():
            point_name = self._get_point_name(point)
            springs = [0.0, 0.0, stiffness_kn_m, 0.0, 0.0, 0.0]
            candidates = (
                lambda: self.sap_model.PointObj.SetSpring(point_name, springs, 1),
                lambda: self.sap_model.PointObj.SetSpring(point_name, springs),
            )
            self._checked_first_success(candidates, f"PointObj.SetSpring({point_name})")

    def define_rigid_diaphragms(self, story_count: int) -> list[str]:
        """Define one rigid diaphragm constraint per story."""
        names: list[str] = []
        for story in range(1, story_count + 1):
            name = f"D{story:02d}"
            self._check(self.sap_model.ConstraintDef.SetDiaphragm(name, 3), f"ConstraintDef.SetDiaphragm({name})")
            names.append(name)
        return names

    def assign_story_diaphragms(self, floor_points: dict[int, Sequence[Point3D]], diaphragm_names: Sequence[str]) -> None:
        """Assign floor points to story diaphragm constraints."""
        for story, points in floor_points.items():
            diaphragm = diaphragm_names[story - 1]
            for point in points:
                point_name = self._get_point_name(point)
                self._check(self.sap_model.PointObj.SetConstraint(point_name, diaphragm), "PointObj.SetConstraint")

    def define_load_patterns(self) -> None:
        """Define dead and live load patterns."""
        self._ensure_load_pattern("DEAD", 1, 1.0)
        self._ensure_load_pattern("LIVE", 3, 0.0)

    def define_pushover_load_patterns(self, directions: Sequence[str]) -> None:
        """Define lateral load patterns used by pushover cases."""
        for direction in directions:
            pattern = f"PUSH_{direction.upper()}"
            self._ensure_load_pattern(pattern, 8, 0.0)

    def assign_uniform_frame_loads(self, frame_names: Iterable[str], dead_kn_m: float, live_kn_m: float) -> None:
        """Assign gravity loads to beam frame objects in global gravity direction."""
        for frame in frame_names:
            self._check(self.sap_model.FrameObj.SetLoadDistributed(frame, "DEAD", 1, 10, 0.0, 1.0, -dead_kn_m, -dead_kn_m, "Global"), "FrameObj.SetLoadDistributed(DEAD)")
            self._check(self.sap_model.FrameObj.SetLoadDistributed(frame, "LIVE", 1, 10, 0.0, 1.0, -live_kn_m, -live_kn_m, "Global"), "FrameObj.SetLoadDistributed(LIVE)")

    def assign_pushover_story_loads(
        self,
        floor_points: dict[int, Sequence[Point3D]],
        story_height: float,
        base_shear_proxy_kn: float,
        directions: Sequence[str],
    ) -> None:
        """Assign normalized triangular lateral pushover loads to story joints."""
        story_weights = {story: story * story_height for story in floor_points}
        weight_sum = sum(story_weights.values())
        if weight_sum <= 0.0:
            raise SapApiError("Pushover story load distribution has zero total weight.")

        for direction in directions:
            pattern = f"PUSH_{direction.upper()}"
            for story, points in floor_points.items():
                story_force = base_shear_proxy_kn * story_weights[story] / weight_sum
                joint_force = story_force / max(len(points), 1)
                for point in points:
                    point_name = self._get_point_name(point)
                    loads = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    if direction.upper() == "X":
                        loads[0] = joint_force
                    elif direction.upper() == "Y":
                        loads[1] = joint_force
                    else:
                        raise SapApiError(f"Unsupported pushover direction: {direction}")
                    self._check(self.sap_model.PointObj.SetLoadForce(point_name, pattern, loads, True, "Global"), f"PointObj.SetLoadForce({pattern})")

    def define_modal_case(self) -> None:
        """Define a modal load case if available in the current SAP2000 version."""
        try:
            self._check(self.sap_model.LoadCases.ModalEigen.SetCase("MODAL"), "LoadCases.ModalEigen.SetCase")
            self._check(self.sap_model.LoadCases.ModalEigen.SetNumberModes("MODAL", 12, 1), "LoadCases.ModalEigen.SetNumberModes")
        except SapApiError:
            print("Warning: Modal case definition failed; continuing with geometry-only model.")

    def define_pushover_case(self, name: str, load_pattern: str, control_point: Point3D, direction: str, target_displacement_m: float) -> str:
        """Define a displacement-controlled nonlinear static pushover case.

        SAP2000 OAPI signatures for nonlinear static controls differ across
        versions. The required case and load pattern are created with checked
        calls; optional control settings are tried through compatible variants.
        """
        point_name = self._get_point_name(control_point)
        self._check(self.sap_model.LoadCases.StaticNonlinear.SetCase(name), f"LoadCases.StaticNonlinear.SetCase({name})")
        self._set_static_nonlinear_initial_case(name)
        self._checked_optional_call(lambda: self.sap_model.LoadCases.StaticNonlinear.SetGeometricNonlinearity(name, 2), f"StaticNonlinear.SetGeometricNonlinearity({name})")
        self._set_static_nonlinear_loads(name, load_pattern)
        self._set_pushover_displacement_control(name, point_name, direction, target_displacement_m)
        self._check(self.sap_model.LoadCases.StaticNonlinear.SetResultsSaved(name, True, 20, 200, True), f"StaticNonlinear.SetResultsSaved({name})")
        saved_settings = self.get_static_nonlinear_results_saved(name)
        if not saved_settings["save_multiple_steps"]:
            raise SapApiError(f"StaticNonlinear.SetResultsSaved({name}) did not enable multiple saved states.")
        self._checked_optional_call(lambda: self.sap_model.LoadCases.StaticNonlinear.SetSolControlParameters(name, 200, 20, 10, 40, 1.0e-4, True, 1.0e-4, 5, 1.0e-4, 1.0), f"StaticNonlinear.SetSolControlParameters({name})")
        return point_name

    def get_static_nonlinear_results_saved(self, case_name: str) -> dict[str, object]:
        """Return verified nonlinear-static saved-state settings."""
        self._require_model()
        result = self.sap_model.LoadCases.StaticNonlinear.GetResultsSaved(case_name)
        if not isinstance(result, (list, tuple)) or len(result) < 5:
            raise SapApiError(f"StaticNonlinear.GetResultsSaved({case_name}) returned an unexpected payload: {result!r}")
        save_multiple_steps, min_saved_states, max_saved_states, positive_only, ret = result[:5]
        self._check(ret, f"StaticNonlinear.GetResultsSaved({case_name})")
        return {
            "case": case_name,
            "save_multiple_steps": bool(save_multiple_steps),
            "min_saved_states": int(min_saved_states),
            "max_saved_states": int(max_saved_states),
            "positive_only": bool(positive_only),
        }

    def assign_plastic_hinges_to_frames(
        self,
        frame_names: Sequence[str],
        hinge_property: str,
        relative_distances: Sequence[float] = (0.0, 1.0),
        required: bool = False,
    ) -> HingeAssignmentSummary:
        """Assign plastic hinges at frame ends using version-tolerant OAPI calls.

        The hinge property names are intentionally configurable. In SAP2000 they
        may point to built-in default hinge properties or to user-defined hinge
        properties created in a later detailing phase.
        """
        summary = HingeAssignmentSummary(attempted=bool(frame_names))
        distances = [float(value) for value in relative_distances]
        hinge_count = len(distances)
        hinge_numbers = list(range(1, hinge_count + 1))
        hinge_properties = [hinge_property] * hinge_count
        hinge_types = [1] * hinge_count
        behavior_types = [1] * hinge_count
        sources = [0] * hinge_count

        for frame_name in frame_names:
            callbacks = (
                lambda frame=frame_name: self.sap_model.FrameObj.SetHingeAssign(frame, hinge_count, hinge_numbers, hinge_properties, hinge_types, behavior_types, sources, distances),
                lambda frame=frame_name: self.sap_model.FrameObj.SetHingeAssign(frame, hinge_count, hinge_properties, distances),
                lambda frame=frame_name: self.sap_model.FrameObj.SetHingeAssign(frame, hinge_property, distances),
                lambda frame=frame_name: self.sap_model.FrameObj.SetHingeAssign(frame, hinge_property, 0.0),
                lambda frame=frame_name: self.sap_model.FrameObj.SetHingeAssign(frame, hinge_property, 1.0),
            )
            if self._try_first_success(callbacks, f"FrameObj.SetHingeAssign({frame_name}, {hinge_property})"):
                summary.assigned_count += 1
            else:
                summary.failed_count += 1
                summary.warnings.append(f"Could not assign {hinge_property} to {frame_name}.")

        if summary.failed_count and required:
            raise SapApiError("; ".join(summary.warnings[:10]))
        if summary.failed_count:
            print(f"Warning: {summary.failed_count} plastic hinge assignment(s) failed for {hinge_property}.")
        return summary

    def read_frame_hinge_state_summary(self, case_name: str) -> dict[str, object]:
        """Best-effort read of frame hinge states after nonlinear analysis.

        SAP2000 result method names and return payloads vary by version. This
        returns a compact status object instead of pretending a result exists
        when the installed OAPI does not expose it through one of these shapes.
        """
        self._checked_optional_call(lambda: self.sap_model.Results.Setup.DeselectAllCasesAndCombosForOutput(), "Results.Setup.DeselectAllCasesAndCombosForOutput")
        self._checked_optional_call(lambda: self.sap_model.Results.Setup.SetCaseSelectedForOutput(case_name), f"Results.Setup.SetCaseSelectedForOutput({case_name})")
        callbacks = (
            lambda: self.sap_model.Results.FrameHinge(),
            lambda: self.sap_model.Results.FrameHinge("", 0),
            lambda: self.sap_model.Results.FrameHingeState(),
            lambda: self.sap_model.Results.FrameHingeState("", 0),
        )
        for callback in callbacks:
            try:
                result = callback()
                return self._summarize_hinge_result(result, case_name)
            except Exception:
                continue
        return {"case": case_name, "available": False, "message": "Frame hinge result method is not available through known OAPI signatures."}

    def read_frame_force_history(self, case_name: str, frame_names: Sequence[str]) -> dict[str, object]:
        """Read frame force histories for selected frames in one output case.

        SAP2000 v22 exposes frame force result tables through OAPI even when
        explicit hinge result tables are not exposed. The returned records are
        used as a best-effort demand proxy for generated hinge states.
        """
        self._require_model()
        result_setup = self._select_case_for_step_by_step_results(case_name)
        records: list[dict[str, object]] = []
        warnings: list[str] = []
        for frame_name in frame_names:
            try:
                result = self.sap_model.Results.FrameForce(frame_name, 0)
            except Exception as exc:  # noqa: BLE001 - keep the rest of the model readable.
                warnings.append(f"{frame_name}: {exc}")
                continue
            arrays = self._result_arrays(result)
            if len(arrays) < 13:
                warnings.append(f"{frame_name}: unexpected FrameForce result shape ({len(arrays)} arrays)")
                continue
            obj = [str(value) for value in arrays[0]]
            obj_sta = self._numeric_result_array(arrays[1])
            elm = [str(value) for value in arrays[2]]
            elm_sta = self._numeric_result_array(arrays[3])
            load_cases = [str(value) for value in arrays[4]]
            step_types = [str(value) for value in arrays[5]]
            step_nums = self._numeric_result_array(arrays[6])
            p_values = self._numeric_result_array(arrays[7])
            v2_values = self._numeric_result_array(arrays[8])
            v3_values = self._numeric_result_array(arrays[9])
            t_values = self._numeric_result_array(arrays[10])
            m2_values = self._numeric_result_array(arrays[11])
            m3_values = self._numeric_result_array(arrays[12])
            count = min(
                len(obj),
                len(obj_sta),
                len(elm),
                len(elm_sta),
                len(load_cases),
                len(step_types),
                len(step_nums),
                len(p_values),
                len(v2_values),
                len(v3_values),
                len(t_values),
                len(m2_values),
                len(m3_values),
            )
            for index in range(count):
                records.append(
                    {
                        "frame": obj[index],
                        "object_station": obj_sta[index],
                        "element": elm[index],
                        "element_station": elm_sta[index],
                        "case": load_cases[index],
                        "load_step": step_types[index],
                        "step_number": step_nums[index],
                        "P": p_values[index],
                        "V2": v2_values[index],
                        "V3": v3_values[index],
                        "T": t_values[index],
                        "M2": m2_values[index],
                        "M3": m3_values[index],
                    }
                )
        return {
            "case": case_name,
            "available": bool(records),
            "record_count": len(records),
            "records": records,
            "result_setup": result_setup,
            "warnings": warnings[:20],
        }

    def read_pushover_curve(self, case_name: str, control_point: Point3D, direction: str, control_point_name: str | None = None) -> dict[str, object]:
        """Read a compact pushover capacity curve from base reactions and roof drift.

        The returned curve pairs the displacement of the pushover control joint
        with the global base shear component in the same direction. SAP2000 OAPI
        result payloads are array-heavy and vary slightly by version, so failures
        are reported in the returned object instead of masking them.
        """
        self._require_model()
        point_name = control_point_name or self._get_point_name(control_point)
        direction_upper = direction.upper()
        disp_index = 5 if direction_upper == "X" else 6
        shear_index = 3 if direction_upper == "X" else 4
        result_setup = self._select_case_for_step_by_step_results(case_name)

        try:
            base = self.sap_model.Results.BaseReact()
        except Exception as exc:  # noqa: BLE001 - keep a win32-style fallback for alternate bindings.
            try:
                base = self.sap_model.Results.BaseReact(0, [], [], [], [], [], [], [], [], [], 0.0, 0.0, 0.0)
            except Exception as fallback_exc:  # noqa: BLE001 - report the method that failed.
                return {
                    "case": case_name,
                    "direction": direction_upper,
                    "available": False,
                    "message": f"Results.BaseReact failed: {exc}; fallback failed: {fallback_exc}",
                    "points": [],
                }

        try:
            joint = self.sap_model.Results.JointDispl(point_name, 0)
        except Exception as exc:  # noqa: BLE001 - SAP2000 result APIs differ by version/license.
            try:
                joint = self.sap_model.Results.JointDispl(point_name, 0, 0, [], [], [], [], [], [], [], [], [], [], [])
            except Exception as fallback_exc:  # noqa: BLE001 - report the method that failed.
                return {
                    "case": case_name,
                    "direction": direction_upper,
                    "available": False,
                    "message": f"Results.JointDispl failed for {point_name}: {exc}; fallback failed: {fallback_exc}",
                    "points": [],
                }

        base_arrays = self._result_arrays(base)
        joint_arrays = self._result_arrays(joint)
        try:
            base_steps = self._numeric_result_array(base_arrays[2])
            base_step_types = self._string_result_array(base_arrays[1])
            shears = [abs(value) for value in self._numeric_result_array(base_arrays[shear_index])]
            joint_steps = self._numeric_result_array(joint_arrays[4])
            joint_step_types = self._string_result_array(joint_arrays[3])
            displacements = self._numeric_result_array(joint_arrays[disp_index])
        except Exception as exc:  # noqa: BLE001 - keep metadata generation alive.
            return {
                "case": case_name,
                "direction": direction_upper,
                "available": False,
                "message": f"Could not parse pushover result arrays: {exc}",
                "base_array_count": len(base_arrays),
                "joint_array_count": len(joint_arrays),
                "points": [],
            }

        if not base_steps or not shears or not joint_steps or not displacements:
            return {
                "case": case_name,
                "direction": direction_upper,
                "available": False,
                "message": "Pushover result arrays were empty after numeric parsing.",
                "base_array_count": len(base_arrays),
                "joint_array_count": len(joint_arrays),
                "base_step_count": len(base_steps),
                "joint_step_count": len(joint_steps),
                "points": [],
            }

        points = self._build_capacity_points(base_steps, base_step_types, shears, joint_steps, joint_step_types, displacements)

        peak = max((point["base_shear_kn"] for point in points), default=None)
        final = points[-1] if points else None
        return {
            "case": case_name,
            "direction": direction_upper,
            "available": bool(points),
            "control_point": point_name,
            "point_count": len(points),
            "peak_base_shear_kn": peak,
            "final_control_displacement_m": final["control_displacement_m"] if final else None,
            "final_base_shear_kn": final["base_shear_kn"] if final else None,
            "curve_note": "Envelope Max/Min result rows are converted to a monotonic capacity estimate when SAP2000 does not return saved nonlinear states.",
            "result_setup": result_setup,
            "points": points,
        }

    def read_story_drift_history(
        self,
        case_name: str,
        story_points: dict[int, Point3D],
        story_height_m: float,
        direction: str,
        point_names: dict[int, str] | None = None,
    ) -> dict[str, object]:
        """Read story displacement histories and compute interstory drift ratios."""
        self._require_model()
        direction_upper = direction.upper()
        disp_index = 5 if direction_upper == "X" else 6
        result_setup = self._select_case_for_step_by_step_results(case_name)
        by_story: dict[int, dict[tuple[float, str], float]] = {0: {(0.0, "Base"): 0.0}}
        warnings: list[str] = []

        for story, point in sorted(story_points.items()):
            point_name = point_names.get(story) if point_names else None
            point_name = point_name or self._get_point_name(point)
            try:
                joint = self.sap_model.Results.JointDispl(point_name, 0)
            except Exception as exc:  # noqa: BLE001 - keep model generation alive.
                try:
                    joint = self.sap_model.Results.JointDispl(point_name, 0, 0, [], [], [], [], [], [], [], [], [], [], [])
                except Exception as fallback_exc:  # noqa: BLE001
                    warnings.append(f"Story {story} point {point_name}: {exc}; fallback: {fallback_exc}")
                    continue
            arrays = self._result_arrays(joint)
            try:
                step_types = self._string_result_array(arrays[3])
                step_numbers = self._numeric_result_array(arrays[4])
                displacements = self._numeric_result_array(arrays[disp_index])
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Story {story} point {point_name}: could not parse JointDispl arrays: {exc}")
                continue
            records: dict[tuple[float, str], float] = {}
            count = min(len(step_types), len(step_numbers), len(displacements))
            for index in range(count):
                key = (round(float(step_numbers[index]), 8), str(step_types[index]))
                records[key] = float(displacements[index])
            by_story[int(story)] = records

        rows: list[dict[str, object]] = []
        max_by_story: dict[int, dict[str, object]] = {}
        story_numbers = sorted(story for story in by_story if story > 0)
        for story in story_numbers:
            upper_records = by_story.get(story, {})
            lower_records = by_story.get(story - 1, {})
            for key, upper_u in upper_records.items():
                lower_u = lower_records.get(key)
                if lower_u is None and story == 1:
                    lower_u = 0.0
                if lower_u is None:
                    continue
                relative = abs(float(upper_u) - float(lower_u))
                ratio = relative / story_height_m if story_height_m else 0.0
                row = {
                    "case": case_name,
                    "direction": direction_upper,
                    "story": story,
                    "step_number": key[0],
                    "load_step": key[1],
                    "lower_displacement_m": lower_u,
                    "upper_displacement_m": upper_u,
                    "relative_displacement_m": relative,
                    "drift_ratio": ratio,
                    "drift_percent": ratio * 100.0,
                }
                rows.append(row)
                current = max_by_story.get(story)
                if current is None or ratio > float(current.get("drift_ratio", 0.0)):
                    max_by_story[story] = row

        max_row = max(rows, key=lambda row: float(row.get("drift_ratio", 0.0)), default=None)
        return {
            "case": case_name,
            "direction": direction_upper,
            "available": bool(rows),
            "story_height_m": story_height_m,
            "record_count": len(rows),
            "max_drift": max_row,
            "max_by_story": [max_by_story[story] for story in sorted(max_by_story)],
            "result_setup": result_setup,
            "warnings": warnings[:20],
        }

    def run_analysis(self, case_names: Sequence[str] | None = None) -> dict[str, object]:
        """Run SAP2000 analysis for the current model.

        SAP2000 sometimes keeps analysis case run flags from the model template
        state. When pushover cases are generated, explicitly marking them for
        run makes the dashboard option behave as "solve these pushover cases",
        not merely "save a model that contains pushover definitions".
        """
        if case_names:
            self.set_analysis_run_cases(case_names)
        self._checked_optional_call(lambda: self.sap_model.Analyze.CreateAnalysisModel(), "Analyze.CreateAnalysisModel")
        self._check(self.sap_model.Analyze.RunAnalysis(), "Analyze.RunAnalysis")
        return self.get_analysis_case_status(case_names or [])

    def set_analysis_run_cases(self, case_names: Sequence[str]) -> None:
        """Best-effort set of analysis cases that SAP2000 should run."""
        self._require_model()
        # Clear all run flags first where supported, then enable the requested
        # cases. Some versions expose the optional All flag, some do not.
        self._checked_optional_call(lambda: self.sap_model.Analyze.SetRunCaseFlag("", False, True), "Analyze.SetRunCaseFlag(clear all)")
        for case_name in case_names:
            callbacks = (
                lambda name=case_name: self.sap_model.Analyze.SetRunCaseFlag(name, True, False),
                lambda name=case_name: self.sap_model.Analyze.SetRunCaseFlag(name, True),
            )
            self._checked_first_success(callbacks, f"Analyze.SetRunCaseFlag({case_name})")

    def get_analysis_case_status(self, case_names: Sequence[str]) -> dict[str, object]:
        """Return a best-effort analysis status summary after solving."""
        status: dict[str, object] = {"requested_cases": list(case_names), "case_status": {}, "raw": None}
        try:
            result = self.sap_model.Analyze.GetCaseStatus(0, [], [])
        except Exception as exc:  # noqa: BLE001 - OAPI availability varies by version.
            status["message"] = f"Analyze.GetCaseStatus is not available: {exc}"
            return status

        status["raw"] = str(result)
        if isinstance(result, (list, tuple)):
            names = [item for item in result if isinstance(item, (list, tuple)) and all(isinstance(value, str) for value in item)]
            values = [item for item in result if isinstance(item, (list, tuple)) and all(isinstance(value, int) for value in item)]
            if names and values:
                status["case_status"] = {name: state for name, state in zip(names[0], values[0])}
        return status

    def capture_model_screenshot(self, path: Path, delay_s: float = 0.8) -> bool:
        """Capture the current SAP2000 model window as a PNG image."""
        self._prepare_model_view()
        return self._capture_sap_window(path, delay_s)

    def capture_deformed_screenshot(self, path: Path, case_name: str = "PUSHOVER_X", delay_s: float = 0.8) -> bool:
        """Try to display an analysis/result view and capture the SAP2000 window."""
        self._prepare_deformed_view(case_name)
        return self._capture_sap_window(path, delay_s)

    def set_mass_source(self) -> None:
        """Define mass source from self-mass and assigned loads where supported."""
        try:
            self._check(self.sap_model.PropMaterial.SetMassSource_1(True, True, True, 2, ["DEAD", "LIVE"], [1.0, 0.3]), "PropMaterial.SetMassSource_1")
        except Exception:  # noqa: BLE001 - mass source signatures vary by SAP2000 version.
            print("Warning: Mass source could not be set with this SAP2000 OAPI signature.")

    def save(self, path: Path) -> None:
        """Save model as .sdb."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._check(self.sap_model.File.Save(str(path)), f"File.Save({path})")

    def save_current(self) -> None:
        """Save the current .sdb without changing its name.

        SAP2000 distinguishes a regular Save from Save As when persisting
        analysis results. Calling the OAPI method without a file name keeps the
        current model path and preserves the solved-result database.
        """
        self._check(self.sap_model.File.Save(""), "File.Save(current model)")

    def save_text(self, path: Path) -> None:
        """Export the current model to SAP2000 text format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._check(self.sap_model.File.Save(str(path)), f"File.Save({path})")

    def open_file(self, path: Path) -> None:
        """Open a SAP2000 file or text model."""
        self._require_model()
        autoclose = path.suffix.lower() in {".$2k", ".s2k"}
        if autoclose:
            self._start_import_log_autoclose(timeout_s=90)
        self._check(self.sap_model.File.OpenFile(str(path)), f"File.OpenFile({path})")
        if autoclose:
            self._start_import_log_autoclose(timeout_s=10)
        self._point_names.clear()

    def count_frame_hinge_assignments(self, frame_names: Sequence[str]) -> dict[str, object]:
        """Count assigned frame hinges using SAP2000's GetHingeAssigns API."""
        checked = 0
        assigned_frames = 0
        total_hinges = 0
        warnings: list[str] = []
        for frame_name in frame_names:
            try:
                result = self.sap_model.FrameObj.GetHingeAssigns(frame_name, 0, [], [], [], [], [], [])
                number = self._first_int(result)
                checked += 1
                if number > 0:
                    assigned_frames += 1
                    total_hinges += number
            except Exception as exc:  # noqa: BLE001 - OAPI compatibility/invalid frame names.
                warnings.append(f"{frame_name}: {exc}")
        return {
            "checked_frames": checked,
            "assigned_frames": assigned_frames,
            "total_hinges": total_hinges,
            "warnings": warnings[:10],
        }

    def close(self, save: bool = False) -> None:
        """Close SAP2000 application."""
        if self.sap_object is not None:
            self.sap_object.ApplicationExit(save)
        self.sap_object = None
        self.sap_model = None

    def _get_point_name(self, point: Point3D) -> str:
        """Return point object name at a coordinate, creating a joint if necessary."""
        key = self._point_key(point)
        if key in self._point_names:
            return self._point_names[key]
        existing = self._find_point_name_at(point)
        if existing:
            self._point_names[key] = existing
            return existing

        user_name = self._point_user_name(point)
        errors: list[str] = []
        candidates = (
            lambda: self.sap_model.PointObj.AddCartesian(point.x, point.y, point.z),
            lambda: self.sap_model.PointObj.AddCartesian(point.x, point.y, point.z, user_name),
            lambda: self.sap_model.PointObj.AddCartesian(point.x, point.y, point.z, user_name, user_name),
            lambda: self.sap_model.PointObj.AddCartesian(point.x, point.y, point.z, user_name, user_name, "Global"),
            lambda: self.sap_model.PointObj.AddCartesian(point.x, point.y, point.z, user_name, "Global"),
            lambda: self.sap_model.PointObj.AddCartesian(point.x, point.y, point.z, "", "Global"),
            lambda: self.sap_model.PointObj.AddCartesian(point.x, point.y, point.z, "", "", "Global"),
        )
        for callback in candidates:
            try:
                result = callback()
                ret, name = self._parse_ret_name(result, "")
                if ret == 0 and name:
                    self._point_names[key] = name
                    return name
                existing = self._find_point_name_at(point)
                if existing:
                    self._point_names[key] = existing
                    return existing
                errors.append(f"ret={ret}, result={result!r}")
            except Exception as exc:  # noqa: BLE001 - SAP2000 point signatures vary.
                errors.append(str(exc))
        raise SapApiError(f"SAP2000 API call failed in PointObj.AddCartesian at {point}. {' | '.join(errors)}")

    def _add_frame_by_coord(
        self,
        point_i: Point3D,
        point_j: Point3D,
        section_name: str,
        user_name: str,
        errors: list[str],
    ) -> str | None:
        """Create a frame directly from coordinates, avoiding fragile point creation calls."""
        candidates = (
            lambda: self.sap_model.FrameObj.AddByCoord(
                point_i.x,
                point_i.y,
                point_i.z,
                point_j.x,
                point_j.y,
                point_j.z,
                user_name,
                section_name,
                user_name,
                "Global",
            ),
            lambda: self.sap_model.FrameObj.AddByCoord(
                point_i.x,
                point_i.y,
                point_i.z,
                point_j.x,
                point_j.y,
                point_j.z,
                user_name,
                section_name,
                user_name,
            ),
            lambda: self.sap_model.FrameObj.AddByCoord(
                point_i.x,
                point_i.y,
                point_i.z,
                point_j.x,
                point_j.y,
                point_j.z,
                user_name,
                section_name,
            ),
            lambda: self.sap_model.FrameObj.AddByCoord(
                point_i.x,
                point_i.y,
                point_i.z,
                point_j.x,
                point_j.y,
                point_j.z,
            ),
        )
        for callback in candidates:
            try:
                result = callback()
                ret, obj_name = self._parse_ret_name(result, user_name)
                if ret == 0:
                    frame_name = obj_name or user_name
                    if section_name:
                        self._checked_optional_call(lambda name=frame_name: self.sap_model.FrameObj.SetSection(name, section_name), f"FrameObj.SetSection({frame_name})")
                    return frame_name
                errors.append(f"ret={ret}, result={result!r}")
            except Exception as exc:  # noqa: BLE001 - keep OAPI signature fallback broad.
                errors.append(str(exc))
        return None

    def _cache_frame_end_points(self, frame_name: str, point_i: Point3D, point_j: Point3D) -> None:
        """Cache SAP point names created implicitly by a frame object."""
        if not frame_name:
            return
        try:
            result = self.sap_model.FrameObj.GetPoints(frame_name)
        except Exception:
            return
        names = self._string_values(result)
        if len(names) >= 2:
            self._point_names[self._point_key(point_i)] = names[0]
            self._point_names[self._point_key(point_j)] = names[1]

    @staticmethod
    def _point_user_name(point: Point3D) -> str:
        """Return a deterministic SAP-safe point name from rounded coordinates."""
        def token(value: float) -> str:
            scaled = int(round(value * 1000.0))
            prefix = "N" if scaled < 0 else "P"
            return f"{prefix}{abs(scaled)}"

        return f"PT_{token(point.x)}_{token(point.y)}_{token(point.z)}"

    def _find_point_name_at(self, point: Point3D, tolerance: float = 1.0e-6) -> str | None:
        """Find an existing point object at the given coordinate, if SAP2000 already has one."""
        try:
            result = self.sap_model.PointObj.GetNameList()
        except Exception:
            return None
        names = self._string_values(result)
        for name in names:
            try:
                coords = self.sap_model.PointObj.GetCoordCartesian(name)
            except Exception:
                continue
            values = self._numeric_values(coords)
            if len(values) < 3:
                continue
            if (
                abs(values[0] - point.x) <= tolerance
                and abs(values[1] - point.y) <= tolerance
                and abs(values[2] - point.z) <= tolerance
            ):
                return name
        return None

    @staticmethod
    def _point_key(point: Point3D) -> tuple[float, float, float]:
        """Stable rounded coordinate key for SAP2000 point-object reuse."""
        return (round(point.x, 6), round(point.y, 6), round(point.z, 6))

    def _require_model(self) -> None:
        """Raise if SAP2000 model object is not available."""
        if self.sap_model is None:
            raise SapApiError("SAP2000 model is not initialized. Call start() first.")

    def _ensure_load_pattern(self, name: str, pattern_type: int, self_weight_multiplier: float) -> None:
        """Create a load pattern only when it is not already present."""
        if name in self._load_pattern_names():
            return
        ret = self.sap_model.LoadPatterns.Add(name, pattern_type, self_weight_multiplier, True)
        code = self._return_code(ret)
        if code == 0:
            return
        if code == 1:
            print(f"Warning: Load pattern {name} may already exist; continuing.")
            return
        self._check(ret, f"LoadPatterns.Add({name})")

    def _load_pattern_names(self) -> set[str]:
        """Return existing load pattern names using the current OAPI signature."""
        try:
            result = self.sap_model.LoadPatterns.GetNameList()
        except Exception:  # noqa: BLE001 - fallback to adding when OAPI cannot list names.
            return set()

        if not isinstance(result, (list, tuple)):
            return set()
        names: set[str] = set()
        for item in result:
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, (list, tuple)):
                names.update(str(value) for value in item if isinstance(value, str))
        return names

    def _set_static_nonlinear_loads(self, case_name: str, load_pattern: str) -> None:
        """Set nonlinear static case loads with version-tolerant signatures."""
        candidates = (
            lambda: self.sap_model.LoadCases.StaticNonlinear.SetLoads(case_name, 1, ["Load"], [load_pattern], [1.0]),
            lambda: self.sap_model.LoadCases.StaticNonlinear.SetLoads(case_name, 1, [load_pattern], [1.0]),
        )
        self._checked_first_success(candidates, f"StaticNonlinear.SetLoads({case_name})")

    def _set_static_nonlinear_initial_case(self, case_name: str) -> None:
        """Set pushover initial conditions to start from zero/none where supported."""
        candidates = (
            lambda: self.sap_model.LoadCases.StaticNonlinear.SetInitialCase(case_name, ""),
            lambda: self.sap_model.LoadCases.StaticNonlinear.SetInitialCase(case_name, "None"),
            lambda: self.sap_model.LoadCases.StaticNonlinear.SetInitialCase(case_name, "NONE"),
        )
        self._checked_first_success(candidates, f"StaticNonlinear.SetInitialCase({case_name})", required=False)

    def _select_case_for_step_by_step_results(self, case_name: str) -> dict[str, object]:
        """Select one case and request saved nonlinear states instead of envelopes."""
        setup: dict[str, object] = {
            "case": case_name,
            "multi_step_static_option": None,
            "nl_static_option": None,
            "warnings": [],
        }
        self._checked_optional_call(lambda: self.sap_model.Results.Setup.DeselectAllCasesAndCombosForOutput(), "Results.Setup.DeselectAllCasesAndCombosForOutput")
        self._checked_optional_call(lambda: self.sap_model.Results.Setup.SetCaseSelectedForOutput(case_name), f"Results.Setup.SetCaseSelectedForOutput({case_name})")
        for method_name, key in (
            ("SetOptionMultiStepStatic", "multi_step_static_option"),
            ("SetOptionNLStatic", "nl_static_option"),
        ):
            try:
                ret = getattr(self.sap_model.Results.Setup, method_name)(2)
                self._check(ret, f"Results.Setup.{method_name}(2)")
                setup[key] = 2
            except Exception as exc:  # noqa: BLE001 - option availability differs by SAP version.
                setup["warnings"].append(f"{method_name}(2): {exc}")
        return setup

    def _set_pushover_displacement_control(self, case_name: str, point_name: str, direction: str, target_displacement_m: float) -> None:
        """Set roof displacement control if supported by the installed OAPI."""
        dof = 1 if direction.upper() == "X" else 2
        candidates = (
            # SAP2000 v22 cCaseStaticNonlinear.SetLoadApplication:
            # Name, LoadControl, DispType, Displ, Monitor, DOF, PointName, GDispl.
            # Try monitored displacement-control variants first.
            lambda: self.sap_model.LoadCases.StaticNonlinear.SetLoadApplication(case_name, 2, 2, target_displacement_m, 1, dof, point_name, ""),
            lambda: self.sap_model.LoadCases.StaticNonlinear.SetLoadApplication(case_name, 2, 1, target_displacement_m, 1, dof, point_name, ""),
            lambda: self.sap_model.LoadCases.StaticNonlinear.SetLoadApplication(case_name, 1, 2, target_displacement_m, 1, dof, point_name, ""),
            lambda: self.sap_model.LoadCases.StaticNonlinear.SetDisplControl(case_name, point_name, dof, target_displacement_m, True),
            lambda: self.sap_model.LoadCases.StaticNonlinear.SetDisplacementControl(case_name, point_name, dof, target_displacement_m),
            lambda: self.sap_model.LoadCases.StaticNonlinear.SetTargetDispl(case_name, point_name, dof, target_displacement_m),
        )
        self._checked_first_success(candidates, f"StaticNonlinear displacement control ({case_name})")

    def _prepare_model_view(self) -> None:
        """Best-effort SAP2000 view setup for geometry screenshots."""
        self._checked_optional_call(lambda: self.sap_model.View.RefreshView(0, False), "View.RefreshView")
        self._checked_optional_call(lambda: self.sap_model.View.ZoomFull(), "View.ZoomFull")
        self._checked_optional_call(lambda: self.sap_model.View.SetView(3), "View.SetView(3D)")
        self._checked_optional_call(lambda: self.sap_model.View.RefreshView(0, False), "View.RefreshView")

    def _prepare_deformed_view(self, case_name: str) -> None:
        """Best-effort SAP2000 view setup for analysis-result screenshots."""
        self._checked_optional_call(lambda: self.sap_model.Results.Setup.DeselectAllCasesAndCombosForOutput(), "Results.Setup.DeselectAllCasesAndCombosForOutput")
        self._checked_optional_call(lambda: self.sap_model.Results.Setup.SetCaseSelectedForOutput(case_name), f"Results.Setup.SetCaseSelectedForOutput({case_name})")
        candidates = (
            lambda: self.sap_model.View.ShowDeformedShape(case_name),
            lambda: self.sap_model.View.ShowDeformedShape(case_name, 0, 0, 1.0),
            lambda: self.sap_model.View.ShowDeformedShape(case_name, "Max", 0, 1.0),
        )
        self._checked_first_success(candidates, f"View.ShowDeformedShape({case_name})", required=False)
        self._checked_optional_call(lambda: self.sap_model.View.RefreshView(0, False), "View.RefreshView")

    def _capture_sap_window(self, path: Path, delay_s: float) -> bool:
        """Capture the visible SAP2000 window using Windows APIs through PowerShell."""
        path.parent.mkdir(parents=True, exist_ok=True)
        time.sleep(delay_s)
        script = rf"""
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class NativeMethods {{
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
}}
public struct RECT {{
  public int Left;
  public int Top;
  public int Right;
  public int Bottom;
}}
"@
$proc = Get-Process | Where-Object {{ $_.MainWindowHandle -ne 0 -and $_.ProcessName -like "SAP2000*" }} | Select-Object -First 1
if (-not $proc) {{
  $proc = Get-Process | Where-Object {{ $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -like "SAP2000*" }} | Select-Object -First 1
}}
if (-not $proc) {{ exit 2 }}
$handle = $proc.MainWindowHandle
[NativeMethods]::SetForegroundWindow($handle) | Out-Null
Start-Sleep -Milliseconds 250
$rect = New-Object RECT
if (-not [NativeMethods]::GetWindowRect($handle, [ref]$rect)) {{ exit 3 }}
$width = $rect.Right - $rect.Left
$height = $rect.Bottom - $rect.Top
if ($width -lt 50 -or $height -lt 50) {{ exit 4 }}
$bitmap = New-Object System.Drawing.Bitmap $width, $height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bitmap.Size)
$bitmap.Save("{str(path).replace('"', '""')}", [System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose()
$bitmap.Dispose()
"""
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if completed.returncode != 0:
            print(f"Warning: SAP2000 screenshot capture failed with code {completed.returncode}: {completed.stderr.strip()}")
            return False
        return path.exists()

    def _start_import_log_autoclose(self, timeout_s: int = 60) -> None:
        """Close SAP2000 text-import log dialogs that otherwise block automation."""
        script = rf"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System;
using System.Text;
using System.Runtime.InteropServices;
public class NativeMethods {{
  public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
  [DllImport("user32.dll", CharSet=CharSet.Auto)] public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);
  [DllImport("user32.dll", CharSet=CharSet.Auto)] public static extern IntPtr FindWindowEx(IntPtr hwndParent, IntPtr hwndChildAfter, string lpszClass, string lpszWindow);
  [DllImport("user32.dll", CharSet=CharSet.Auto)] public static extern int GetClassName(IntPtr hWnd, StringBuilder lpClassName, int nMaxCount);
  [DllImport("user32.dll")] public static extern IntPtr SendMessage(IntPtr hWnd, int Msg, IntPtr wParam, IntPtr lParam);
  [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr hWnd, int Msg, IntPtr wParam, IntPtr lParam);
  [DllImport("user32.dll")] public static extern bool IsWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool IsWindowEnabled(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
}}
"@
$deadline = (Get-Date).AddSeconds({int(timeout_s)})
$bmClick = 0x00F5
$wmClose = 0x0010
$wshell = New-Object -ComObject WScript.Shell
while ((Get-Date) -lt $deadline) {{
  foreach ($title in @("Access Database Import Log", "Database Import")) {{
    try {{
      if ($wshell.AppActivate($title)) {{
        Start-Sleep -Milliseconds 120
        $wshell.SendKeys("{{ENTER}}")
        Start-Sleep -Milliseconds 500
      }}
    }} catch {{ }}
  }}
  $script:target = [IntPtr]::Zero
  $callback = [NativeMethods+EnumWindowsProc] {{
    param([IntPtr]$hWnd, [IntPtr]$lParam)
    $buffer = New-Object System.Text.StringBuilder 512
    [NativeMethods]::GetWindowText($hWnd, $buffer, $buffer.Capacity) | Out-Null
    $title = $buffer.ToString()
    if ($title -like "*Access Database Import Log*" -or $title -like "*Database Import*") {{
      $script:target = $hWnd
      return $false
    }}
    return $true
  }}
  [NativeMethods]::EnumWindows($callback, [IntPtr]::Zero) | Out-Null
  if ($script:target -ne [IntPtr]::Zero) {{
    $button = [NativeMethods]::FindWindowEx($script:target, [IntPtr]::Zero, "Button", "Done")
    if ($button -eq [IntPtr]::Zero) {{
      $child = [IntPtr]::Zero
      do {{
        $child = [NativeMethods]::FindWindowEx($script:target, $child, "Button", $null)
        if ($child -ne [IntPtr]::Zero) {{
          $textBuffer = New-Object System.Text.StringBuilder 256
          [NativeMethods]::GetWindowText($child, $textBuffer, $textBuffer.Capacity) | Out-Null
          if ($textBuffer.ToString() -like "*Done*") {{
            $button = $child
            break
          }}
        }}
      }} while ($child -ne [IntPtr]::Zero)
    }}
    if ($button -ne [IntPtr]::Zero -and [NativeMethods]::IsWindowEnabled($button)) {{
      [NativeMethods]::SetForegroundWindow($script:target) | Out-Null
      [NativeMethods]::SendMessage($button, $bmClick, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null
      Start-Sleep -Milliseconds 400
      if (-not [NativeMethods]::IsWindow($script:target)) {{ exit 0 }}
    }} else {{
      [NativeMethods]::SetForegroundWindow($script:target) | Out-Null
      Start-Sleep -Milliseconds 120
      [System.Windows.Forms.SendKeys]::SendWait("{{ENTER}}")
      Start-Sleep -Milliseconds 400
      if (-not [NativeMethods]::IsWindow($script:target)) {{ exit 0 }}
      [NativeMethods]::PostMessage($script:target, $wmClose, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null
    }}
  }}
  Start-Sleep -Milliseconds 500
}}
"""
        try:
            subprocess.Popen(
                ["powershell", "-NoProfile", "-Sta", "-ExecutionPolicy", "Bypass", "-Command", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
        except Exception as exc:  # noqa: BLE001 - this helper must never stop model generation.
            print(f"Warning: SAP2000 import log autoclose helper could not start: {exc}")

    def _checked_optional_call(self, callback: object, context: str) -> None:
        """Run an OAPI call and warn instead of failing when a version lacks it."""
        try:
            self._check(callback(), context)
        except Exception as exc:  # noqa: BLE001 - COM version compatibility shim.
            print(f"Warning: {context} could not be applied: {exc}")

    def _try_first_success(self, callbacks: Sequence[object], context: str) -> bool:
        """Try compatible OAPI calls and return whether any succeeded."""
        for callback in callbacks:
            try:
                self._check(callback(), context)
                return True
            except Exception:
                continue
        return False

    def _checked_first_success(self, callbacks: Sequence[object], context: str, required: bool = True) -> None:
        """Try several OAPI signatures and accept the first successful one."""
        errors: list[str] = []
        for callback in callbacks:
            try:
                self._check(callback(), context)
                return
            except Exception as exc:  # noqa: BLE001 - compatibility variants are intentionally broad.
                errors.append(str(exc))
        message = f"{context} failed for all known SAP2000 OAPI signatures: {' | '.join(errors)}"
        if required:
            raise SapApiError(message)
        print(f"Warning: {message}")

    @staticmethod
    def _parse_ret_name(result: object, fallback_name: str) -> tuple[int, str]:
        """Parse common COM return shapes containing return code and object name."""
        if isinstance(result, (list, tuple)):
            int_values = [item for item in result if isinstance(item, int)]
            if not int_values:
                raise SapApiError(f"Could not parse SAP2000 return code from: {result!r}")
            ret = int_values[-1]
            names = [item for item in result if isinstance(item, str) and item]
            return ret, names[-1] if names else fallback_name
        return int(result), fallback_name

    @staticmethod
    def _first_int(result: object) -> int:
        """Return the first integer value from a common COM return shape."""
        if isinstance(result, (list, tuple)):
            for item in result:
                if isinstance(item, int):
                    return item
        return int(result)

    @staticmethod
    def _check(ret: object, context: str) -> None:
        """Check SAP2000 integer return code and raise a meaningful error."""
        code = Sap2000Api._return_code(ret)
        if code != 0:
            raise SapApiError(f"SAP2000 API call failed in {context}. Return code: {code}")

    @staticmethod
    def _return_code(ret: object) -> int:
        """Extract SAP2000 integer return code from common COM return shapes."""
        if isinstance(ret, (list, tuple)):
            int_values = [item for item in ret if isinstance(item, int)]
            if not int_values:
                raise SapApiError(f"SAP2000 API call returned an unparsable value: {ret!r}")
            return int_values[-1]
        return int(ret)

    @staticmethod
    def _summarize_hinge_result(result: object, case_name: str) -> dict[str, object]:
        """Create a JSON-safe compact summary from a raw hinge result payload."""
        if not isinstance(result, (list, tuple)):
            return {"case": case_name, "available": True, "raw_return_code": int(result), "record_count": 0}

        strings: list[str] = []
        numbers: list[float] = []
        for item in result:
            if isinstance(item, str):
                strings.append(item)
            elif isinstance(item, (int, float)):
                numbers.append(float(item))
            elif isinstance(item, (list, tuple)):
                strings.extend(str(value) for value in item if isinstance(value, str))
                numbers.extend(float(value) for value in item if isinstance(value, (int, float)))

        state_tokens = [value for value in strings if value.upper() in {"A", "B", "C", "D", "E", "IO", "LS", "CP"}]
        return {
            "case": case_name,
            "available": True,
            "string_count": len(strings),
            "numeric_count": len(numbers),
            "state_counts": {token: state_tokens.count(token) for token in sorted(set(state_tokens))},
            "sample_strings": strings[:10],
        }

    @staticmethod
    def _result_arrays(result: object) -> list[list[object]]:
        """Extract array-like output fields from a SAP2000 result tuple."""
        if not isinstance(result, (list, tuple)):
            return []
        arrays: list[list[object]] = []
        for item in result:
            if isinstance(item, (list, tuple)):
                arrays.append(Sap2000Api._flatten_result_values(item))
        return arrays

    @staticmethod
    def _build_capacity_points(
        base_steps: Sequence[float],
        base_step_types: Sequence[str],
        shears: Sequence[float],
        joint_steps: Sequence[float],
        joint_step_types: Sequence[str],
        displacements: Sequence[float],
    ) -> list[dict[str, object]]:
        """Pair roof displacement and base shear into monotonic capacity points."""
        base_labels = {label.lower() for label in base_step_types}
        joint_labels = {label.lower() for label in joint_step_types}
        is_envelope = bool(base_labels | joint_labels) and (base_labels | joint_labels).issubset({"max", "min", ""})
        if is_envelope:
            max_displacement = max((abs(value) for value in displacements), default=0.0)
            max_shear = max((abs(value) for value in shears), default=0.0)
            return [
                {
                    "step": 0.0,
                    "step_number": 0.0,
                    "load_step": "Origin",
                    "joint_load_step": "Origin",
                    "control_displacement_m": 0.0,
                    "base_shear_kn": 0.0,
                    "source": "origin",
                },
                {
                    "step": max((abs(value) for value in base_steps), default=0.0),
                    "step_number": max((abs(value) for value in base_steps), default=0.0),
                    "load_step": "Envelope",
                    "joint_load_step": "Envelope",
                    "control_displacement_m": round(max_displacement, 6),
                    "base_shear_kn": round(max_shear, 3),
                    "source": "max_abs_displacement_and_base_shear",
                },
            ]

        displacement_by_step = {Sap2000Api._step_key(step): disp for step, disp in zip(joint_steps, displacements)}
        raw_points: list[dict[str, object]] = []
        for index, (step, shear) in enumerate(zip(base_steps, shears)):
            step_key = Sap2000Api._step_key(step)
            if step_key not in displacement_by_step:
                continue
            displacement = displacement_by_step[step_key]
            raw_points.append(
                {
                    "step": step,
                    "step_number": step,
                    "load_step": base_step_types[index] if index < len(base_step_types) else "",
                    "joint_load_step": joint_step_types[index] if index < len(joint_step_types) else "",
                    "control_displacement_m": round(abs(displacement), 6),
                    "base_shear_kn": round(abs(shear), 3),
                    "source": "matched_step",
                }
            )

        by_displacement: dict[float, dict[str, object]] = {
            0.0: {
                "step": 0.0,
                "step_number": 0.0,
                "load_step": "Origin",
                "joint_load_step": "Origin",
                "control_displacement_m": 0.0,
                "base_shear_kn": 0.0,
                "source": "origin",
            }
        }
        for point in raw_points:
            displacement_key = round(float(point["control_displacement_m"]), 6)
            existing = by_displacement.get(displacement_key)
            if existing is None or float(point["base_shear_kn"]) > float(existing["base_shear_kn"]):
                by_displacement[displacement_key] = point
        return [by_displacement[key] for key in sorted(by_displacement)]

    @staticmethod
    def _flatten_result_values(value: object) -> list[object]:
        """Flatten nested OAPI result arrays while preserving scalar values."""
        if isinstance(value, (list, tuple)):
            flattened: list[object] = []
            for item in value:
                flattened.extend(Sap2000Api._flatten_result_values(item))
            return flattened
        return [value]

    @staticmethod
    def _numeric_result_array(values: Sequence[object]) -> list[float]:
        """Convert a SAP2000 result field to floats, tolerating nested arrays."""
        numbers: list[float] = []
        for value in Sap2000Api._flatten_result_values(values):
            try:
                numbers.append(float(value))
            except (TypeError, ValueError):
                continue
        return numbers

    @staticmethod
    def _numeric_values(values: object) -> list[float]:
        """Return numeric scalar values from any common OAPI return shape."""
        source = values if isinstance(values, (list, tuple)) else [values]
        return Sap2000Api._numeric_result_array(source)

    @staticmethod
    def _string_result_array(values: Sequence[object]) -> list[str]:
        """Convert a SAP2000 result field to strings, tolerating nested arrays."""
        return [str(value) for value in Sap2000Api._flatten_result_values(values)]

    @staticmethod
    def _string_values(values: object) -> list[str]:
        """Return non-empty string scalar values from any common OAPI return shape."""
        source = values if isinstance(values, (list, tuple)) else [values]
        return [str(value) for value in Sap2000Api._flatten_result_values(source) if isinstance(value, str) and value]

    @staticmethod
    def _step_key(value: float) -> float:
        """Normalize analysis step numbers for matching across result tables."""
        return round(float(value), 8)
