#!/usr/bin/env python3
"""W434 statistical baseline builder and evaluator.

This program parses WEETECH W434 text report files, keeps only reports whose
overall test result is PASSED, builds a statistical baseline grouped by
"Unit under test", saves that baseline to JSON, and evaluates individual new
reports against the saved baseline.

The program is intentionally conservative and audit-oriented:

- It only uses reports whose overall summary indicates PASSED.
- It groups baselines by unit under test.
- It normalizes numeric units into SI base values.
- It tracks censored measurements such as ">8.168GOhm" and "<26.98uA".
- It computes descriptive statistics only from uncensored numeric values.
- It separately tracks censored pass behavior so a new report can be flagged if
  it drops out of the censored region into a lower/measurable value.

Typical usage:

  Build a baseline from a directory tree of text reports:
    python w434_statistical_baseline_tool.py build \
        --input-dir ./reports \
        --output-json ./baseline.json

  Evaluate a new report against a saved baseline:
    python w434_statistical_baseline_tool.py evaluate \
        --baseline-json ./baseline.json \
        --report ./new_report.txt

  Optionally limit evaluation to one unit under test from the report:
    python w434_statistical_baseline_tool.py evaluate \
        --baseline-json ./baseline.json \
        --report ./new_report.txt \
        --z-threshold 3.0
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Iterable
from typing import Optional

from tqdm import tqdm


UNIT_MULTIPLIERS = {
    "nOhm": 1e-9,
    "uOhm": 1e-6,
    "mOhm": 1e-3,
    "Ohm": 1.0,
    "kOhm": 1e3,
    "MOhm": 1e6,
    "GOhm": 1e9,
    "pF": 1e-12,
    "nF": 1e-9,
    "uF": 1e-6,
    "mF": 1e-3,
    "F": 1.0,
    "nA": 1e-9,
    "uA": 1e-6,
    "mA": 1e-3,
    "A": 1.0,
    "uV": 1e-6,
    "mV": 1e-3,
    "V": 1.0,
}

RESULT_LINE_PATTERN = re.compile(r"^(Passed|Failed|Open|Short|Arc)\s{2,}(.*)$", re.IGNORECASE)
FORMFEED_PATTERN = re.compile(r"FORMFEED", re.IGNORECASE)
PAGE_PATTERN = re.compile(r"^Page\s+\d+\s+of\s+\d+$", re.IGNORECASE)
IR_II_PATTERN = re.compile(
    r"Ir\s*=\s*([^;]+?)\s*;\s*Ii\s*=\s*([^;]+?)\s*$",
    re.IGNORECASE,
)
NUMERIC_VALUE_PATTERN = re.compile(
    r"^(?P<censor>[<>])?\s*(?P<number>[+-]?\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]+)$"
)
PARAMETERS_SECTION_PATTERN = re.compile(
    r"^\s*Parameters\s+for\s+(.+?)\s*$",
    re.IGNORECASE,
)
OVERALL_PASSED_PATTERN = re.compile(r"\bP\s*A\s*S\s*S\s*E\s*D\b", re.IGNORECASE)
OVERALL_FAILED_PATTERN = re.compile(r"\bF\s*A\s*I\s*L\s*E\s*D\b", re.IGNORECASE)
HEADER_FAILURE_RESULT_PATTERN = re.compile(
    r"(fail(?:ed)?|errors?|open|short|arc)",
    re.IGNORECASE,
)

MEASUREMENT_CATEGORY_CONTINUITY = "continuity"
MEASUREMENT_CATEGORY_LV_ISOLATION = "lv_isolation"
MEASUREMENT_CATEGORY_HV_ISOLATION_RESISTANCE = "hv_isolation_resistance"
MEASUREMENT_CATEGORY_HV_DIELECTRIC_BREAKDOWN = "hv_dielectric_breakdown"
MEASUREMENT_CATEGORY_OTHER = "other"

SECTION_KIND_CONTINUITY = "continuity"
SECTION_KIND_LV_ISOLATION = "lv_isolation"
SECTION_KIND_HV_ISOLATION = "hv_isolation"
SECTION_KIND_OTHER = "other"

RESISTANCE_UNITS = {"nOhm", "uOhm", "mOhm", "Ohm", "kOhm", "MOhm", "GOhm"}
CURRENT_UNITS = {"nA", "uA", "mA", "A"}


@dataclass
class ParsedValue:
    """Represents a parsed numeric measurement value.

    Attributes:
        raw_text: Original measurement text from the report.
        numeric_value_si: Parsed numeric value converted to SI base units.
        display_unit: Original engineering unit suffix.
        censor_type: None for direct values, "lower" for >X, "upper" for <X.
    """

    raw_text: str
    numeric_value_si: float
    display_unit: str
    censor_type: Optional[str]


@dataclass
class MeasurementRecord:
    """Normalized measurement record extracted from a W434 report.

    Attributes:
        unit_under_test: Report unit under test.
        report_path: Source report path.
        report_datetime: Date/time text from the report, if present.
        serial_number: Serial number text from the report, if present.
        overall_result: Overall report result such as PASSED.
        line_result: Per-line result such as Passed or Failed.
        test_name: Test or measurement name field.
        from_node: Source node/pin field, if present.
        to_node: Destination node/pin field, if present.
        metric_name: Logical submetric name such as VALUE, IR, or II.
        measurement_text: Original value text.
        numeric_value_si: Parsed numeric value in SI base units.
        engineering_unit: Original engineering unit.
        censor_type: None, "lower", or "upper".
        measurement_category: Measurement category such as continuity or lv_isolation.
        measurement_key: Stable grouping key for baseline statistics.
    """

    unit_under_test: str
    report_path: str
    report_datetime: str
    serial_number: str
    overall_result: str
    line_result: str
    test_name: str
    from_node: str
    to_node: str
    metric_name: str
    measurement_text: str
    numeric_value_si: float
    engineering_unit: str
    censor_type: Optional[str]
    measurement_category: str
    measurement_key: str


@dataclass
class MeasurementBaseline:
    """Statistical baseline for one measurement key within one unit under test."""

    measurement_key: str
    test_name: str
    from_node: str
    to_node: str
    metric_name: str
    engineering_unit: str
    measurement_category: str
    sample_count_total: int = 0
    sample_count_uncensored: int = 0
    sample_count_lower_censored: int = 0
    sample_count_upper_censored: int = 0
    minimum_value_si: Optional[float] = None
    maximum_value_si: Optional[float] = None
    mean_value_si: Optional[float] = None
    median_value_si: Optional[float] = None
    population_stddev_si: Optional[float] = None
    baseline_floor_value_si: Optional[float] = None
    baseline_ceiling_value_si: Optional[float] = None


@dataclass
class UnitBaseline:
    """Baseline bundle for one unit under test."""

    unit_under_test: str
    report_count: int = 0
    passed_report_count: int = 0
    measurement_baselines: dict[str, MeasurementBaseline] = field(default_factory=dict)


@dataclass
class BaselineDatabase:
    """Serializable baseline database saved to disk."""

    created_by_program: str
    units: dict[str, UnitBaseline]


class W434ReportParser:
    """Parses W434 text reports into normalized metadata and measurements."""

    def parse_report(self, report_path: Path) -> tuple[dict[str, str], list[MeasurementRecord]]:
        """Parses one W434 report file.

        Args:
            report_path: Path to the text report.

        Returns:
            Tuple of metadata dictionary and list of normalized measurement
            records.
        """
        raw_text = report_path.read_text(encoding="utf-8", errors="replace")
        lines = [self._clean_line(line) for line in raw_text.splitlines()]

        metadata = self._extract_metadata(lines)
        if self._header_has_failure(lines):
            return metadata, []

        records: list[MeasurementRecord] = []

        unit_under_test = metadata.get("unit_under_test", "")
        overall_result = metadata.get("overall_result", "")
        report_datetime = metadata.get("datetime", "")
        serial_number = metadata.get("serial_number", "")
        active_section_kind = SECTION_KIND_OTHER
        active_parameters_text = ""

        for line in lines:
            if not line or self._is_non_data_line(line):
                continue

            parameter_section_text = self._extract_parameter_section_text(line)
            if parameter_section_text is not None:
                active_parameters_text = parameter_section_text
                active_section_kind = self._section_kind_from_parameters_text(parameter_section_text)
                continue

            line_match = RESULT_LINE_PATTERN.match(line)
            if not line_match:
                continue

            line_result = line_match.group(1)
            remainder = line_match.group(2)
            split_parts = re.split(r"\s{2,}", remainder)
            split_parts = [part.strip() for part in split_parts if part.strip()]
            if len(split_parts) < 2:
                continue

            test_name = split_parts[0]
            payload_parts = split_parts[1:]
            parsed_records = self._parse_measurement_payload(
                unit_under_test=unit_under_test,
                report_path=report_path,
                report_datetime=report_datetime,
                serial_number=serial_number,
                overall_result=overall_result,
                line_result=line_result,
                test_name=test_name,
                payload_parts=payload_parts,
                active_section_kind=active_section_kind,
                active_parameters_text=active_parameters_text,
            )
            records.extend(parsed_records)

        return metadata, records

    def _header_has_failure(self, lines: list[str]) -> bool:
        """Returns True if the header summary indicates any failure condition."""
        test_result_start_index: Optional[int] = None
        for index, line in enumerate(lines):
            normalized_letters = re.sub(r"[^a-z]", "", line.lower())
            if normalized_letters == "testresult":
                test_result_start_index = index + 1
                break

        if test_result_start_index is None:
            return False

        seen_summary_entry = False
        for line in lines[test_result_start_index:]:
            stripped = line.strip()
            if not stripped and seen_summary_entry:
                break

            overall_banner_result = self._extract_overall_banner_result(stripped)
            if overall_banner_result == "FAILED":
                return True
            if overall_banner_result == "PASSED":
                break

            if ":" not in line:
                continue

            result_text = line.split(":", 1)[1].strip()
            if not result_text:
                continue
            seen_summary_entry = True

            if result_text.lower() == "no commands":
                continue
            if HEADER_FAILURE_RESULT_PATTERN.search(result_text):
                return True

        return False

    def _clean_line(self, line: str) -> str:
        """Cleans up one raw line from the report."""
        line = line.replace("\x0c", " ")
        return line.rstrip()

    def _is_non_data_line(self, line: str) -> bool:
        """Returns True for known page break or decoration lines."""
        if FORMFEED_PATTERN.search(line):
            return True
        if PAGE_PATTERN.match(line.strip()):
            return True
        stripped = line.strip()
        if not stripped:
            return True
        if set(stripped) <= set("-:=* "):
            return True
        return False

    def _extract_metadata(self, lines: list[str]) -> dict[str, str]:
        """Extracts report metadata fields from header text."""
        metadata: dict[str, str] = {
            "unit_under_test": "",
            "filename": "",
            "datetime": "",
            "serial_number": "",
            "overall_result": "",
        }

        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()
            if lower.startswith("unit under test:"):
                metadata["unit_under_test"] = stripped.split(":", 1)[1].strip()
            elif lower.startswith("filename :") or lower.startswith("filename:"):
                metadata["filename"] = stripped.split(":", 1)[1].strip()
            elif lower.startswith("date/time:"):
                metadata["datetime"] = stripped.split(":", 1)[1].strip()
            elif lower.startswith("serial number:"):
                metadata["serial_number"] = stripped.split(":", 1)[1].strip()
            overall_banner_result = self._extract_overall_banner_result(stripped)
            if overall_banner_result is not None:
                metadata["overall_result"] = overall_banner_result

        return metadata

    def _extract_overall_banner_result(self, stripped_line: str) -> Optional[str]:
        """Extracts explicit overall PASSED/FAILED banner from one line."""
        if not stripped_line or ":" in stripped_line:
            return None
        if OVERALL_FAILED_PATTERN.search(stripped_line):
            return "FAILED"
        if OVERALL_PASSED_PATTERN.search(stripped_line):
            return "PASSED"
        return None

    def _parse_measurement_payload(
        self,
        unit_under_test: str,
        report_path: Path,
        report_datetime: str,
        serial_number: str,
        overall_result: str,
        line_result: str,
        test_name: str,
        payload_parts: list[str],
        active_section_kind: str,
        active_parameters_text: str,
    ) -> list[MeasurementRecord]:
        """Parses the measurement part of a passed/failed result line."""
        measurement_text = payload_parts[-1]
        node_parts = payload_parts[:-1]

        if len(node_parts) == 0:
            from_node = ""
            to_node = ""
        elif len(node_parts) == 1:
            from_node = node_parts[0]
            to_node = ""
        else:
            from_node = node_parts[0]
            to_node = node_parts[1]

        ir_ii_match = IR_II_PATTERN.match(measurement_text)
        if ir_ii_match:
            ir_value = self._parse_scalar_value(ir_ii_match.group(1))
            ii_value = self._parse_scalar_value(ir_ii_match.group(2))
            ir_record = self._build_measurement_record(
                unit_under_test=unit_under_test,
                report_path=report_path,
                report_datetime=report_datetime,
                serial_number=serial_number,
                overall_result=overall_result,
                line_result=line_result,
                test_name=test_name,
                from_node=from_node,
                to_node=to_node,
                metric_name="IR",
                parsed_value=ir_value,
                measurement_category=self._resolve_measurement_category(
                    active_section_kind=active_section_kind,
                    active_parameters_text=active_parameters_text,
                    engineering_unit=ir_value.display_unit,
                ),
            )
            ii_record = self._build_measurement_record(
                unit_under_test=unit_under_test,
                report_path=report_path,
                report_datetime=report_datetime,
                serial_number=serial_number,
                overall_result=overall_result,
                line_result=line_result,
                test_name=test_name,
                from_node=from_node,
                to_node=to_node,
                metric_name="II",
                parsed_value=ii_value,
                measurement_category=self._resolve_measurement_category(
                    active_section_kind=active_section_kind,
                    active_parameters_text=active_parameters_text,
                    engineering_unit=ii_value.display_unit,
                ),
            )
            return [ir_record, ii_record]

        parsed_value = self._parse_scalar_value(measurement_text)
        record = self._build_measurement_record(
            unit_under_test=unit_under_test,
            report_path=report_path,
            report_datetime=report_datetime,
            serial_number=serial_number,
            overall_result=overall_result,
            line_result=line_result,
            test_name=test_name,
            from_node=from_node,
            to_node=to_node,
            metric_name="VALUE",
            parsed_value=parsed_value,
            measurement_category=self._resolve_measurement_category(
                active_section_kind=active_section_kind,
                active_parameters_text=active_parameters_text,
                engineering_unit=parsed_value.display_unit,
            ),
        )
        return [record]

    def _build_measurement_record(
        self,
        unit_under_test: str,
        report_path: Path,
        report_datetime: str,
        serial_number: str,
        overall_result: str,
        line_result: str,
        test_name: str,
        from_node: str,
        to_node: str,
        metric_name: str,
        parsed_value: ParsedValue,
        measurement_category: str,
    ) -> MeasurementRecord:
        """Builds a normalized measurement record."""
        measurement_key = self._make_measurement_key(
            measurement_category=measurement_category,
            test_name=test_name,
            from_node=from_node,
            to_node=to_node,
            metric_name=metric_name,
        )
        return MeasurementRecord(
            unit_under_test=unit_under_test,
            report_path=str(report_path),
            report_datetime=report_datetime,
            serial_number=serial_number,
            overall_result=overall_result,
            line_result=line_result,
            test_name=test_name,
            from_node=from_node,
            to_node=to_node,
            metric_name=metric_name,
            measurement_text=parsed_value.raw_text,
            numeric_value_si=parsed_value.numeric_value_si,
            engineering_unit=parsed_value.display_unit,
            censor_type=parsed_value.censor_type,
            measurement_category=measurement_category,
            measurement_key=measurement_key,
        )

    def _make_measurement_key(
        self,
        measurement_category: str,
        test_name: str,
        from_node: str,
        to_node: str,
        metric_name: str,
    ) -> str:
        """Builds a stable baseline key for one logical measurement."""
        return "|".join([
            measurement_category.strip(),
            test_name.strip(),
            from_node.strip(),
            to_node.strip(),
            metric_name.strip(),
        ])

    def _extract_parameter_section_text(self, line: str) -> Optional[str]:
        """Extracts parameters section text from a line.

        Args:
            line: Raw report line.

        Returns:
            Section text after "Parameters for", or None if not a section line.
        """
        section_match = PARAMETERS_SECTION_PATTERN.match(line.strip())
        if section_match is None:
            return None
        return section_match.group(1).strip()

    def _section_kind_from_parameters_text(self, parameters_text: str) -> str:
        """Maps parameters section text to a section kind.

        Args:
            parameters_text: Text after "Parameters for".

        Returns:
            Section kind used for in-context measurement classification.
        """
        normalized = re.sub(r"\s+", " ", parameters_text.strip().lower())
        if "continuity" in normalized:
            return SECTION_KIND_CONTINUITY
        if "lv isolation" in normalized:
            return SECTION_KIND_LV_ISOLATION
        if "hv isolation" in normalized:
            return SECTION_KIND_HV_ISOLATION
        return SECTION_KIND_OTHER

    def is_resistance_unit(self, unit_text: str) -> bool:
        """Checks whether a parsed engineering unit is a resistance unit."""
        return unit_text in RESISTANCE_UNITS

    def is_current_unit(self, unit_text: str) -> bool:
        """Checks whether a parsed engineering unit is a current unit."""
        return unit_text in CURRENT_UNITS

    def _resolve_measurement_category(
        self,
        active_section_kind: str,
        active_parameters_text: str,
        engineering_unit: str,
    ) -> str:
        """Determines the measurement category using active section context.

        Args:
            active_section_kind: Most recently seen parameters section kind.
            active_parameters_text: Most recently seen parameters section text.
            engineering_unit: Parsed engineering unit for this measurement.

        Returns:
            One of the supported measurement category values.
        """
        _ = active_parameters_text
        if active_section_kind == SECTION_KIND_CONTINUITY:
            return MEASUREMENT_CATEGORY_CONTINUITY
        if active_section_kind == SECTION_KIND_LV_ISOLATION:
            return MEASUREMENT_CATEGORY_LV_ISOLATION
        if active_section_kind == SECTION_KIND_HV_ISOLATION:
            if self.is_resistance_unit(engineering_unit):
                return MEASUREMENT_CATEGORY_HV_ISOLATION_RESISTANCE
            if self.is_current_unit(engineering_unit):
                return MEASUREMENT_CATEGORY_HV_DIELECTRIC_BREAKDOWN
            return MEASUREMENT_CATEGORY_OTHER
        return MEASUREMENT_CATEGORY_OTHER

    def _parse_scalar_value(self, raw_value_text: str) -> ParsedValue:
        """Parses one scalar engineering value such as >8.168GOhm or 6.741nF."""
        stripped_text = raw_value_text.strip()
        numeric_match = NUMERIC_VALUE_PATTERN.match(stripped_text)
        if not numeric_match:
            raise ValueError(f"Unable to parse measurement value: {raw_value_text!r}")

        censor_symbol = numeric_match.group("censor")
        number_text = numeric_match.group("number")
        unit_text = numeric_match.group("unit")
        if unit_text not in UNIT_MULTIPLIERS:
            raise ValueError(f"Unsupported engineering unit: {unit_text!r}")

        numeric_value_si = float(number_text) * UNIT_MULTIPLIERS[unit_text]
        if censor_symbol == ">":
            censor_type = "lower"
        elif censor_symbol == "<":
            censor_type = "upper"
        else:
            censor_type = None

        return ParsedValue(
            raw_text=stripped_text,
            numeric_value_si=numeric_value_si,
            display_unit=unit_text,
            censor_type=censor_type,
        )


class BaselineBuilder:
    """Builds and saves baseline databases from parsed measurement records."""

    def build_from_reports(self, report_paths: Iterable[Path]) -> BaselineDatabase:
        """Builds a baseline database from many report files.

        Args:
            report_paths: Iterable of report paths.

        Returns:
            Baseline database grouped by unit under test.
        """
        parser = W434ReportParser()
        unit_records: dict[str, list[MeasurementRecord]] = {}
        unit_report_counts: dict[str, int] = {}
        unit_passed_report_counts: dict[str, int] = {}

        for report_path in tqdm(list(report_paths), desc="Processing baseline files", unit="file"):
            try:
                metadata, measurement_records = parser.parse_report(report_path)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"WARNING: Failed to parse {report_path}: {exc}")
                continue

            unit_under_test = metadata.get("unit_under_test", "").strip()
            overall_result = metadata.get("overall_result", "").strip().upper()
            if not unit_under_test:
                print(f"WARNING: Skipping {report_path} because Unit under test is missing.")
                continue

            unit_report_counts[unit_under_test] = unit_report_counts.get(unit_under_test, 0) + 1
            if overall_result != "PASSED":
                continue

            unit_passed_report_counts[unit_under_test] = (
                unit_passed_report_counts.get(unit_under_test, 0) + 1
            )
            unit_records.setdefault(unit_under_test, []).extend(measurement_records)

        units: dict[str, UnitBaseline] = {}
        for unit_under_test, records in unit_records.items():
            measurement_baselines = self._build_measurement_baselines(records)
            units[unit_under_test] = UnitBaseline(
                unit_under_test=unit_under_test,
                report_count=unit_report_counts.get(unit_under_test, 0),
                passed_report_count=unit_passed_report_counts.get(unit_under_test, 0),
                measurement_baselines=measurement_baselines,
            )

        return BaselineDatabase(
            created_by_program="w434_statistical_baseline_tool.py",
            units=units,
        )

    def save_to_json(self, baseline_database: BaselineDatabase, output_json_path: Path) -> None:
        """Saves a baseline database to JSON."""
        serializable = {
            "created_by_program": baseline_database.created_by_program,
            "units": {
                unit_name: {
                    "unit_under_test": unit_baseline.unit_under_test,
                    "report_count": unit_baseline.report_count,
                    "passed_report_count": unit_baseline.passed_report_count,
                    "measurement_baselines": {
                        measurement_key: asdict(measurement_baseline)
                        for measurement_key, measurement_baseline in unit_baseline.measurement_baselines.items()
                    },
                }
                for unit_name, unit_baseline in baseline_database.units.items()
            },
        }
        output_json_path.write_text(
            json.dumps(serializable, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _build_measurement_baselines(
        self,
        records: list[MeasurementRecord],
    ) -> dict[str, MeasurementBaseline]:
        """Builds baselines for each measurement key from normalized records."""
        grouped_records: dict[str, list[MeasurementRecord]] = {}
        for record in records:
            grouped_records.setdefault(record.measurement_key, []).append(record)

        measurement_baselines: dict[str, MeasurementBaseline] = {}
        for measurement_key, measurement_records in grouped_records.items():
            prototype_record = measurement_records[0]
            uncensored_values = [
                record.numeric_value_si
                for record in measurement_records
                if record.censor_type is None
            ]
            lower_censored_values = [
                record.numeric_value_si
                for record in measurement_records
                if record.censor_type == "lower"
            ]
            upper_censored_values = [
                record.numeric_value_si
                for record in measurement_records
                if record.censor_type == "upper"
            ]
            all_values = [record.numeric_value_si for record in measurement_records]

            mean_value_si: Optional[float] = None
            median_value_si: Optional[float] = None
            population_stddev_si: Optional[float] = None
            if uncensored_values:
                mean_value_si = statistics.fmean(uncensored_values)
                median_value_si = statistics.median(uncensored_values)
                if len(uncensored_values) >= 2:
                    population_stddev_si = statistics.pstdev(uncensored_values)
                else:
                    population_stddev_si = 0.0

            baseline_floor_value_si = max(lower_censored_values) if lower_censored_values else None
            baseline_ceiling_value_si = min(upper_censored_values) if upper_censored_values else None

            measurement_baselines[measurement_key] = MeasurementBaseline(
                measurement_key=measurement_key,
                test_name=prototype_record.test_name,
                from_node=prototype_record.from_node,
                to_node=prototype_record.to_node,
                metric_name=prototype_record.metric_name,
                engineering_unit=prototype_record.engineering_unit,
                measurement_category=prototype_record.measurement_category,
                sample_count_total=len(measurement_records),
                sample_count_uncensored=len(uncensored_values),
                sample_count_lower_censored=len(lower_censored_values),
                sample_count_upper_censored=len(upper_censored_values),
                minimum_value_si=min(all_values) if all_values else None,
                maximum_value_si=max(all_values) if all_values else None,
                mean_value_si=mean_value_si,
                median_value_si=median_value_si,
                population_stddev_si=population_stddev_si,
                baseline_floor_value_si=baseline_floor_value_si,
                baseline_ceiling_value_si=baseline_ceiling_value_si,
            )

        return measurement_baselines


class BaselineLoader:
    """Loads a baseline database from JSON."""

    def load_from_json(self, baseline_json_path: Path) -> BaselineDatabase:
        """Loads baseline JSON from disk.

        Args:
            baseline_json_path: Saved baseline JSON path.

        Returns:
            In-memory baseline database.
        """
        payload = json.loads(baseline_json_path.read_text(encoding="utf-8"))
        units: dict[str, UnitBaseline] = {}
        for unit_name, unit_payload in payload["units"].items():
            measurement_baselines = {
                measurement_key: MeasurementBaseline(
                    **{
                        **measurement_payload,
                        "measurement_category": measurement_payload.get(
                            "measurement_category",
                            MEASUREMENT_CATEGORY_OTHER,
                        ),
                    }
                )
                for measurement_key, measurement_payload in unit_payload["measurement_baselines"].items()
            }
            units[unit_name] = UnitBaseline(
                unit_under_test=unit_payload["unit_under_test"],
                report_count=unit_payload["report_count"],
                passed_report_count=unit_payload["passed_report_count"],
                measurement_baselines=measurement_baselines,
            )
        return BaselineDatabase(
            created_by_program=payload["created_by_program"],
            units=units,
        )


class ReportEvaluator:
    """Evaluates a new report against a saved statistical baseline."""

    def evaluate_report(
        self,
        baseline_database: BaselineDatabase,
        report_path: Path,
        z_threshold: float,
    ) -> dict[str, Any]:
        """Evaluates one report against the baseline database.

        Args:
            baseline_database: Loaded baseline database.
            report_path: New report path to evaluate.
            z_threshold: Z-score threshold used for numeric flags.

        Returns:
            Dictionary containing metadata, per-measurement findings, and a
            high-level summary.
        """
        parser = W434ReportParser()
        metadata, measurement_records = parser.parse_report(report_path)
        unit_under_test = metadata.get("unit_under_test", "").strip()
        overall_result = metadata.get("overall_result", "").strip().upper()
        if not unit_under_test:
            raise ValueError("New report is missing 'Unit under test'.")
        if unit_under_test not in baseline_database.units:
            raise ValueError(f"No saved baseline exists for unit under test: {unit_under_test!r}")

        unit_baseline = baseline_database.units[unit_under_test]
        findings: list[dict[str, Any]] = []
        matched_count = 0
        flagged_count = 0
        missing_from_baseline_count = 0
        missing_from_report_count = 0

        report_measurement_map = {record.measurement_key: record for record in measurement_records}

        for measurement_key, measurement_baseline in unit_baseline.measurement_baselines.items():
            report_record = report_measurement_map.get(measurement_key)
            if report_record is None:
                findings.append({
                    "measurement_key": measurement_key,
                    "status": "MISSING_FROM_REPORT",
                    "message": "Baseline expects this measurement but it was not found in the new report.",
                })
                missing_from_report_count += 1
                flagged_count += 1
                continue

            matched_count += 1
            finding = self._evaluate_one_measurement(
                measurement_baseline=measurement_baseline,
                report_record=report_record,
                z_threshold=z_threshold,
            )
            findings.append(finding)
            if finding["status"] != "OK":
                flagged_count += 1

        for measurement_key, report_record in report_measurement_map.items():
            if measurement_key not in unit_baseline.measurement_baselines:
                findings.append({
                    "measurement_key": measurement_key,
                    "status": "NOT_IN_BASELINE",
                    "message": "Measurement exists in new report but not in baseline.",
                    "test_name": report_record.test_name,
                    "from_node": report_record.from_node,
                    "to_node": report_record.to_node,
                    "metric_name": report_record.metric_name,
                })
                missing_from_baseline_count += 1
                flagged_count += 1

        summary_status = "PASS_STATISTICALLY_CONSISTENT"
        if overall_result != "PASSED":
            summary_status = "REPORT_FAILED"
        elif flagged_count > 0:
            summary_status = "PASS_BUT_REVIEW_REQUIRED"

        return {
            "report_path": str(report_path),
            "serial_number": metadata.get("serial_number", "").strip(),
            "unit_under_test": unit_under_test,
            "report_overall_result": overall_result,
            "baseline_report_count": unit_baseline.report_count,
            "baseline_passed_report_count": unit_baseline.passed_report_count,
            "matched_measurements": matched_count,
            "missing_from_report_count": missing_from_report_count,
            "not_in_baseline_count": missing_from_baseline_count,
            "flagged_measurements": flagged_count,
            "summary_status": summary_status,
            "findings": findings,
        }

    def _evaluate_one_measurement(
        self,
        measurement_baseline: MeasurementBaseline,
        report_record: MeasurementRecord,
        z_threshold: float,
    ) -> dict[str, Any]:
        """Evaluates one measurement against its baseline."""
        finding: dict[str, Any] = {
            "measurement_key": measurement_baseline.measurement_key,
            "test_name": measurement_baseline.test_name,
            "from_node": measurement_baseline.from_node,
            "to_node": measurement_baseline.to_node,
            "metric_name": measurement_baseline.metric_name,
            "status": "OK",
            "message": "Measurement is consistent with baseline.",
            "report_measurement_text": report_record.measurement_text,
            "report_numeric_value_si": report_record.numeric_value_si,
            "report_censor_type": report_record.censor_type,
            "engineering_unit": measurement_baseline.engineering_unit,
            "baseline_sample_count_total": measurement_baseline.sample_count_total,
            "baseline_sample_count_uncensored": measurement_baseline.sample_count_uncensored,
            "baseline_mean_value_si": measurement_baseline.mean_value_si,
            "baseline_median_value_si": measurement_baseline.median_value_si,
            "baseline_population_stddev_si": measurement_baseline.population_stddev_si,
            "baseline_minimum_value_si": measurement_baseline.minimum_value_si,
            "baseline_maximum_value_si": measurement_baseline.maximum_value_si,
            "baseline_floor_value_si": measurement_baseline.baseline_floor_value_si,
            "baseline_ceiling_value_si": measurement_baseline.baseline_ceiling_value_si,
        }

        if report_record.censor_type == "lower":
            return finding

        if report_record.censor_type == "upper":
            return finding

        if (
            measurement_baseline.sample_count_lower_censored > 0
            and report_record.censor_type is None
        ):
            finding["status"] = "BECAME_MEASURABLE"
            finding["message"] = (
                "Baseline behavior was >X censored, but the new measurement became directly measurable."
            )
            return finding

        if (
            measurement_baseline.sample_count_upper_censored > 0
            and report_record.censor_type is None
        ):
            finding["status"] = "EXCEEDED_CENSORED_REGION"
            finding["message"] = (
                "Baseline behavior was <X censored, but the new measurement became directly measurable."
            )
            return finding

        if measurement_baseline.sample_count_uncensored >= 2 and measurement_baseline.mean_value_si is not None:
            baseline_stddev = measurement_baseline.population_stddev_si
            if baseline_stddev is None:
                finding["status"] = "INSUFFICIENT_BASELINE"
                finding["message"] = "Baseline standard deviation is unavailable."
                return finding

            if math.isclose(baseline_stddev, 0.0, abs_tol=1e-18):
                if not math.isclose(
                    report_record.numeric_value_si,
                    measurement_baseline.mean_value_si,
                    rel_tol=1e-9,
                    abs_tol=1e-18,
                ):
                    finding["status"] = "ZERO_SPREAD_MISMATCH"
                    finding["message"] = (
                        "Baseline has zero spread but the new measurement differs from the baseline mean."
                    )
                return finding

            z_score = (
                report_record.numeric_value_si - measurement_baseline.mean_value_si
            ) / baseline_stddev
            finding["z_score"] = z_score
            if abs(z_score) > z_threshold:
                finding["status"] = "Z_SCORE_OUTLIER"
                finding["message"] = (
                    f"Measurement differs from baseline by {abs(z_score):.3f} standard deviations."
                )
            return finding

        if measurement_baseline.sample_count_uncensored == 1 and measurement_baseline.mean_value_si is not None:
            baseline_mean = measurement_baseline.mean_value_si
            if not math.isclose(
                report_record.numeric_value_si,
                baseline_mean,
                rel_tol=1e-9,
                abs_tol=1e-18,
            ):
                finding["status"] = "SINGLE_SAMPLE_DIFFERENCE"
                finding["message"] = (
                    "Baseline only has one uncensored sample and the new value differs from it."
                )
            return finding

        baseline_floor = measurement_baseline.baseline_floor_value_si
        if baseline_floor is not None and report_record.numeric_value_si < baseline_floor:
            finding["status"] = "DROPPED_BELOW_BASELINE_FLOOR"
            finding["message"] = (
                "New uncensored value fell below the lower censored floor seen in the baseline."
            )
            return finding

        baseline_ceiling = measurement_baseline.baseline_ceiling_value_si
        if baseline_ceiling is not None and report_record.numeric_value_si > baseline_ceiling:
            finding["status"] = "ROSE_ABOVE_BASELINE_CEILING"
            finding["message"] = (
                "New uncensored value rose above the upper censored ceiling seen in the baseline."
            )
            return finding

        finding["status"] = "INSUFFICIENT_BASELINE"
        finding["message"] = (
            "Baseline did not contain enough uncensored data for a statistical comparison, but no floor/ceiling regression was detected."
        )
        return finding



def flatten_input_directories(input_directories_argument: list[list[str]] | list[str]) -> list[Path]:
    """Flattens one or more --input-dir argument groups into Path objects.

    Args:
        input_directories_argument: Parsed argparse values for input directories.

    Returns:
        Flat list of Path objects.
    """
    flattened_directories: list[Path] = []
    for entry in input_directories_argument:
        if isinstance(entry, list):
            flattened_directories.extend(Path(directory) for directory in entry)
        else:
            flattened_directories.append(Path(entry))
    return flattened_directories



def collect_report_paths(
    input_directories: list[Path],
    recursive: bool,
    search_label: str = "Searching files",
) -> list[Path]:
    """Collects candidate report files from one or more directories.

    Args:
        input_directories: Root directories containing report files.
        recursive: Whether to recurse into subdirectories.
        search_label: Progress label for file discovery.

    Returns:
        Sorted list of candidate text-like file paths.
    """
    candidate_paths: list[Path] = []
    for input_directory in tqdm(input_directories, desc=search_label, unit="dir"):
        if recursive:
            candidate_paths.extend(
                path for path in input_directory.rglob("*") if path.is_file()
            )
        else:
            candidate_paths.extend(
                path for path in input_directory.iterdir() if path.is_file()
            )

    text_suffixes = {".txt", ".log", ".rpt", ".out", ""}
    filtered_paths: list[Path] = []
    for path in tqdm(candidate_paths, desc="Filtering candidate files", unit="file"):
        if path.suffix.lower() in text_suffixes:
            filtered_paths.append(path)
    return sorted(filtered_paths)



def print_build_summary(baseline_database: BaselineDatabase) -> None:
    """Prints a short build summary to stdout."""
    print("Baseline build complete.")
    print(f"Units in baseline: {len(baseline_database.units)}")
    for unit_name, unit_baseline in sorted(baseline_database.units.items()):
        print(
            f"  {unit_name}: reports={unit_baseline.report_count}, "
            f"passed_reports={unit_baseline.passed_report_count}, "
            f"measurements={len(unit_baseline.measurement_baselines)}"
        )
        grouped_counts: dict[str, int] = {}
        for measurement_baseline in unit_baseline.measurement_baselines.values():
            display_group = classify_test_group(measurement_baseline.measurement_category)
            grouped_counts[display_group] = grouped_counts.get(display_group, 0) + 1
        for display_group, group_count in sorted(grouped_counts.items()):
            print(f"    - {display_group}: {group_count}")


def classify_test_group(measurement_category: str) -> str:
    """Maps measurement category values to user-facing baseline group labels."""
    label_map = {
        MEASUREMENT_CATEGORY_CONTINUITY: "Continuity",
        MEASUREMENT_CATEGORY_LV_ISOLATION: "Low Voltage Isolation",
        MEASUREMENT_CATEGORY_HV_ISOLATION_RESISTANCE: "High Voltage Isolation",
        MEASUREMENT_CATEGORY_HV_DIELECTRIC_BREAKDOWN: "Dielectric Breakdown",
        MEASUREMENT_CATEGORY_OTHER: "Other",
    }
    return label_map.get(measurement_category, "Other")


def format_plain_table(headers: list[str], rows: list[list[str]]) -> str:
    """Formats rows as a simple left/right aligned plain-text table."""
    all_rows = [headers] + rows
    column_widths = [
        max(len(str(row[column_index])) for row in all_rows)
        for column_index in range(len(headers))
    ]
    right_aligned_headers = {"Reports", "Passed", "Measurements", "Samples"}

    def format_row(row: list[str]) -> str:
        rendered_cells: list[str] = []
        for column_index, cell in enumerate(row):
            width = column_widths[column_index]
            header_name = headers[column_index]
            if header_name in right_aligned_headers:
                rendered_cells.append(str(cell).rjust(width))
            else:
                rendered_cells.append(str(cell).ljust(width))
        return "   ".join(rendered_cells)

    header_line = format_row(headers)
    separator_line = "-" * len(header_line)
    body_lines = [format_row(row) for row in rows]
    return "\n".join([header_line, separator_line] + body_lines)


def print_assembly_list_table(baseline_database: BaselineDatabase) -> None:
    """Prints a compact summary table of all assemblies in the baseline."""
    print("Assemblies in baseline:")
    print("")
    headers = ["Unit Under Test", "Reports", "Passed", "Measurements"]
    rows: list[list[str]] = []
    for unit_name, unit_baseline in sorted(baseline_database.units.items()):
        rows.append(
            [
                unit_name,
                str(unit_baseline.report_count),
                str(unit_baseline.passed_report_count),
                str(len(unit_baseline.measurement_baselines)),
            ]
        )

    if not rows:
        print("No assemblies found in baseline.")
        return

    print(format_plain_table(headers=headers, rows=rows))


def iter_grouped_measurement_baselines(
    unit_baseline: UnitBaseline,
) -> Iterable[tuple[str, list[MeasurementBaseline]]]:
    """Yields populated baseline display groups in the required display order."""
    display_order = [
        "Continuity",
        "Low Voltage Isolation",
        "High Voltage Isolation",
        "Dielectric Breakdown",
        "Other",
    ]
    grouped_measurements: dict[str, list[MeasurementBaseline]] = {
        display_name: [] for display_name in display_order
    }
    for measurement_baseline in unit_baseline.measurement_baselines.values():
        display_group = classify_test_group(measurement_baseline.measurement_category)
        grouped_measurements.setdefault(display_group, []).append(measurement_baseline)

    for display_group in display_order:
        records = grouped_measurements.get(display_group, [])
        if not records:
            continue
        sorted_records = sorted(
            records,
            key=lambda baseline: (
                baseline.test_name,
                baseline.from_node,
                baseline.to_node,
                baseline.metric_name,
            ),
        )
        yield display_group, sorted_records


def print_unit_baseline_table(unit_baseline: UnitBaseline) -> None:
    """Prints detailed grouped baseline rows for one unit under test."""
    print(f"Unit under test: {unit_baseline.unit_under_test}")
    print(f"Reports: {unit_baseline.report_count}")
    print(f"Passed reports: {unit_baseline.passed_report_count}")
    print(f"Measurement baselines: {len(unit_baseline.measurement_baselines)}")

    headers = [
        "Test Name",
        "From",
        "To",
        "Metric",
        "Unit",
        "Samples",
        "Mean",
        "Median",
        "Std Dev",
        "Min",
        "Max",
        "Floor",
        "Ceiling",
    ]
    for display_group, records in iter_grouped_measurement_baselines(unit_baseline):
        print("")
        print(f"=== {display_group} ===")
        rows: list[list[str]] = []
        for measurement_baseline in records:
            engineering_unit = measurement_baseline.engineering_unit
            rows.append(
                [
                    measurement_baseline.test_name,
                    measurement_baseline.from_node,
                    measurement_baseline.to_node,
                    measurement_baseline.metric_name,
                    engineering_unit or "N/A",
                    str(measurement_baseline.sample_count_total),
                    format_si_value(measurement_baseline.mean_value_si, engineering_unit),
                    format_si_value(measurement_baseline.median_value_si, engineering_unit),
                    format_si_value(measurement_baseline.population_stddev_si, engineering_unit),
                    format_si_value(measurement_baseline.minimum_value_si, engineering_unit),
                    format_si_value(measurement_baseline.maximum_value_si, engineering_unit),
                    format_si_value(measurement_baseline.baseline_floor_value_si, engineering_unit),
                    format_si_value(measurement_baseline.baseline_ceiling_value_si, engineering_unit),
                ]
            )
        print(format_plain_table(headers=headers, rows=rows))


def baseline_info_command(arguments: argparse.Namespace) -> int:
    """Prints baseline information for all units or one selected unit."""
    baseline_json_path = Path(arguments.baseline_json)
    loader = BaselineLoader()
    baseline_database = loader.load_from_json(baseline_json_path)

    if not arguments.unit_under_test:
        print_assembly_list_table(baseline_database)
        return 0

    selected_unit = arguments.unit_under_test.strip()
    unit_baseline = baseline_database.units.get(selected_unit)
    if unit_baseline is None:
        print(f"ERROR: Unit under test not found in baseline: {selected_unit!r}")
        return 1

    print_unit_baseline_table(unit_baseline)
    return 0



def format_si_value(value_si: Optional[float], engineering_unit: Optional[str]) -> str:
    """Formats an SI-base numeric value back into the stored engineering unit.

    Args:
        value_si: Numeric value in SI base units.
        engineering_unit: Engineering unit string such as mOhm or GOhm.

    Returns:
        Human-readable formatted value string.
    """
    if value_si is None:
        return "N/A"
    if not engineering_unit:
        return f"{value_si:.6g}"

    multiplier = UNIT_MULTIPLIERS.get(engineering_unit)
    if multiplier is None or math.isclose(multiplier, 0.0):
        return f"{value_si:.6g}"

    engineering_value = value_si / multiplier
    return f"{engineering_value:.6g}{engineering_unit}"



def print_evaluation_summary(evaluation_result: dict[str, Any]) -> None:
    """Prints a readable evaluation summary to stdout."""
    print(f"Report: {evaluation_result['report_path']}")
    print(f"Unit under test: {evaluation_result['unit_under_test']}")
    print(f"Overall report result: {evaluation_result['report_overall_result']}")
    print(f"Baseline passed reports: {evaluation_result['baseline_passed_report_count']}")
    print(f"Summary status: {evaluation_result['summary_status']}")
    print(f"Matched measurements: {evaluation_result['matched_measurements']}")
    print(f"Flagged measurements: {evaluation_result['flagged_measurements']}")
    print(f"Missing from report: {evaluation_result['missing_from_report_count']}")
    print(f"Not in baseline: {evaluation_result['not_in_baseline_count']}")

    flagged_findings = [
        finding for finding in evaluation_result["findings"] if finding["status"] != "OK"
    ]
    if flagged_findings:
        print("Flagged findings:")
        for finding in flagged_findings:
            measurement_key = finding.get("measurement_key", "")
            status = finding.get("status", "")
            message = finding.get("message", "")
            print(f"  [{status}] {measurement_key} - {message}")

            report_measurement_text = finding.get("report_measurement_text")
            engineering_unit = finding.get("engineering_unit")
            baseline_mean_value_si = finding.get("baseline_mean_value_si")
            baseline_population_stddev_si = finding.get("baseline_population_stddev_si")
            baseline_minimum_value_si = finding.get("baseline_minimum_value_si")
            baseline_maximum_value_si = finding.get("baseline_maximum_value_si")
            baseline_sample_count_total = finding.get("baseline_sample_count_total")
            baseline_sample_count_uncensored = finding.get("baseline_sample_count_uncensored")
            z_score = finding.get("z_score")

            if report_measurement_text is not None:
                print(f"      current: {report_measurement_text}")
            print(
                f"      baseline samples: total={baseline_sample_count_total}, "
                f"uncensored={baseline_sample_count_uncensored}"
            )

            if baseline_mean_value_si is not None:
                print(
                    "      baseline mean: "
                    f"{format_si_value(baseline_mean_value_si, engineering_unit)}"
                )
            if baseline_population_stddev_si is not None:
                print(
                    "      baseline std dev: "
                    f"{format_si_value(baseline_population_stddev_si, engineering_unit)}"
                )
            if baseline_minimum_value_si is not None or baseline_maximum_value_si is not None:
                minimum_text = (
                    format_si_value(baseline_minimum_value_si, engineering_unit)
                    if baseline_minimum_value_si is not None else "N/A"
                )
                maximum_text = (
                    format_si_value(baseline_maximum_value_si, engineering_unit)
                    if baseline_maximum_value_si is not None else "N/A"
                )
                print(f"      baseline range: {minimum_text} to {maximum_text}")
            if z_score is not None:
                print(f"      z-score: {z_score:.3f}")
    else:
        print("No flagged findings.")



def build_command(arguments: argparse.Namespace) -> int:
    """Runs the baseline build command.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        Process exit code.
    """
    input_directories = flatten_input_directories(arguments.input_dirs)
    output_json_path = Path(arguments.output_json)
    report_paths = collect_report_paths(
        input_directories=input_directories,
        recursive=arguments.recursive,
        search_label="Searching baseline directories",
    )
    if not report_paths:
        print("ERROR: No candidate report files were found.")
        return 1

    builder = BaselineBuilder()
    baseline_database = builder.build_from_reports(report_paths)
    builder.save_to_json(baseline_database, output_json_path)
    print_build_summary(baseline_database)
    print(f"Saved baseline JSON: {output_json_path}")
    return 0



def evaluate_command(arguments: argparse.Namespace) -> int:
    """Runs the report evaluation command.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        Process exit code.
    """
    baseline_json_path = Path(arguments.baseline_json)
    report_path = Path(arguments.report)
    loader = BaselineLoader()
    evaluator = ReportEvaluator()
    baseline_database = loader.load_from_json(baseline_json_path)
    evaluation_result = evaluator.evaluate_report(
        baseline_database=baseline_database,
        report_path=report_path,
        z_threshold=arguments.z_threshold,
    )
    print_evaluation_summary(evaluation_result)

    if arguments.output_json:
        output_json_path = Path(arguments.output_json)
        output_json_path.write_text(
            json.dumps(evaluation_result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Saved evaluation JSON: {output_json_path}")
    return 0



def print_batch_evaluation_summary(batch_results: list[dict[str, Any]]) -> None:
    """Prints a readable summary for a batch evaluation run.

    Args:
        batch_results: Per-report evaluation results.
    """
    total_reports = len(batch_results)
    passed_consistent_count = sum(
        1 for result in batch_results
        if result["summary_status"] == "PASS_STATISTICALLY_CONSISTENT"
    )
    review_required_count = sum(
        1 for result in batch_results
        if result["summary_status"] == "PASS_BUT_REVIEW_REQUIRED"
    )
    failed_count = sum(
        1 for result in batch_results
        if result["summary_status"] == "REPORT_FAILED"
    )

    print(f"Batch reports evaluated: {total_reports}")
    print(f"Statistically consistent: {passed_consistent_count}")
    print(f"Pass but review required: {review_required_count}")
    print(f"Report failed: {failed_count}")
    print("")

    for result in batch_results:
        report_name = Path(result["report_path"]).name
        serial_number = result.get("serial_number", "")
        serial_text = f", serial={serial_number}" if serial_number else ""
        print(
            f"{report_name}: {result['summary_status']}"
            f" (flagged={result['flagged_measurements']}{serial_text})"
        )

        flagged_findings = [
            finding for finding in result["findings"] if finding["status"] != "OK"
        ]
        for finding in flagged_findings:
            measurement_key = finding.get("measurement_key", "")
            status = finding.get("status", "")
            report_measurement_text = finding.get("report_measurement_text", "")
            z_score = finding.get("z_score")
            if z_score is not None:
                print(
                    f"  [{status}] {measurement_key} | current={report_measurement_text} "
                    f"| z={z_score:.3f}"
                )
            else:
                print(
                    f"  [{status}] {measurement_key} | current={report_measurement_text}"
                )
        if flagged_findings:
            print("")



def evaluate_directory_command(arguments: argparse.Namespace) -> int:
    """Runs batch evaluation for all candidate reports in a directory.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        Process exit code.
    """
    baseline_json_path = Path(arguments.baseline_json)
    input_directories = flatten_input_directories(arguments.input_dirs)
    loader = BaselineLoader()
    evaluator = ReportEvaluator()
    baseline_database = loader.load_from_json(baseline_json_path)

    report_paths = collect_report_paths(
        input_directories=input_directories,
        recursive=arguments.recursive,
        search_label="Searching evaluation directories",
    )
    if not report_paths:
        print("ERROR: No candidate report files were found.")
        return 1

    batch_results: list[dict[str, Any]] = []
    for report_path in tqdm(report_paths, desc="Evaluating batch reports", unit="file"):
        try:
            evaluation_result = evaluator.evaluate_report(
                baseline_database=baseline_database,
                report_path=report_path,
                z_threshold=arguments.z_threshold,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"WARNING: Failed to evaluate {report_path}: {exc}")
            continue

        batch_results.append(evaluation_result)

    if not batch_results:
        print("ERROR: No reports were successfully evaluated.")
        return 1

    if arguments.only_review_required:
        filtered_results = [
            result for result in batch_results
            if result["summary_status"] != "PASS_STATISTICALLY_CONSISTENT"
        ]
        print_batch_evaluation_summary(filtered_results)
    else:
        print_batch_evaluation_summary(batch_results)

    if arguments.output_json:
        output_json_path = Path(arguments.output_json)
        output_json_path.write_text(
            json.dumps(batch_results, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Saved batch evaluation JSON: {output_json_path}")

    if arguments.output_csv:
        output_csv_path = Path(arguments.output_csv)
        csv_lines = [
            "report_path,unit_under_test,serial_number,report_overall_result,summary_status,flagged_measurements,matched_measurements,missing_from_report_count,not_in_baseline_count"
        ]
        for result in batch_results:
            csv_fields = [
                result.get("report_path", ""),
                result.get("unit_under_test", ""),
                result.get("serial_number", ""),
                result.get("report_overall_result", ""),
                result.get("summary_status", ""),
                str(result.get("flagged_measurements", "")),
                str(result.get("matched_measurements", "")),
                str(result.get("missing_from_report_count", "")),
                str(result.get("not_in_baseline_count", "")),
            ]
            escaped_fields = [
                '"' + field.replace('"', '""') + '"' for field in csv_fields
            ]
            csv_lines.append(",".join(escaped_fields))
        output_csv_path.write_text("".join(csv_lines) + "", encoding="utf-8")
        print(f"Saved batch evaluation CSV: {output_csv_path}")

    return 0



def create_argument_parser() -> argparse.ArgumentParser:
    """Creates the command-line argument parser.

    Returns:
        Configured argparse.ArgumentParser instance.
    """
    argument_parser = argparse.ArgumentParser(
        description=(
            "Build and use statistical baselines from W434 text report files."
        ),
        epilog=(
            "Examples:\n"
            "  Build a baseline from all reports under a directory tree:\n"
            "    python w434_statistical_baseline_tool.py build --input-dir ./reports_a ./reports_b --output-json ./baseline.json\n\n"
            "  Evaluate one new report against that baseline:\n"
            "    python w434_statistical_baseline_tool.py evaluate --baseline-json ./baseline.json --report ./new_report.txt\n\n"
            "Notes:\n"
            "  - Only reports with overall result PASSED are used to build the baseline.\n"
            "  - Baselines are grouped by the 'Unit under test' field in the report.\n"
            "  - Censored values such as >8.168GOhm and <26.98uA are tracked separately."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = argument_parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="Parse a directory of reports and build a baseline JSON file.",
    )
    build_parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dirs",
        action="append",
        nargs="+",
        required=True,
        metavar="DIR",
        help=(
            "One or more directories containing W434 text report files. "
            "This option may be repeated."
        ),
    )
    build_parser.add_argument(
        "-o",
        "--output-json",
        required=True,
        help="Path to write the baseline JSON file.",
    )
    build_parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories for report files.",
    )
    build_parser.set_defaults(func=build_command)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate one report against a saved baseline JSON file.",
    )
    evaluate_parser.add_argument(
        "-b",
        "--baseline-json",
        required=True,
        help="Path to the saved baseline JSON file.",
    )
    evaluate_parser.add_argument(
        "-p",
        "--report",
        required=True,
        help="Path to the new report file to evaluate.",
    )
    evaluate_parser.add_argument(
        "-z",
        "--z-threshold",
        type=float,
        default=3.0,
        help="Absolute Z-score threshold used to flag numeric outliers. Default: 3.0",
    )
    evaluate_parser.add_argument(
        "-o",
        "--output-json",
        help="Optional path to save the evaluation result as JSON.",
    )
    evaluate_parser.set_defaults(func=evaluate_command)

    evaluate_dir_parser = subparsers.add_parser(
        "evaluate-dir",
        help="Evaluate all candidate reports in a directory against a saved baseline JSON file.",
    )
    evaluate_dir_parser.add_argument(
        "-b",
        "--baseline-json",
        required=True,
        help="Path to the saved baseline JSON file.",
    )
    evaluate_dir_parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dirs",
        action="append",
        nargs="+",
        required=True,
        metavar="DIR",
        help=(
            "One or more directories containing report files to evaluate. "
            "This option may be repeated."
        ),
    )
    evaluate_dir_parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories for report files.",
    )
    evaluate_dir_parser.add_argument(
        "-z",
        "--z-threshold",
        type=float,
        default=3.0,
        help="Absolute Z-score threshold used to flag numeric outliers. Default: 3.0",
    )
    evaluate_dir_parser.add_argument(
        "--only-review-required",
        action="store_true",
        help="Only print reports that are not statistically consistent.",
    )
    evaluate_dir_parser.add_argument(
        "-o",
        "--output-json",
        help="Optional path to save batch evaluation results as JSON.",
    )
    evaluate_dir_parser.add_argument(
        "-c",
        "--output-csv",
        help="Optional path to save a one-line-per-report CSV summary.",
    )
    evaluate_dir_parser.set_defaults(func=evaluate_directory_command)

    baseline_info_parser = subparsers.add_parser(
        "baseline-info",
        help="Print summary information from a saved baseline JSON file.",
    )
    baseline_info_parser.add_argument(
        "-b",
        "--baseline-json",
        required=True,
        help="Path to the saved baseline JSON file.",
    )
    baseline_info_parser.add_argument(
        "-u",
        "--unit-under-test",
        help="Optional unit under test to print detailed grouped baseline rows.",
    )
    baseline_info_parser.set_defaults(func=baseline_info_command)

    return argument_parser



def main() -> int:
    """Program entry point.

    Returns:
        Process exit code.
    """
    argument_parser = create_argument_parser()
    arguments = argument_parser.parse_args()
    return arguments.func(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
