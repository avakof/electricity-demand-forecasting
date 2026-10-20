"""Validate executed notebooks/artifact invariants and render portable previews."""
import base64
import gzip
import json
import sys
from pathlib import Path

import nbformat
import pandas as pd
from nbconvert import HTMLExporter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.common import fingerprint, local_day_grid, write_json


def main():
    previews=ROOT/"reports/notebook_previews"
    figures=ROOT/"reports/figures"
    previews.mkdir(parents=True,exist_ok=True)
    figures.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in sorted((ROOT/"notebooks").glob("0[1-6]_*.ipynb")):
        notebook=nbformat.read(path,as_version=4)
        nbformat.validate(notebook)
        count=0
        for i,cell in enumerate(notebook.cells):
            if cell.cell_type!="code": continue
            assert cell.execution_count is not None,f"Unexecuted cell: {path.name}:{i}"
            for output in cell.get("outputs",[]):
                assert output.output_type!="error",f"Cell error: {path.name}:{i}"
                if "image/png" in output.get("data",{}):
                    png=output.data["image/png"]
                    (figures/f"{path.stem}_cell_{i}.png").write_bytes(base64.b64decode(png))
                    count+=1
        html,_=HTMLExporter(template_name="lab").from_notebook_node(notebook)
        (previews/f"{path.stem}.html").write_text(html)
        rows.append({"notebook":path.name,"code_cells":sum(c.cell_type=="code" for c in notebook.cells),"figures":count,"executed_without_errors":True})
    # Validate that original CSVs are still byte-for-byte identical.
    manifest=json.loads((ROOT/"data/raw/source_manifest.json").read_text())
    for name,source in manifest["sources"].items():
        assert fingerprint(ROOT/"data"/name)==source["sha256"],f"Raw source changed: {name}"
    original=pd.read_csv(ROOT/"data/france_electricity_hourly.csv")
    audited=pd.read_csv(ROOT/"data/processed/audited_country.csv.gz")
    for column in ("load_mw","rte_forecast_d1_mw","temperature_c"):
        pd.testing.assert_series_equal(original[column],audited[column],check_names=True)
    for path in (ROOT/"reports/forecasts").glob("*.csv.gz"):
        frame=pd.read_csv(path,low_memory=False)
        assert "actual_mw" not in frame,f"Outcomes in forecast archive: {path.name}"
        if {"model","origin","delivery_time"}.issubset(frame):
            assert not frame.duplicated(["model","origin","delivery_time"]).any(),path.name
    origin_ledger_checks=0
    for path in (ROOT/"reports/forecasts").glob("*.manifest.json"):
        meta=json.loads(path.read_text())
        if "immutable_archive" in meta:
            immutable=ROOT/meta["immutable_archive"]
            assert fingerprint(immutable)==meta["forecast_sha256"],path.name
        if "origin_ledger" in meta:
            ledger=ROOT/meta["origin_ledger"]
            assert fingerprint(ledger)==meta["origin_ledger_sha256"],path.name
            total=0; previous=None
            with gzip.open(ledger,"rt") as stream:
                for line in stream:
                    event=json.loads(line)
                    origin=pd.Timestamp(event["origin"])
                    assert previous is None or origin>previous
                    assert all("actual_mw" not in row for row in event["predictions"])
                    total+=len(event["predictions"]); previous=origin
            assert total==meta["rows"],path.name
            origin_ledger_checks+=1
    for path in (ROOT/"reports/models").glob("*_fits.csv"):
        fits=pd.read_csv(path)
        if "latest_label_available_at" in fits:
            available=pd.to_datetime(fits.latest_label_available_at,utc=True)
            origins=pd.to_datetime(fits.fit_origin,utc=True)
            assert (available.dropna()<origins.loc[available.notna()]).all(),path.name
    forecasts=pd.read_csv(ROOT/"reports/forecasts/all_test.csv.gz")
    for (model,day),group in forecasts.groupby(["model","local_day"]):
        assert len(group)==len(local_day_grid(day)),f"Incomplete delivery trajectory {model} {day}"
    report={"notebooks":rows,"raw_sources_unchanged":True,"audited_values_preserved":True,"forecast_archives_have_no_outcomes":True,"verified_per_origin_ledgers":origin_ledger_checks,"training_labels_before_fit":True,"test_delivery_grids_complete":True,"visual_inspection":"PNG outputs exported for human/assistant inspection; structural checks alone do not establish visual acceptance"}
    write_json(ROOT/"reports/execution_validation.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
