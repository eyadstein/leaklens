import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from fastapi import HTTPException

from leaklens.audit import audit
from leaklens.fixes import advice
from leaklens.intelligence import analyze_dataset_impact
from leaklens.leakage import scan, score, split_overlap, temporal_overlap
from leaklens.report import render
from leaklens.scanner import scan_file, scan_source

st.set_page_config(page_title="LeakLens", layout="wide")
st.title("LeakLens")
st.caption("Catch data leakage before it fools you.")
t1, t2, t3 = st.tabs(["Dataset audit", "Code scanner", "Model arena"])


def show(fs, key, audit_data=None, impact_data=None):
    findings = [f.to_dict() for f in fs]
    trust_score = score(fs)

    c1, c2 = st.columns(2)
    c1.metric("Trust score", f"{trust_score}/100")
    c2.metric("High-severity leaks", sum(f.severity == "high" for f in fs))
    if fs:
        st.dataframe(pd.DataFrame(findings))
        for finding in findings:
            suggestion = advice(finding["kind"])
            if suggestion:
                st.markdown(
                    f"**Suggested fix for {finding['kind']} ({finding['column']}):**"
                )
                st.write(suggestion)
    else:
        st.success("No leakage found.")

    report = {"trust_score": trust_score, "findings": findings}
    if audit_data is not None:
        report["audit"] = audit_data
    if impact_data is not None:
        report["impact"] = impact_data
    st.download_button(
        "Download HTML report",
        data=render(report),
        file_name="leaklens-report.html",
        mime="text/html",
        key=f"{key}_report",
    )


with t1:
    up = st.file_uploader("Training CSV", type="csv", key="tr")
    if up:
        from leaklens.api import UploadLimits, validate_upload_size
        from leaklens.readers import DatasetLimitError, read_dataset

        try:
            validate_upload_size(up.name or "upload", up.size or len(up.getvalue()))
            up.seek(0)
            df = read_dataset(up, up.name, max_rows=UploadLimits.MAX_ROWS)
        except (DatasetLimitError, HTTPException, ValueError) as exc:
            st.error(f"Invalid dataset: {exc}")
            st.stop()
        target = st.selectbox(
            "Target column", list(df.columns), index=len(df.columns) - 1
        )
        thr = st.slider("Leak threshold", 0.80, 1.00, 0.95, 0.01)
        te_up = st.file_uploader(
            "Optional test CSV (checks overlap)", type="csv", key="te"
        )
        fs = scan(df, target, thr)
        impact_data = None
        if te_up:
            try:
                validate_upload_size(
                    te_up.name or "upload", te_up.size or len(te_up.getvalue())
                )
                te_up.seek(0)
                te = read_dataset(te_up, te_up.name, max_rows=UploadLimits.MAX_ROWS)
            except (DatasetLimitError, HTTPException, ValueError) as exc:
                st.error(f"Invalid test dataset: {exc}")
                st.stop()
            fs += split_overlap(df, te)
            tc = st.selectbox("Time column (optional)", ["", *list(df.columns)])
            if tc and tc in te.columns:
                fs += temporal_overlap(df, te, tc)
            impact = analyze_dataset_impact(
                df, te, time_col=tc if tc and tc in te.columns else None
            )
            impact_data = impact.to_dict()
            st.subheader("Measured dataset impact")
            m1, m2, m3 = st.columns(3)
            m1.metric("Duplicate test rows", impact.duplicate_test_rows)
            m2.metric("Duplicate test rows (%)", f"{impact.duplicate_test_pct:.2f}%")
            m3.metric("Test rows", impact.test_rows)
            if impact.time_column:
                st.write(
                    {
                        "Temporal boundary overlap": impact.temporal_overlap,
                        "Latest training timestamp": impact.train_max_time,
                        "Earliest test timestamp": impact.test_min_time,
                    }
                )
        audit_data = audit(df, target)
        show(fs, key="dataset", audit_data=audit_data, impact_data=impact_data)
        st.json(audit_data, expanded=False)

with t2:
    code = st.text_area("Paste Python code", height=220)
    nb = st.file_uploader("...or a .py / .ipynb file", type=["py", "ipynb"])
    if nb:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / Path(nb.name).name
            p.write_bytes(nb.getvalue())
            show(scan_file(p), key="code")
    elif code.strip():
        show(scan_source(code, "pasted"), key="code")

with t3:
    up3 = st.file_uploader("Training CSV", type="csv", key="ar")
    if up3:
        from leaklens.api import UploadLimits, validate_upload_size
        from leaklens.readers import DatasetLimitError, read_dataset

        try:
            validate_upload_size(up3.name or "upload", up3.size or len(up3.getvalue()))
            up3.seek(0)
            df3 = read_dataset(up3, up3.name, max_rows=UploadLimits.MAX_ROWS)
        except (DatasetLimitError, HTTPException, ValueError) as exc:
            st.error(f"Invalid dataset: {exc}")
            st.stop()
        tgt = st.selectbox(
            "Target", list(df3.columns), index=len(df3.columns) - 1, key="tg"
        )
        tcol = st.selectbox(
            "Time column (optional)", ["", *list(df3.columns)], key="tc"
        )
        if st.button("Run arena"):
            try:
                from leaklens.arena import run
            except ImportError:
                st.error('pip install -e ".[arena]"')
            else:
                with st.spinner("Training 3 model families..."):
                    res = run(df3, tgt, tcol or None)
                st.write(
                    f"Ranked by {res.attrs['primary']}. Auto-dropped: {', '.join(map(str, res.attrs['dropped'])) or 'none'}"
                )
                st.dataframe(res.round(4))
                st.caption("Best model saved to models/best.joblib")
