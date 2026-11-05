import pandas as pd
from fastapi.testclient import TestClient

from leaklens.api import app
from leaklens.cli import main


def test_parquet_input_supported(tmp_path, capsys):
    df = pd.DataFrame({"x": [0, 1, 0, 1], "y": [0, 1, 0, 1]})
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path, index=False)

    response = TestClient(app).post(
        "/audit",
        files={
            "file": (
                "data.parquet",
                parquet_path.read_bytes(),
                "application/octet-stream",
            )
        },
        data={"target": "y"},
    )
    assert response.status_code == 200, response.text

    assert main([str(parquet_path), "--target", "y", "--json"]) == 0
    assert '"trust_score"' in capsys.readouterr().out

    test_path = tmp_path / "test.parquet"
    pd.DataFrame({"x": [4, 5, 6, 7], "y": [0, 1, 0, 1]}).to_parquet(
        test_path, index=False
    )
    assert (
        main([str(parquet_path), "--test", str(test_path), "--target", "y", "--json"])
        == 0
    )
    output = capsys.readouterr().out
    assert '"trust_score"' in output
