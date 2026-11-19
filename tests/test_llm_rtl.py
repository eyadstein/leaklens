import pytest

from leaklens import llm
from leaklens.leakage import Finding


class _Provider:
    def __init__(self, response):
        self.response = response

    def explain(self, finding):
        return self.response


@pytest.mark.parametrize(
    "response",
    [
        "تم اكتشاف تسرب في البيانات؛ انقل المعالجة إلى Pipeline قبل التقسيم.",
        "זוהתה דליפת נתונים; יש להעביר את העיבוד ל־Pipeline לפני הפיצול.",
    ],
    ids=["arabic", "hebrew"],
)
def test_explanations_preserve_right_to_left_language_text(response):
    finding = Finding(
        "high",
        "preprocess_before_split",
        "train.py:8",
        "scaler fitted before split",
    )

    assert llm.explain_findings([finding], _Provider(response)) == [response]
