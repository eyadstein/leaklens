from leaklens.scanner import scan_source

SRC = """
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
le = LabelEncoder()
y = le.fit_transform(y)
a = train_test_split(X, y)
"""


def test_label_encoder_is_low():
    fs = scan_source(SRC)
    assert fs and all(f.severity == "low" for f in fs)
