from leaklens import __version__
from leaklens.cli import main


def test_version_option(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0

    assert capsys.readouterr().out.strip() == __version__
