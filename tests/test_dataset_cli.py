"""Tests for the dataset CLI introduced with the first real adapter."""

from formant_benchmark.cli.main import main


def test_dataset_list_includes_mcqll(capsys) -> None:
    assert main(["dataset", "list"]) == 0
    assert "mcqll_formants" in capsys.readouterr().out.splitlines()
