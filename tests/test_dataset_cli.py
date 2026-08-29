"""Tests for the dataset CLI introduced with the first real adapter."""

from formant_benchmark.cli.main import main


def test_dataset_list_includes_builtin_adapters(capsys) -> None:
    assert main(["dataset", "list"]) == 0
    assert set(capsys.readouterr().out.splitlines()) >= {"mcqll_formants", "vtr"}
