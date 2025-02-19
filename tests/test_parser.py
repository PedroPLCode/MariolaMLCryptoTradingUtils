import pytest
from utils.parser_utils import get_parsed_arguments


def test_get_parsed_arguments_with_one_argument(monkeypatch):
    test_args = ["script.py", "required_argument"]
    monkeypatch.setattr("sys.argv", test_args)

    result = get_parsed_arguments("Description for the first argument")
    assert result == ("required_argument", None)


def test_get_parsed_arguments_with_two_arguments(monkeypatch):
    test_args = ["script.py", "required_argument", "optional_argument"]
    monkeypatch.setattr("sys.argv", test_args)

    result = get_parsed_arguments(
        "Description for the first argument", "Description for the second argument"
    )
    assert result == ("required_argument", "optional_argument")


def test_get_parsed_arguments_with_missing_required_argument(monkeypatch):
    test_args = ["script.py"]
    monkeypatch.setattr("sys.argv", test_args)

    with pytest.raises(SystemExit):
        get_parsed_arguments("Description for the first argument")


def test_get_parsed_arguments_help_message(monkeypatch, capsys):
    test_args = ["script.py", "--help"]
    monkeypatch.setattr("sys.argv", test_args)

    with pytest.raises(SystemExit):
        get_parsed_arguments(
            "Description for the first argument", "Description for the second argument"
        )

    captured = capsys.readouterr()
    assert "A script that accepts one or two arguments." in captured.out
    assert "A required argument. Description for the first argument" in captured.out
    assert "An optional argument." in captured.out
