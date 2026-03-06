"""Tests for dlnm.datasets module."""

import pandas as pd

from dlnm.datasets import load_chicagoNMMAPS, load_drug, load_nested


class TestChicagoNMMAPS:
    def test_load(self):
        df = load_chicagoNMMAPS()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (5114, 14)

    def test_columns(self):
        df = load_chicagoNMMAPS()
        expected = [
            "date",
            "time",
            "year",
            "month",
            "doy",
            "dow",
            "death",
            "cvd",
            "resp",
            "temp",
            "dptp",
            "rhum",
            "pm10",
            "o3",
        ]
        assert df.columns.tolist() == expected

    def test_no_all_null_columns(self):
        df = load_chicagoNMMAPS()
        for col in ["death", "temp"]:
            assert df[col].notna().sum() > 0


class TestDrug:
    def test_load(self):
        df = load_drug()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (200, 7)

    def test_columns(self):
        df = load_drug()
        assert "id" in df.columns
        assert "out" in df.columns


class TestNested:
    def test_load(self):
        df = load_nested()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (600, 14)

    def test_columns(self):
        df = load_nested()
        assert "case" in df.columns
        assert "age" in df.columns
