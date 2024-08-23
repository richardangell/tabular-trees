import pytest

from tabular_trees.xgboost.dump_parser import XGBoostParser
from tabular_trees.xgboost.dump_reader import (
    JsonDumpReader,
    TextDumpReader,
)


class TestInitialisation:
    """Tests for the XGBoostParser.__init__ method."""

    def test_model_not_booster_exception(self):
        """Test that a TypeError is raised if model is not a Booster."""
        with pytest.raises(
            TypeError,
            match=(
                "model is not in expected types <class 'xgboost.core.Booster'>, "
                "got <class 'tuple'>"
            ),
        ):
            XGBoostParser((1, 2, 3))

    @pytest.mark.parametrize("dump_reader", [(JsonDumpReader), (TextDumpReader)])
    def test_attributes_set(self, xgb_diabetes_model, dump_reader):
        """Test model, dump_type and reader attributes are correctly set."""
        xgboost_parser = XGBoostParser(xgb_diabetes_model, dump_reader())

        assert (
            xgboost_parser.model == xgb_diabetes_model
        ), "model attribute not set correctly"

        assert (
            type(xgboost_parser.reader) is dump_reader
        ), "incorrect type for reader attribute"

    def test_depreceation_warning(self, xgb_diabetes_model):
        """Test a FutureWarning is raised when initialising an XGBoostParser object."""
        expected_warning = (
            "XGBoostDumpParser class is depreceated, "
            "Booster.trees_to_dataframe is available instead"
        )

        with pytest.warns(FutureWarning, match=expected_warning):
            XGBoostParser(xgb_diabetes_model)
