"""Module containing BoosterString class."""

from collections import OrderedDict

import lightgbm as lgb


class BoosterString:
    """String version of a LightGBM Booster."""

    new_line = "\n"

    def __init__(self, rows: list[str]):
        """Initialise the BoosterString from a list of strings.

        Parameters
        ----------
        rows: list[str]
            List of strings defining a Booster.

        """
        self.rows = rows[:]

        try:
            self.to_booster()
        except Exception as err:
            raise ValueError("supplied rows do not produce a valid booster") from err

        self.row_markers, self.tree_rows = self._gather_line_markers()

    @classmethod
    def from_booster(cls, booster: lgb.Booster) -> "BoosterString":
        """Create BoosterString from a lgb.Booster object.

        Returns
        -------
        model : BoosterString
            Model as a BoosterString object.

        """
        booster_string: str = booster.model_to_string()
        booster_string_split = booster_string.split(cls.new_line)

        return BoosterString(booster_string_split)

    def to_booster(self) -> lgb.Booster:
        """Convert the BoosterString back to a Booster.

        Returns
        -------
        model : lgb.Booster
            BoosterString as lgb.Booster object.

        """
        booster_string = self.new_line.join(self.rows)

        return lgb.Booster(model_str=booster_string)

    def extract_header_rows(self) -> list[str]:
        """Extract the header rows from BoosterString object.

        Returns
        -------
        rows : list[str]
            Header rows from Booster text.

        """
        end_row_index = self.row_markers["end_header"] + 1
        return self._extract_rows(0, end_row_index)

    def extract_tree_rows(self, tree_number: int) -> list[str]:
        """Extract rows for given tree number.

        Returns
        -------
        rows : list[str]
            Rows from Booster text for the given tree.

        """
        try:
            start_row_index = self.tree_rows[tree_number]
        except KeyError as err:
            raise ValueError(
                f"requested tree number {tree_number} does not exist in model"
            ) from err

        return self._extract_rows(start_row_index, start_row_index + 19)

    def extract_bottom_rows(self) -> list[str]:
        """Return all rows after the 'end of trees' line to the end.

        Returns
        -------
        rows : list[str]
            Final rows from Booster text.

        """
        return self._extract_rows(
            self.row_markers["end_of_trees"], self._get_number_of_rows()
        )

    def _gather_line_markers(self) -> tuple[OrderedDict, OrderedDict]:
        """Find specific lines in the booster string data."""
        row_to_find = [
            "tree_sizes=",
            "end of trees",
            "feature_importances:",
            "parameters:",
            "end of parameters",
            "pandas_categorical:",
        ]

        row_to_find_description = [
            "end_header",
            "end_of_trees",
            "feature_importances",
            "parameters",
            "end_of_parameters",
            "pandas_categorical",
        ]

        row_found_indexes: list[int] = []

        tree_rows = OrderedDict()

        tree_sizes_line = self._find_text_in_booster_data(row_to_find[0])
        row_found_indexes.append(tree_sizes_line)

        n_trees = self._get_number_trees_from_tree_sizes_line(
            self.rows[tree_sizes_line]
        )

        find_tree_from_row = tree_sizes_line

        for _ in range(n_trees):
            tree_line = self._find_text_in_booster_data("Tree=", find_tree_from_row)
            tree_number = self._get_tree_number_from_line(self.rows[tree_line])
            tree_rows[tree_number] = tree_line
            find_tree_from_row = tree_line + 1

        find_from_row = find_tree_from_row

        for string_to_find in row_to_find[1:]:
            string_found_row = self._find_text_in_booster_data(
                string_to_find, find_from_row
            )
            row_found_indexes.append(string_found_row)
            find_from_row = string_found_row + 1

        row_markers = OrderedDict()
        for name, row_index in zip(row_to_find_description, row_found_indexes):
            row_markers[name] = row_index

        return row_markers, tree_rows

    def _find_text_in_booster_data(self, text: str, start: int = 0) -> int:
        for subet_row_number, row in enumerate(self.rows[start:]):
            if row.find(text, 0, len(text)) == 0:
                row_number = subet_row_number + start

                return row_number

        raise ValueError(
            f"""unable to find row starting with text '{text}' in booster_data """
            f"starting from index {start}"
        )

    def _get_number_of_rows(self) -> int:
        return len(self.rows)

    def _extract_rows(self, start: int, end: int) -> list[str]:
        """Extract booster rows within range."""
        return self.rows[start:end]

    def _get_number_trees_from_tree_sizes_line(self, tree_sizes_line: str) -> int:
        """Get the number of trees in the booster from the tree sizes line."""
        return len(tree_sizes_line.split("=")[1].split(" "))

    def _get_tree_number_from_line(self, tree_line: str) -> int:
        """Extract the tree index from the first line in a tree section."""
        return int(tree_line.replace("Tree=", ""))
