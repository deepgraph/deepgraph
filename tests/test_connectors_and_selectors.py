import textwrap

import pytest
from unittest.mock import patch, Mock
from deepgraph.connectors_and_selectors import CreatorFunction
from deepgraph import output_names


# dummy values
x = y = z = 1


def connector_valid(a, b, c_s, c_t):
    r1 = (c_s - c_t) * a
    r2 = (c_t - c_s) * b
    return r1, r2


@output_names("r1", "r2")
def connector_valid_decorated(a, b, c_s, c_t):
    r1 = (c_s - c_t) * a
    r2 = (c_t - c_s) * b
    return r1, r2


def selector_valid(a, b, c_s, c_t, sources, targets):
    r1 = (c_s - c_t) * a
    r2 = (c_t - c_s) * b
    sources = sources[a == b]
    targets = targets[a == b]
    return r1, r2, sources, targets

@output_names("r1", "r2", "sources", "targets")
def selector_valid_decorated(a, b, c_s, c_t, sources, targets):
    r1 = (c_s - c_t) * a
    r2 = (c_t - c_s) * b
    sources = sources[a == b]
    targets = targets[a == b]
    return r1, r2, sources, targets


class TestCreatorFunctionInit:

    @pytest.mark.parametrize(
        "fct, expected_name, expected_features, expected_rs",
        [
            (connector_valid, "connector_valid", ["c_s", "c_t"], ["a", "b"]),
            (connector_valid_decorated, "connector_valid_decorated", ["c_s", "c_t"], ["a", "b"]),
            (selector_valid, "selector_valid", ["c_s", "c_t"], ["a", "b"]),
            (selector_valid_decorated, "selector_valid_decorated", ["c_s", "c_t"], ["a", "b"]),
        ]
    )
    def test_input_extraction(self, fct, expected_name, expected_features, expected_rs):
        cf = CreatorFunction(fct)
        assert cf.fct == fct
        assert cf.name == expected_name
        assert cf.input_features == expected_features
        assert cf.input_rs == expected_rs

    @pytest.mark.parametrize(
        "fct, use_mock, expected_output_rs, expected_output",
        [
            (connector_valid, False, ["r1", "r2"], ["r1", "r2"]),
            (connector_valid_decorated, True, ["r1", "r2"], ["r1", "r2"]),
            (selector_valid, False, ["r1", "r2"], ["r1", "r2", "sources", "targets"]),
            (selector_valid_decorated, True, ["r1", "r2"], ["r1", "r2", "sources", "targets"]),
        ]
    )
    def test_output_extraction(self, fct, use_mock, expected_output_rs, expected_output):
        if use_mock:
            with patch.object(
                    CreatorFunction,
                    "_extract_return_variables",
                    wraps=CreatorFunction._extract_return_variables
            ) as mock_extract:
                cf = CreatorFunction(fct)
                mock_extract.assert_not_called()
                assert cf.output_rs == expected_output_rs
                assert cf.output == expected_output
        else:
            cf = CreatorFunction(fct)
            assert cf.output_rs == expected_output_rs
            assert cf.output == expected_output

    def test_getsource_raises_oserror(self):
        exec_namespace = {}
        fct_string = """
        def connector_valid_via_exec(a, b, c_s, c_t):
            r1 = (c_s - c_t) * a
            r2 = (c_t - c_s) * b
            return r1, r2
        """
        exec(textwrap.dedent(fct_string), exec_namespace)
        fct = exec_namespace["connector_valid_via_exec"]

        with pytest.raises(OSError) as excinfo:
            CreatorFunction(fct)

        assert "Unable to retrieve the source code of the function" in str(excinfo.value)
        assert "connector_valid_via_exec" in str(excinfo.value)


    class TestOutputExtractionUsingSourceCode:

        def test_single_variable_output(self):
            def f():
                return x
            cf = CreatorFunction(f)
            assert cf.output == ["x"]

        def test_multiple_variable_output(self):
            def f():
                return x, y, z
            cf = CreatorFunction(f)
            assert cf.output == ["x", "y", "z"]

        def test_multiple_variable_explicit_tuple_output(self):
            def f():
                return (x, y, z)
            cf = CreatorFunction(f)
            assert cf.output == ["x", "y", "z"]

        def test_multiple_variable_explicit_tuple_output_multiple_lines(self):
            def f():
                return (
                    x,
                    y,
                    z
                )
            cf = CreatorFunction(f)
            assert cf.output == ["x", "y", "z"]

        def test_return_expression_raises(self):
            def f():
                return x + y
            with pytest.raises(ValueError, match="Return must consist of variable names only."):
                CreatorFunction(f)

        def test_return_function_call_raises(self):
            func = lambda x: x
            def f():
                return func(y)
            with pytest.raises(ValueError, match="Return must consist of variable names only."):
                CreatorFunction(f)

        def test_no_return_statement_raises(self):
            def f():
                pass
            with pytest.raises(ValueError, match="No return statement found in the function."):
                CreatorFunction(f)

        def test_tuple_with_expression_raises(self):
            def f():
                return x, y + z
            with pytest.raises(ValueError, match="Return tuple must contain only variable names."):
                CreatorFunction(f)
