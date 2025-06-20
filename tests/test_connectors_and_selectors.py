import pytest

from deepgraph.connectors_and_selectors import CreatorFunction


class TestCreatorFunctionInit:

    class TestOutputExtraction:

        def test_single_variable_output(self):
            x = 1
            def f():
                return x
            cf = CreatorFunction(f)
            assert cf.output == ["x"]

        def test_multiple_variable_output(self):
            a = b = c = 1
            def f():
                return a, b, c
            cf = CreatorFunction(f)
            assert cf.output == ["a", "b", "c"]

        def test_multiple_variable_explicit_tuple_output(self):
            a = b = c = 1
            def f():
                return (a, b, c)
            cf = CreatorFunction(f)
            assert cf.output == ["a", "b", "c"]

        def test_multiple_variable_explicit_tuple_output_multiple_lines(self):
            a = b = c = 1
            def f():
                return (
                    a,
                    b,
                    c
                )
            cf = CreatorFunction(f)
            assert cf.output == ["a", "b", "c"]

        def test_excludes_sources_targets_in_output_rs(self):
            sources = targets = r1 = r2 = 1
            def f():
                return sources, targets, r1, r2
            cf = CreatorFunction(f)
            assert cf.output == ["sources", "targets", "r1", "r2"]
            assert cf.output_rs == ["r1", "r2"]

        def test_return_expression_raises(self):
            a = b = 1
            def f():
                return a + b
            with pytest.raises(ValueError, match="Return must consist of variable names only."):
                CreatorFunction(f)

        def test_return_function_call_raises(self):
            func = lambda x: x
            a = 1
            def f():
                return func(a)
            with pytest.raises(ValueError, match="Return must consist of variable names only."):
                CreatorFunction(f)

        def test_no_return_statement_raises(self):
            def f():
                pass
            with pytest.raises(ValueError, match="No return statement found in the function."):
                CreatorFunction(f)

        def test_tuple_with_expression_raises(self):
            x = y = z = 1
            def f():
                return x, y + z
            with pytest.raises(ValueError, match="Return tuple must contain only variable names."):
                CreatorFunction(f)
