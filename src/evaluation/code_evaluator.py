from smolagents import LocalPythonExecutor


class CodeEvaluator:
    """Class for evaluating Python code execution."""

    def __init__(self):
        self.python_executor = LocalPythonExecutor(additional_authorized_imports=[])
        self.python_executor.send_tools(tools={})

    def evaluate_task(self, code: str, test_list: list[str]) -> tuple[bool, str, str]:
        """
        Evaluate code execution and return success status.

        Args:
            code (str): Python code to evaluate
            test_list (list[str]): List of asserts to evaluate the code agaist.

        Returns:
            tuple[bool, str, str]: Tuple containing:
                - bool: True if code executes without errors, False otherwise
                - str: Output from code execution
                - str: Execution logs
        """
        try:
            # Create a new string combining code and tests instead of modifying input
            code_with_tests = code + "\n" + "\n".join(test_list)

            output, execution_logs, _ = self.python_executor(code_with_tests)
            return True, output, execution_logs
        except Exception as e:
            return False, "", str(e)
