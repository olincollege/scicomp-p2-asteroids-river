"""
Generalized runner for the project. Takes a step to run from the command line, or prompts for one if not provided, and runs that step.
The steps are defined in the steps/ directory, and are prefixed with a 2-digit number to indicate order.
"""

import sys
import importlib
import os

def main():
    steps = [f[:-3] for f in os.listdir("steps") if f.endswith(".py") and f != "__init__.py"]
    if len(sys.argv) > 1:
        step_choice = sys.argv[1]
    else:
        print("Available steps:")
        for step in steps:
            print(f"- {step}")
        step_choice = input("Enter the number of the step to run: ")
    step_ids = [step.split("_")[0] for step in steps]
    if step_choice not in step_ids:
        print("Invalid step choice, exiting.")
        return
    step_module_name = [step for step in steps if step.startswith(step_choice)][0]
    step_module = importlib.import_module(f"steps.{step_module_name}")
    step_module.main()
    print("==== Step complete ====")

if __name__ == "__main__":
    while True:
        main()