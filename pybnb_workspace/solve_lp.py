import gurobipy as gp
import argparse
import sys

def solve_lp_file(file_path):
    """
    Loads and solves an LP file using Gurobi.

    Args:
        file_path (str): The path to the LP file.
    """
    try:
        # Create a Gurobi environment
        env = gp.Env(empty=True)
        env.setParam("LogFile", "gurobi.log") # Optional: set a log file
        env.start()

        # Read the model from the specified LP file
        print(f"Reading model from {file_path}...")
        model = gp.read(file_path, env=env)

        # Optimize the model
        print("Optimizing model...")
        model.optimize()

        # Check the status of the optimization
        if model.status == gp.GRB.OPTIMAL:
            print("\nOptimization successful! 🎉")
            print(f"Objective value: {model.objVal}")
            print("\nSolution:")
            var_name_col = []
            for v in model.getVars(): 
                if v.x > 1e-6: var_name_col.append(v.varName)
            print(var_name_col)
            for v in model.getVars():
                if v.x > 1e-6: # Print variables with non-zero values
                    print(f"{v.varName}: {v.x} * {v.obj} = {v.x * v.obj}")
                    # print constraint coefficient of this variable with its name
                    for constr in model.getConstrs():
                        coeff = model.getCoeff(constr, v)
                        if abs(coeff) > 1e-6:
                            print(f"  Coefficient in {constr.constrName}: {coeff}")
        elif model.status == gp.GRB.INF_OR_UNBD:
            print("Model is infeasible or unbounded.")
        elif model.status == gp.GRB.INFEASIBLE:
            print("Model is infeasible.")
        elif model.status == gp.GRB.UNBOUNDED:
            print("Model is unbounded.")
        else:
            print(f"Optimization ended with status: {model.status}")

    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def main():
    """
    Main function to parse command-line arguments and run the solver.
    """
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Solve a Gurobi LP file from the command line.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "lp_file_path",
        help="The path to the LP file to be solved."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function to solve the LP file
    solve_lp_file(args.lp_file_path)

if __name__ == "__main__":
    main()

