import json
import argparse
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str)
    args = parser.parse_args()

    with open(args.fname, "r") as f:
        data = json.load(f)
    
    results = data["results"]
    abbrv_results = [
        [
            result["name"],
            result["median(elapsed)"],
            result["medianAbsolutePercentError(elapsed)"],
            result["median(instructions)"],
            result["medianAbsolutePercentError(instructions)"],
            result["median(cpucycles)"],
            result["median(branchinstructions)"],
            result["median(branchmisses)"]
        ] for result in results
    ]

    with open(args.fname.removesuffix(".json") + ".csv", "r") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "elapsed", "err%", "instr.", "err%", "cycles", "branches", "misses"])
        for result in abbrv_results:
            writer.writerow(result)