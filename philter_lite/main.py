import argparse
import distutils.util
import os

import philter_lite


def main():
    # get input/output/filename
    help_str = """ Philter -- PHI filter for clinical notes """
    ap = argparse.ArgumentParser(description=help_str)
    ap.add_argument(
        "-i",
        "--input",
        default="./data/i2b2_notes/",
        help="Path to the directory or the file that contains the PHI note, the default is "
        "./data/i2b2_notes/",
        type=str,
    )
    ap.add_argument(
        "-a",
        "--anno",
        default="./data/i2b2_anno/",
        help="Path to the directory or the file that contains the PHI annotation, the default is "
        "./data/i2b2_anno/",
        type=str,
    )
    ap.add_argument(
        "-o",
        "--output",
        default="./data/i2b2_results/",
        help="Path to the directory to save the PHI-reduced notes in, the default is ./data/i2b2_results/",
        type=str,
    )
    ap.add_argument(
        "-f",
        "--filters",
        default="./configs/integration_1.json",
        help="Path to our config file, the default is ./configs/integration_1.json",
        type=str,
    )
    ap.add_argument(
        "-x",
        "--xml",
        default="./data/phi_notes.json",
        help="Path to the json file that contains all xml data",
        type=str,
    )
    ap.add_argument(
        "-c",
        "--coords",
        default="./data/coordinates.json",
        help="Path to the json file that contains the coordinate map data",
        type=str,
    )
    ap.add_argument(
        "--eval_output",
        default="./data/phi/",
        help="Path to the directory that the detailed eval files will be outputted to",
        type=str,
    )
    ap.add_argument(
        "-v",
        "--verbose",
        default=True,
        help="When verbose is true, will emit messages about script progress",
        type=lambda x: bool(distutils.util.strtobool(x)),
    )
    ap.add_argument(
        "-t",
        "--freq_table",
        default=False,
        help="When freqtable is true, will output a unigram/bigram frequency table of all note words and "
        "their PHI/non-PHI counts",
        type=lambda x: bool(distutils.util.strtobool(x)),
    )
    ap.add_argument(
        "-n",
        "--initials",
        default=True,
        help="When initials is true, will include initials PHI in recall/precision calculations",
        type=lambda x: bool(distutils.util.strtobool(x)),
    )
    ap.add_argument(
        "--outputformat",
        default="i2b2",
        help='Define format of annotation, allowed values are "asterisk", "i2b2". Default '
        'is "asterisk"',
        type=str,
    )
    ap.add_argument(
        "--ucsfformat",
        default=False,
        help="When ucsfformat is true, will adjust eval script for slightly different xml format",
        type=lambda x: bool(distutils.util.strtobool(x)),
    )
    ap.add_argument(
        "--prod",
        default=False,
        help="When prod is true, this will run the script with output in i2b2 xml format without running "
        "the eval script",
        type=lambda x: bool(distutils.util.strtobool(x)),
    )
    ap.add_argument(
        "--cachepos",
        default=None,
        help="Path to a directoy to store/load the pos data for all notes. If no path is specified then "
        "memory caching will be used.",
        type=str,
    )

    args = ap.parse_args()
    verbose = args.verbose

    if args.prod:
        verbose = False

        philter_config = {
            "verbose": verbose,
            "finpath": args.input,
            "foutpath": args.output,
            "outformat": args.outputformat,
            "filters": args.filters,
            "cachepos": args.cachepos,
        }

    else:
        philter_config = {
            "verbose": args.verbose,
            "freq_table": args.freq_table,
            "initials": args.initials,
            "finpath": args.input,
            "foutpath": args.output,
            "outformat": args.outputformat,
            "ucsfformat": args.ucsfformat,
            "anno_folder": args.anno,
            "filters": args.filters,
            "xml": args.xml,
            "coords": args.coords,
            "eval_out": args.eval_output,
            "cachepos": args.cachepos,
        }

    if verbose:
        print("RUNNING ", philter_config["filters"])

    filters = philter_lite.load_filters(philter_config["filters"])

    for root, dirs, files in os.walk(philter_config["finpath"]):
        for file in files:
            with open(os.path.join(root, file)) as inf:
                text = inf.read()
                include_map, exclude_map, data_tracker = philter_lite.detect_phi(
                    text, patterns=filters
                )
                if philter_config["outformat"] == "i2b2":
                    with open(
                        os.path.join(philter_config["foutpath"], f"{file}.txt"), "w"
                    ) as fout:
                        fout.write(philter_lite.transform_text_i2b2(data_tracker))
                elif philter_config["outformat"] == "asterisk":
                    with open(
                        os.path.join(philter_config["foutpath"], f"{file}.txt"), "w"
                    ) as fout:
                        fout.write(
                            philter_lite.transform_text_asterisk(text, include_map)
                        )


if __name__ == "__main__":
    main()
