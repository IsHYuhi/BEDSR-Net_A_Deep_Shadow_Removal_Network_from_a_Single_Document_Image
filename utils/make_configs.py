import argparse
import dataclasses
import itertools
import os
import sys

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from typing import Any, Dict, List, Tuple

from libs.config import Config


def str2bool(val: str) -> bool:
    if isinstance(val, bool):
        return val
    if val.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif val.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(description="make configuration yaml files.")

    parser.add_argument(
        "--root_dir",
        type=str,
        default="./configs/",
        help="path to a directory where you want to make config files and directories.",
    )

    fields = dataclasses.fields(Config)

    for field in fields:
        type_func = str2bool if field.type is bool else field.type

        if isinstance(field.default, dataclasses._MISSING_TYPE):
            # default value is not set.
            # do not specify boolean type in argparse
            # ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
            parser.add_argument(
                f"--{field.name}",
                type=type_func,
                nargs="*",
                required=True,
            )
        elif hasattr(field.type, "__origin__"):
            # the field type is Tuple or not.
            # https://github.com/zalando/connexion/issues/739
            parser.add_argument(
                f"--{field.name}",
                type=field.type.__args__[0],
                action="append",
                nargs="+",
                default=[list(field.default)],
            )
        else:
            # default value is provided in config dataclass.
            parser.add_argument(
                f"--{field.name}",
                type=type_func,
                nargs="*",
                default=field.default,
            )

    return parser.parse_args()


def convert_tuple2list(_dict: Dict[str, Any]) -> Dict[str, Any]:
    # cannot use tuple in yaml file for safe loading.
    for key, val in _dict.items():
        if isinstance(val, tuple):
            _dict[key] = tuple(val)

    return _dict


def parse_params(
    args_dict: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str], List[List[Any]]]:

    base_config = {}
    variable_keys = []
    variable_values = []

    for k, v in args_dict.items():
        if isinstance(v, list):
            variable_keys.append(k)
            variable_values.append(v)
        else:
            base_config[k] = v

    return base_config, variable_keys, variable_values


def get_n_options(
    variable_keys: List[str], variable_values: List[List[Any]]
) -> Dict[str, int]:
    cnt = {}
    for k, v in zip(variable_keys, variable_values):
        cnt[k] = len(v)

    return cnt


def generate_and_save_config(
    base_config: Dict[str, Any],
    variable_keys: List[str],
    values: Tuple[Any],
    root_dir: str,
    n_options_dict: Dict[str, int],
) -> None:
    config = base_config.copy()
    param_list = []
    for k, v in zip(variable_keys, values):
        config[k] = v

        if n_options_dict[k] == 1:
            continue
        else:
            param_list.append(f"{k}={v}")

    dir_name = "-".join(param_list)
    dir_path = os.path.join(root_dir, dir_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = os.path.join(dir_path, "config.yaml")

    # save configuration file as yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def main() -> None:
    args = get_arguments()

    # convert Namespace to dictionary.
    args_dict = vars(args).copy()
    del args_dict["root_dir"]

    base_config, variable_keys, variable_values = parse_params(args_dict)

    # base_config may contain tuple object and they should be converted.
    base_config = convert_tuple2list(base_config)

    # get direct product
    product = itertools.product(*variable_values)

    # get the number of options for each key.
    n_options_dict = get_n_options(variable_keys, variable_values)

    # make a directory and save configuration file there.
    for values in product:
        generate_and_save_config(
            base_config, variable_keys, values, args.root_dir, n_options_dict
        )

    print("Finished making configuration files.")


if __name__ == "__main__":
    main()