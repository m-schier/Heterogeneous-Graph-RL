import sys

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_main(cfg: DictConfig):
    # Just required because hydra drops the exception if we don't write it out
    try:
        inner_main(cfg)
    except Exception:
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        raise


def inner_main(*args):
    from HeterogeneousGraphRL.Training import main
    main(*args)


if __name__ == '__main__':
    hydra_main()  # Ok to be missing args as filled by hydra
