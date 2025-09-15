# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from training.trainer import Trainer

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg):
    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()