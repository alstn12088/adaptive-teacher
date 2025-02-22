from typing import List
import itertools

from torch import nn


class MLP(nn.Sequential):
    def __init__(
        self,
        dim_list: List[int],
        activation: nn.Module = nn.LeakyReLU(),
        tail: List[nn.Module] = [],
    ) -> None:
        super().__init__(
            *(
                list(
                    itertools.chain(
                        *[
                            [nn.Linear(i, o)] + ([activation] if n < len(dim_list) - 2 else [])
                            for n, (i, o) in enumerate(zip(dim_list, dim_list[1:]))
                        ]
                    )
                )
                + tail
            )
        )
