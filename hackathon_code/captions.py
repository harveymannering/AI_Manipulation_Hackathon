"""Define our captioning strategy."""

from typing import List


class ConstantCaptions:
    """Every image has the same caption."""

    def __init__(self, default_caption: str = "") -> None:
        """Always output a constant caption.

        Args:
            default_caption (str): The default text prompt which we always return."".
        """
        self.default_caption = default_caption

    def get_caption(self, batch_size: int) -> List[str]:
        """Return the text prompt.

        Args:
            batch_size (int): The total number of times our prompt will be copied."".

        Returns:
            List[str]: A text prompt which will be always be the same.
        """
        return [self.default_caption] * batch_size

