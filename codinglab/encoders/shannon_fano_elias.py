# codinglab/encoders/shannon_fano_elias.py
"""Shannon-Fano-Elias encoder implementation."""

import math
from collections import OrderedDict
from enum import Enum
from typing import Dict, Optional, Sequence

from codinglab.types import SourceChar
from codinglab.encoders.prefix_code_tree import PrefixCodeTree
from codinglab.encoders.prefix_coder import PrefixEncoderDecoder


class BinaryAlphabet(str, Enum):
    """Binary alphabet for encoding/decoding."""

    zero = "0"
    one = "1"


class ShannonFanoEliasBinaryCoder(PrefixEncoderDecoder[SourceChar, BinaryAlphabet]):
    """
    Shannon-Fano-Elias Coder for prefix codes.

    This coder implements the Shannon-Fano-Elias coding algorithm, which
    constructs prefix codes based on the cumulative distribution function.
    For each symbol x with probability p(x), the code length is:
    l(x) = -⌈log₂(p(x))⌉ + 1
    The code is the binary representation of the modified cumulative
    probability F̄(x) = Σ_{a<x} p(a) + p(x)/2, truncated to l(x) bits.

    The generated codes are guaranteed to be:
    - Prefix-free (no code is a prefix of another)
    - In lexicographic order (codes ordered like their respective symbols)
    - Binary (using only 0 and 1)

    Attributes:
        probabilities: Ordered dictionary mapping source symbols to their probabilities
        expected_code_length: Average length of codewords
        entropy: Shannon entropy of the source
        coding_efficiency: Ratio of entropy to expected code length
    """

    def __init__(
        self,
        probabilities: OrderedDict[SourceChar, float],
    ) -> None:
        """Create a new Shannon-Fano-Elias coder.

        Args:
            probabilities: ordered dictionary mapping each source symbol to
                its probability. The sum of the probabilities must be
                approximately one and all values must be positive.

        Raises:
            ValueError: if probabilities are non-positive or do not sum to
                one (within a small tolerance).
        """
        if not probabilities:
            raise ValueError("Probabilities dictionary cannot be empty")

        total = sum(probabilities.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Probabilities must sum to 1.0")

        for sym, p in probabilities.items():
            if p <= 0:
                raise ValueError(f"Probability for {sym!r} must be positive")

        # Store a copy so the caller cannot mutate under us
        self.probabilities = OrderedDict(probabilities)
        self._F: Dict[SourceChar, float] = {}
        # Store codes as strings internally for efficiency
        self._code_strings: Dict[SourceChar, str] = {}
        self._tree = None
        super().__init__(
            list(probabilities.keys()),
            [BinaryAlphabet.zero, BinaryAlphabet.one],
        )
        self._build_prefix_code_tree()

    def _build_prefix_code_tree(self) -> None:
        """Build the prefix code tree and code table using Shannon-Fano-Elias algorithm.

        For each symbol in order:
        1. Calculate modified cumulative probability F̄(x) = Σ_{a<x} p(a) + p(x)/2
        2. Determine code length l(x) = -⌈log₂(p(x))⌉ + 1
        3. Extract binary representation of F̄(x) to l(x) bits
        4. Insert code into prefix tree
        """
        cumul_sum = 0.0
        self._tree = PrefixCodeTree()
        self._code_strings = {}

        for symbol, p in self.probabilities.items():
            # Calculate modified cumulative distribution function
            self._F[symbol] = cumul_sum + p / 2
            cumul_sum += p

            # Calculate code length
            word_len = -math.ceil(math.log2(p)) + 1

            # Get binary code as string
            codeword_str = self._get_binary_code(self._F[symbol], word_len)
            self._code_strings[symbol] = codeword_str

            # Convert to BinaryAlphabet sequence for tree
            codeword_seq = [
                BinaryAlphabet.zero if bit == "0" else BinaryAlphabet.one
                for bit in codeword_str
            ]

            # Insert into tree for decoding
            self._tree.insert_code(codeword_seq, symbol)

    @property
    def expected_code_length(self) -> float:
        """Calculate expected code length (average bits per symbol).

        Returns:
            Average length of codewords weighted by their probabilities.
        """
        return sum(
            p * len(self._code_strings[symbol])
            for symbol, p in self.probabilities.items()
        )

    @property
    def entropy(self) -> float:
        """Calculate Shannon entropy of the source.

        Returns:
            Shannon entropy H = -Σ p(x) * log₂(p(x)) in bits
        """
        return -sum(p * math.log2(p) for p in self.probabilities.values())

    @property
    def coding_efficiency(self) -> float:
        """Calculate coding efficiency (entropy / expected code length).

        Returns:
            Efficiency in range [0, 1], where 1 is optimal.
        """
        E = self.expected_code_length
        H = self.entropy
        return H / E if E > 0 else 0.0

    @property
    def code_table(self) -> Optional[Dict[SourceChar, Sequence[BinaryAlphabet]]]:
        """Get the code table as a dictionary.

        Returns:
            Dictionary mapping source symbols to sequences of BinaryAlphabet,
            or None if code table not available.
        """
        if not self._code_strings:
            return None
        return {
            sym: [
                BinaryAlphabet.zero if bit == "0" else BinaryAlphabet.one
                for bit in code_str
            ]
            for sym, code_str in self._code_strings.items()
        }

    def _get_binary_code(self, value: float, length: int) -> str:
        """Convert a fractional value to binary code.

        Extracts the first `length` bits from the binary representation
        of the fractional part of `value`.

        Args:
            value: Fractional value in [0, 1] to convert
            length: Number of bits to extract

        Returns:
            Binary string of length `length` representing the value
        """
        code = []
        for _ in range(length):
            value *= 2
            bit = int(value)
            code.append(str(bit))
            value -= bit
        return "".join(code)
