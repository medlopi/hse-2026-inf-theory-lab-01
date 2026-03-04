# tests/test_encoders/test_shannon_fano_elias.py
"""Tests for the Shannon-Fano-Elias encoder module."""

import pytest
import math
from collections import OrderedDict

from codinglab.encoders.shannon_fano_elias import (
    ShannonFanoEliasBinaryCoder,
    BinaryAlphabet,
)


class TestShannonFanoEliasBinaryCoder:
    """Tests for ShannonFanoEliasBinaryCoder."""

    @pytest.fixture
    def simple_probabilities(self):
        """Provide simple probability distribution."""
        return OrderedDict([("a", 0.5), ("b", 0.3), ("c", 0.2)])

    @pytest.fixture
    def skewed_probabilities(self):
        """Provide a skewed probability distribution."""
        return OrderedDict(
            [("a", 0.75), ("b", 0.025), ("c", 0.1), ("d", 0.05), ("e", 0.075)]
        )

    @pytest.fixture
    def uniform_probabilities(self):
        """Provide uniform probability distribution."""
        return OrderedDict([("a", 0.25), ("b", 0.25), ("c", 0.25), ("d", 0.25)])

    @pytest.fixture
    def two_symbol_probabilities(self):
        """Provide two-symbol probability distribution."""
        return OrderedDict([("a", 0.7), ("b", 0.3)])

    # ============= Initialization Tests =============

    def test_initialization_valid_simple(self, simple_probabilities):
        """Test valid initialization with simple probabilities."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        assert encoder.probabilities == simple_probabilities
        assert encoder._code_table is not None
        assert len(encoder._code_table) == 3

    def test_initialization_valid_skewed(self, skewed_probabilities):
        """Test valid initialization with skewed probabilities."""
        encoder = ShannonFanoEliasBinaryCoder(skewed_probabilities)
        assert encoder.probabilities == skewed_probabilities
        assert len(encoder._code_table) == 5

    def test_initialization_empty_probabilities(self):
        """Test initialization fails with empty probabilities."""
        with pytest.raises(ValueError, match="Probabilities dictionary cannot be empty"):
            ShannonFanoEliasBinaryCoder(OrderedDict())

    def test_initialization_probabilities_not_sum_to_one(self):
        """Test initialization fails when probabilities don't sum to 1."""
        probs = OrderedDict([("a", 0.5), ("b", 0.3)])
        with pytest.raises(ValueError, match="Probabilities must sum to 1.0"):
            ShannonFanoEliasBinaryCoder(probs)

    def test_initialization_negative_probability(self):
        """Test initialization fails with negative probability."""
        probs = OrderedDict([("a", 0.7), ("b", -0.3), ("c", 0.6)])
        with pytest.raises(ValueError, match="Probability.*must be positive"):
            ShannonFanoEliasBinaryCoder(probs)

    def test_initialization_zero_probability(self):
        """Test initialization fails with zero probability."""
        probs = OrderedDict([("a", 0.5), ("b", 0.0), ("c", 0.5)])
        with pytest.raises(ValueError, match="Probability.*must be positive"):
            ShannonFanoEliasBinaryCoder(probs)

    def test_initialization_single_symbol(self):
        """Test initialization with single symbol."""
        probs = OrderedDict([("a", 1.0)])
        encoder = ShannonFanoEliasBinaryCoder(probs)
        assert len(encoder._code_table) == 1

    # ============= Code Properties Tests =============

    def test_codes_are_prefix_free(self, simple_probabilities):
        """Test that generated codes are prefix-free."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        codes = list(encoder._code_strings.values())

        # Check no code is a prefix of another
        for i, code1 in enumerate(codes):
            for j, code2 in enumerate(codes):
                if i != j:
                    assert not code2.startswith(code1), (
                        f"Code {code2} starts with {code1} - "
                        "prefix-free property violated"
                    )

    def test_codes_are_binary(self, simple_probabilities):
        """Test that all codes contain only 0 and 1."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        for symbol, code in encoder._code_table.items():
            assert all(bit in ["0", "1"] for bit in code), (
                f"Code for {symbol!r} contains non-binary characters: {code}"
            )

    def test_code_lengths_correct(self, simple_probabilities):
        """Test that code lengths match the Shannon-Fano-Elias formula."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)

        for symbol, prob in encoder.probabilities.items():
            expected_length = -math.ceil(math.log2(prob)) + 1
            actual_length = len(encoder._code_table[symbol])
            assert actual_length == expected_length, (
                f"Code length for {symbol!r} mismatch: "
                f"expected {expected_length}, got {actual_length}"
            )

    def test_lexicographic_order(self, simple_probabilities):
        """Test that codes are in lexicographic order."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        symbols = list(encoder.probabilities.keys())
        codes = [encoder._code_table[sym] for sym in symbols]

        # Codes should be in lexicographic order
        for i in range(len(codes) - 1):
            assert codes[i] <= codes[i + 1], (
                f"Code for {symbols[i]!r} ({codes[i]}) comes after "
                f"code for {symbols[i+1]!r} ({codes[i+1]})"
            )

    # ============= Properties Tests =============

    def test_entropy_calculation(self, simple_probabilities):
        """Test entropy property calculation."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        expected_entropy = -sum(p * math.log2(p) for p in [0.5, 0.3, 0.2])
        assert abs(encoder.entropy - expected_entropy) < 1e-9

    def test_entropy_single_symbol(self):
        """Test entropy for single symbol (should be 0)."""
        probs = OrderedDict([("a", 1.0)])
        encoder = ShannonFanoEliasBinaryCoder(probs)
        assert encoder.entropy == 0.0

    def test_entropy_uniform_distribution(self, uniform_probabilities):
        """Test entropy for uniform distribution."""
        encoder = ShannonFanoEliasBinaryCoder(uniform_probabilities)
        # For uniform distribution with 4 symbols, entropy should be 2
        assert abs(encoder.entropy - 2.0) < 1e-9

    def test_expected_code_length(self, simple_probabilities):
        """Test expected code length calculation."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        expected = sum(
            p * len(encoder._code_table[sym])
            for sym, p in encoder.probabilities.items()
        )
        assert abs(encoder.expected_code_length - expected) < 1e-9

    def test_coding_efficiency(self, simple_probabilities):
        """Test coding efficiency calculation."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        efficiency = encoder.entropy / encoder.expected_code_length
        assert abs(encoder.coding_efficiency - efficiency) < 1e-9

    def test_coding_efficiency_at_most_one(self, skewed_probabilities):
        """Test that coding efficiency is at most 1."""
        encoder = ShannonFanoEliasBinaryCoder(skewed_probabilities)
        assert encoder.coding_efficiency <= 1.0

    def test_coding_efficiency_at_least_entropy_div_limit(self, simple_probabilities):
        """Test lower bound on coding efficiency."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        # Efficiency should be at least H / (H + 1)
        lower_bound = encoder.entropy / (encoder.entropy + 1)
        assert encoder.coding_efficiency >= lower_bound - 1e-9

    # ============= Code Table Property Tests =============

    def test_code_table_property(self, simple_probabilities):
        """Test that code_table property returns correct structure."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        table = encoder.code_table
        assert table is not None
        assert len(table) == 3
        # Table should map symbols to lists (from interface)
        for sym, code in table.items():
            assert isinstance(code, list)
            assert all(bit in ["0", "1"] for bit in code)

    # ============= Encoding/Decoding Tests =============

    def test_encode_simple_message(self, simple_probabilities):
        """Test encoding a simple message."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        message = ["a", "b", "c", "a"]
        encoded = encoder.encode(message)
        # Encoded message should be a flat sequence of channel symbols
        assert all(bit in ["0", "1"] for bit in encoded)

    def test_encode_empty_message(self, simple_probabilities):
        """Test encoding empty message."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        encoded = encoder.encode([])
        assert encoded == []

    def test_encode_single_symbol(self, simple_probabilities):
        """Test encoding single symbol."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        encoded = encoder.encode(["a"])
        assert len(encoded) > 0
        assert all(bit in ["0", "1"] for bit in encoded)

    def test_encode_repeated_symbols(self, simple_probabilities):
        """Test encoding repeated symbols."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        message = ["a", "a", "a"]
        encoded = encoder.encode(message)
        # Length should be 3 * len(code for 'a')
        expected_length = 3 * len(encoder._code_table["a"])
        assert len(encoded) == expected_length

    def test_encode_symbol_not_in_alphabet(self, simple_probabilities):
        """Test encoding symbol not in alphabet."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        with pytest.raises(ValueError, match="Symbol.*not in source alphabet"):
            encoder.encode(["a", "z", "b"])

    def test_decode_simple_message(self, simple_probabilities):
        """Test decoding a simple message."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        original = ["a", "b", "c", "a"]
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded)
        assert decoded == original

    def test_decode_empty_message(self, simple_probabilities):
        """Test decoding empty message."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        decoded = encoder.decode([])
        assert decoded == []

    def test_roundtrip_encoding_decoding(self, skewed_probabilities):
        """Test roundtrip encoding and decoding."""
        encoder = ShannonFanoEliasBinaryCoder(skewed_probabilities)
        original = ["a", "b", "c", "d", "e", "a", "b", "c"]
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded)
        assert decoded == original

    def test_roundtrip_long_message(self, simple_probabilities):
        """Test roundtrip with longer message."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        original = ["a", "b", "c"] * 100
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded)
        assert decoded == original

    def test_decode_invalid_sequence(self, simple_probabilities):
        """Test decoding invalid bit sequence."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        # A sequence that cannot be decoded (all 1s might fail depending on codes)
        invalid = ["1"] * 1000
        with pytest.raises((ValueError, KeyError)):
            encoder.decode(invalid)

    # ============= Special Distribution Tests =============

    def test_two_symbol_distribution(self, two_symbol_probabilities):
        """Test with two-symbol alphabet."""
        encoder = ShannonFanoEliasBinaryCoder(two_symbol_probabilities)
        assert len(encoder._code_table) == 2
        # Two different symbols should have different codes
        codes = list(encoder._code_table.values())
        assert codes[0] != codes[1]

    def test_uniform_distribution(self, uniform_probabilities):
        """Test with uniform distribution."""
        encoder = ShannonFanoEliasBinaryCoder(uniform_probabilities)
        # With 4 equally likely symbols, entropy should be 2 bits
        assert abs(encoder.entropy - 2.0) < 1e-9

    def test_highly_skewed_distribution(self):
        """Test with highly skewed distribution (one symbol dominates)."""
        probs = OrderedDict([("a", 0.99), ("b", 0.005), ("c", 0.005)])
        encoder = ShannonFanoEliasBinaryCoder(probs)
        # Most frequent symbol should have short code
        assert len(encoder._code_table["a"]) <= len(encoder._code_table["b"])

    def test_many_symbols(self):
        """Test with many symbols."""
        # Create distribution with 10 symbols
        probs = OrderedDict(
            [(chr(ord("a") + i), 1 / 10) for i in range(10)]
        )
        encoder = ShannonFanoEliasBinaryCoder(probs)
        assert len(encoder._code_table) == 10
        # All codes should be valid and prefix-free
        codes = list(encoder._code_strings.values())
        for i, code1 in enumerate(codes):
            for j, code2 in enumerate(codes):
                if i != j:
                    assert not code2.startswith(code1)

    # ============= Comparison with Theoretical Bounds =============

    def test_code_length_satisfies_kraft_inequality(self, simple_probabilities):
        """Test Kraft inequality is satisfied."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        # Sum of 2^(-l_i) should be <= 1
        kraft_sum = sum(2 ** (-len(code)) for code in encoder._code_strings.values())
        assert kraft_sum <= 1.0 + 1e-9

    def test_expected_length_greater_than_entropy(self, simple_probabilities):
        """Test that expected code length >= entropy."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        assert encoder.expected_code_length >= encoder.entropy - 1e-9

    def test_expected_length_less_than_entropy_plus_one(self, simple_probabilities):
        """Test that expected code length < entropy + 1."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        # This is a theoretical property of Shannon codes
        assert encoder.expected_code_length < encoder.entropy + 1 + 1e-6

    # ============= Modified CDF Tests =============

    def test_modified_cdf_in_valid_range(self, simple_probabilities):
        """Test that modified CDF values are in [0, 1]."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        for symbol, f_val in encoder._F.items():
            assert 0 <= f_val <= 1, (
                f"Modified CDF for {symbol!r} out of range: {f_val}"
            )

    def test_modified_cdf_ordering(self, simple_probabilities):
        """Test that modified CDF values are in order of symbols."""
        encoder = ShannonFanoEliasBinaryCoder(simple_probabilities)
        symbols = list(encoder.probabilities.keys())
        f_values = [encoder._F[sym] for sym in symbols]

        # F values should be increasing (symbols in order)
        for i in range(len(f_values) - 1):
            assert f_values[i] < f_values[i + 1], (
                f"Modified CDF values not in order: "
                f"{f_values[i]} >= {f_values[i+1]}"
            )
