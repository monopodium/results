import math

FP8_E5M2_BIAS = 15

def float_to_fp8_e5m2(x: float):
    """
    Encode a Python float into a simplified FP8 E5M2 format.

    Format:
      - 1 sign bit
      - 5 exponent bits
      - 2 mantissa bits
      - exponent bias = 15

    Returns:
      dict with sign/exponent/mantissa fields and final 8-bit encoding.
    """
    # Sign bit
    sign = 0 if math.copysign(1.0, x) > 0 else 1

    # Special case: zero
    if x == 0.0:
        exp_bits = 0
        mant_bits = 0
        bits = (sign << 7) | (exp_bits << 2) | mant_bits
        return {
            "value": x,
            "sign": sign,
            "exponent_bits": format(exp_bits, "05b"),
            "mantissa_bits": format(mant_bits, "02b"),
            "encoding_bits": format(bits, "08b"),
            "encoding_hex": f"0x{bits:02X}",
            "note": "zero"
        }

    ax = abs(x)

    # Handle inf / nan for demonstration
    if math.isinf(ax):
        exp_bits = 0b11111
        mant_bits = 0
        bits = (sign << 7) | (exp_bits << 2) | mant_bits
        return {
            "value": x,
            "sign": sign,
            "exponent_bits": format(exp_bits, "05b"),
            "mantissa_bits": format(mant_bits, "02b"),
            "encoding_bits": format(bits, "08b"),
            "encoding_hex": f"0x{bits:02X}",
            "note": "infinity"
        }

    if math.isnan(x):
        exp_bits = 0b11111
        mant_bits = 1  # simple quiet-NaN example
        bits = (sign << 7) | (exp_bits << 2) | mant_bits
        return {
            "value": x,
            "sign": sign,
            "exponent_bits": format(exp_bits, "05b"),
            "mantissa_bits": format(mant_bits, "02b"),
            "encoding_bits": format(bits, "08b"),
            "encoding_hex": f"0x{bits:02X}",
            "note": "nan"
        }

    # Normal-number path
    e = math.floor(math.log2(ax))
    normalized = ax / (2 ** e)   # in [1, 2)

    # Fractional part after removing implicit leading 1
    frac = normalized - 1.0

    # Quantize to 2 mantissa bits
    mant = round(frac * (2 ** 2))

    # Handle carry from rounding, e.g. 1.11... rounds to 10.00
    if mant == 4:
        mant = 0
        e += 1

    exp = e + FP8_E5M2_BIAS

    # Overflow -> infinity
    if exp >= 0b11111:
        exp_bits = 0b11111
        mant_bits = 0
        bits = (sign << 7) | (exp_bits << 2) | mant_bits
        return {
            "value": x,
            "sign": sign,
            "exponent_bits": format(exp_bits, "05b"),
            "mantissa_bits": format(mant_bits, "02b"),
            "encoding_bits": format(bits, "08b"),
            "encoding_hex": f"0x{bits:02X}",
            "note": "overflow -> infinity"
        }

    # Subnormal / underflow handling (simplified)
    if exp <= 0:
        # Subnormal value = mantissa * 2^(1-bias-2)
        # Here we do a simple quantization for demonstration.
        scaled = ax / (2 ** (1 - FP8_E5M2_BIAS - 2))
        mant_bits = round(scaled)

        if mant_bits <= 0:
            exp_bits = 0
            mant_bits = 0
            note = "underflow -> zero"
        elif mant_bits >= 4:
            exp_bits = 1
            mant_bits = 0
            note = "rounded into smallest normal"
        else:
            exp_bits = 0
            note = "subnormal"

        bits = (sign << 7) | (exp_bits << 2) | mant_bits
        return {
            "value": x,
            "sign": sign,
            "exponent_bits": format(exp_bits, "05b"),
            "mantissa_bits": format(mant_bits, "02b"),
            "encoding_bits": format(bits, "08b"),
            "encoding_hex": f"0x{bits:02X}",
            "note": note
        }

    exp_bits = exp
    mant_bits = mant
    bits = (sign << 7) | (exp_bits << 2) | mant_bits

    return {
        "value": x,
        "sign": sign,
        "exponent_bits": format(exp_bits, "05b"),
        "mantissa_bits": format(mant_bits, "02b"),
        "encoding_bits": format(bits, "08b"),
        "encoding_hex": f"0x{bits:02X}",
        "note": "normal"
    }


def pretty_print_fp8(x: float):
    result = float_to_fp8_e5m2(x)
    print(f"value:          {result['value']}")
    print(f"sign:           {result['sign']}")
    print(f"exponent bits:  {result['exponent_bits']}")
    print(f"mantissa bits:  {result['mantissa_bits']}")
    print(f"encoding bits:  {result['encoding_bits']}")
    print(f"encoding hex:   {result['encoding_hex']}")
    print(f"note:           {result['note']}")
    print("-" * 40)


if __name__ == "__main__":
    test_values = [
        0.0,
        1.0,
        -1.0,
        0.75,
        1.5,
        3.0,
        10.0,
        0.1,
        1e-4,
        float("inf"),
        float("nan"),
    ]

    for v in test_values:
        pretty_print_fp8(v)