from autos.normalize import as_int, as_number, clean_price_and_currency, parse_motor, parse_tecnico, _slug


def test_as_number_miles_argentinos():
    assert as_number("$ 12.500.000") == 12500000.0
    assert as_number("15,5") == 15.5
    assert as_number("") is None and as_number(None) is None
    assert as_int("118.545 km") == 118545


def test_tecnico_y_motor():
    assert parse_tecnico("1,6") == 1.6
    assert parse_tecnico("120 CV") == 120.0
    assert parse_motor(1600) == 1.6
    assert parse_motor("1.7 lts") == 1.7
    assert parse_motor(None) is None and parse_motor("0") is None


def test_precio_y_moneda():
    assert clean_price_and_currency("U$S 15.000") == (15000.0, "USD")
    assert clean_price_and_currency("$ 21.500.000") == (21500000.0, "ARS")
    assert clean_price_and_currency("Consultar") == (None, "ARS")


def test_slug():
    assert _slug("Citroën DS") == "citroen-ds"
