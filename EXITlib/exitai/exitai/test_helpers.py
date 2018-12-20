from helpers import get_value_in_log_scale


def test_first_func():
    # if index is divisible
    assert get_value_in_log_scale(
        vals=(10, 20), current_index=0, length=2) == 10
    assert get_value_in_log_scale(
        vals=(10, 20), current_index=3, length=4) == 20

    # not divisible
    assert get_value_in_log_scale(
        vals=(0.0001, 0.1), current_index=1, length=4) == 0.001
    assert get_value_in_log_scale(
        vals=(10, 1000), current_index=1, length=3) == 100
