"""Consumer lag on one partition, from its committed offset and watermarks."""

from broker.rollouts import partition_lag


def test_lag_is_the_messages_after_the_committed_offset():
    assert partition_lag(committed=7, low=2, high=10) == 3


def test_a_partition_the_group_never_committed_on_lags_by_everything_retained():
    assert partition_lag(committed=-1001, low=2, high=10) == 8


def test_messages_removed_by_retention_are_not_counted_as_lag():
    assert partition_lag(committed=1, low=4, high=10) == 6
