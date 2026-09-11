from app.dataset import TimeSeriesDataset, create_windows


def test_create_windows_slices_correctly():
    series = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    X, y = create_windows(series, window_size=3)

    assert X == [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]
    assert y == [4.0, 5.0, 6.0]


def test_create_windows_with_series_shorter_than_window_is_empty():
    X, y = create_windows([1.0, 2.0], window_size=5)
    assert X == []
    assert y == []


def test_time_series_dataset_shapes():
    X, y = create_windows([1.0, 2.0, 3.0, 4.0, 5.0], window_size=2)
    dataset = TimeSeriesDataset(X, y)

    assert len(dataset) == 3
    x_sample, y_sample = dataset[0]
    assert tuple(x_sample.shape) == (2, 1)
    assert tuple(y_sample.shape) == (1,)
