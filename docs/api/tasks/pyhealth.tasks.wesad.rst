pyhealth.tasks.wesad
====================

Overview
--------

:class:`~pyhealth.tasks.wesad.WESADStressClassification` extracts labeled
wrist electrodermal activity (EDA) windows from
:class:`~pyhealth.datasets.WESADDataset`. The adapter indexes one ``eda``
event per participant without extracting the archive. The task reads the
recording pickle using the event's ``archive_path`` and ``member_path`` and
uses ``signal["wrist"]["EDA"]`` and ``label`` from the recording.

Parameters and windowing
------------------------

``window_seconds`` (float, default 60) is the only constructor parameter.
It must be finite, positive, and represent a whole number of samples at
4 Hz. For example, 30 and 60 seconds produce 120 and 240 EDA samples,
respectively.

The task divides each recording into consecutive, non-overlapping windows
starting at recording time zero. Each 4 Hz EDA sample aligns to 175
consecutive labels at 700 Hz. The task uses the shared aligned extent,
tolerates a difference of up to one EDA sample between the signal length
and the label-derived EDA length, and raises ``ValueError`` for a larger
difference. Incomplete trailing intervals and partial windows are dropped.

A window is retained when all its source labels are code 1 (``baseline``)
or all are code 2 (``stress``). Mixed-label windows and windows containing
codes 0 or 3-7 are excluded. Invalid recordings, including non-finite EDA
or labels and unsupported label codes, raise ``ValueError``.

Sample output
-------------

* ``signal`` is raw wrist EDA in the recorded units, as a float32 tensor
  with shape ``(1, window_samples)``. The input schema uses the tensor
  processor.
* ``label`` is the string ``baseline`` or ``stress`` in the raw task output.
  After ``set_task()``, the multiclass processor encodes it using its fitted
  vocabulary. Read ``samples.output_processors["label"].label_vocab`` to
  interpret the numeric IDs.
* ``patient_id`` identifies the participant, and ``record_id`` identifies the
  recording. Use ``patient_id`` to keep a participant's windows together
  when splitting data.
* ``window_index`` is the zero-based position on the original window grid.
  ``start_seconds`` is the recording-relative start time. Excluded windows
  can leave gaps in the retained indices.

Usage
-----

Pass the directory containing the original ``WESAD.zip`` to the dataset
adapter and apply the task with ``dataset.set_task()``.

.. code-block:: python

    from pyhealth.datasets import WESADDataset
    from pyhealth.tasks import WESADStressClassification

    dataset = WESADDataset(root="data/wesad")
    samples = dataset.set_task(WESADStressClassification(window_seconds=60))

API reference
-------------

.. automodule:: pyhealth.tasks.wesad
    :members:
    :undoc-members:
    :show-inheritance:



Validation
----------

All 12 WESAD adapter and task tests passed on Windows using Pixi. Real-data
validation covered all 15 participants using complete, non-overlapping
60-second windows. The task produced 434 windows (282 baseline and 152
stress). Every signal had shape ``(1, 240)`` and all values were finite.
These results describe data validation. The run did not measure model performance.

Run the combined tests from the repository root:

.. code-block:: powershell

    pixi run -e test python -m unittest discover -s tests/core -p "test_wesad*.py" -v

To repeat the real-data validation, save this example as
``validate_wesad.py``. Pass the directory containing ``WESAD.zip`` and a
writable cache directory. Keep the archive zipped. The script uses one
worker and checks every signal before printing the label counts.

.. code-block:: python

    import argparse
    from collections import Counter

    from pyhealth.datasets import WESADDataset
    from pyhealth.tasks import WESADStressClassification


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset-root", required=True)
        parser.add_argument("--cache-dir", required=True)
        args = parser.parse_args()

        dataset = WESADDataset(root=args.dataset_root, cache_dir=args.cache_dir)
        task = WESADStressClassification(window_seconds=60)
        counts = Counter()
        with dataset.set_task(task, num_workers=1) as samples:
            vocab = samples.output_processors["label"].label_vocab
            labels = {index: label for label, index in vocab.items()}
            try:
                for index in range(len(samples)):
                    sample = samples[index]
                    signal = sample["signal"]
                    if tuple(signal.shape) != (1, 240) or not signal.isfinite().all().item():
                        raise ValueError(f"Invalid signal in window {index}")
                    counts[labels[sample["label"].item()]] += 1
            finally:
                samples.reset()
        print(f"Participants: {len(dataset.unique_patient_ids)}")
        print(f"{counts.total()} finite windows of shape (1, 240)")
        print(f"Baseline: {counts['baseline']}, stress: {counts['stress']}")


    if __name__ == "__main__":
        main()

.. code-block:: powershell

    pixi run -e test python validate_wesad.py --dataset-root "D:\Datasets\WESAD" --cache-dir "D:\WESAD-cache"
