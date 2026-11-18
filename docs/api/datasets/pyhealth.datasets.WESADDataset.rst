pyhealth.datasets.WESADDataset
=============================

Place the `WESAD archive <https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html>`_
at ``root/WESAD.zip``. Each participant has one ``eda`` event with ``record_id``,
``archive_path`` and ``member_path``. The adapter does not load signals.

Initialization rebuilds the index and invalidates processed caches. Metadata
falls back to PyHealth's home cache if the data directory is not writable.

.. autoclass:: pyhealth.datasets.WESADDataset
    :members:
    :undoc-members:
    :show-inheritance:
