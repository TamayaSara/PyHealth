Tasks
===============

We support various real-world healthcare predictive tasks defined by **function calls**. The following example tasks are collected from top AI/Medical venues, such as:

(i) Drug Recommendation [Yang et al. IJCAI 2021a, Yang et al. IJCAI 2021b, Shang et al. AAAI 2020]

(ii) Readmission Prediction [Choi et al. AAAI 2021]

(iii) Mortality Prediction [Choi et al. AAAI 2021]

(iv) Length of Stay Prediction

(v) Sleep Staging [Yang et al. ArXix 2021]

Getting Started
---------------

New to PyHealth tasks? Start here:

- **Browse tasks**: `pyhealth.dev/tasks <https://pyhealth.dev/tasks.html>`_ lists every task with the dataset it runs on. Each task page below also links to its dataset, and each dataset page lists its tasks
- **Tutorial**: `Introduction to pyhealth.tasks <https://colab.research.google.com/drive/1QB0acnGb-wOuK53UNSgHxjCW74QeYjUl?usp=sharing>`_ - Learn the basics of defining and using tasks
- **Code Examples**: Browse all examples online at https://github.com/sunlabuiuc/PyHealth/tree/master/examples
- **Pipeline Examples**: Check out our :doc:`../tutorials` page for complete end-to-end examples including:

  - Mortality Prediction Pipeline
  - Readmission Prediction Pipeline
  - Medical Coding Pipeline
  - Chest X-ray Classification Pipeline

These tutorials demonstrate how to load datasets, apply tasks, train models, and evaluate results.

Understanding Tasks and Processors
-----------------------------------

Tasks define **what** data to extract (via ``input_schema`` and ``output_schema``), while **processors** define **how** to transform that data into tensors for model training.

After you define a task:

1. **Task execution**: The task function extracts relevant features from patient records and generates samples
2. **Processor application**: Processors automatically transform these samples into model-ready tensors based on the schemas

**Example workflow:**

.. code-block:: python

    # 1. Define a task with input/output schemas
    task = MortalityPredictionMIMIC4()
    # input_schema = {"conditions": "sequence", "procedures": "sequence"}
    # output_schema = {"mortality": "binary"}

    # 2. Apply task to dataset
    sample_dataset = base_dataset.set_task(task)

    # 3. Processors automatically transform samples:
    #    - "sequence" -> SequenceProcessor (converts codes to indices)
    #    - "binary" -> BinaryLabelProcessor (converts labels to tensors)

    # 4. Get model-ready tensors
    sample = sample_dataset[0]
    # sample["conditions"] is now a tensor of token indices
    # sample["mortality"] is now a binary tensor [0] or [1]

**Learn more about processors:**

- See the :doc:`processors` documentation for details on all available processors
- Learn about string keys (``"sequence"``, ``"binary"``, etc.) that map to specific processors
- Discover how to customize processor behavior with kwargs tuples
- Understand processor types for different data modalities (text, images, signals, etc.)

WESAD wrist EDA stress classification
--------------------------------------

The WESAD adapter indexes each participant's ``WESAD.zip`` member without
extracting it. ``WESADStressClassification`` reads the original recording
pickle directly from the ZIP using the event's ``archive_path`` and
``member_path``. It aligns each wrist EDA sample at 4 Hz with 175 recording
labels at 700 Hz and creates non-overlapping windows of configurable duration.

.. code-block:: python

    from pyhealth.datasets import WESADDataset
    from pyhealth.tasks import WESADStressClassification

    dataset = WESADDataset(root="data/wesad")
    samples = dataset.set_task(WESADStressClassification(window_seconds=60))

Label code 1 means baseline and code 2 means stress. A window is kept only
when all its 700 Hz labels have the same target code. Mixed-label windows
have no single state across the full interval, and codes 0 and 3–7 fall
outside the two target classes. Incomplete windows lack enough aligned EDA
and labels to fill the configured duration.

Each window contains a raw wrist EDA tensor of shape ``(1, 240)`` at the
default 60-second duration. The task emits a ``baseline`` or ``stress``
label and preserves ``patient_id`` for participant-level splits.
``set_task`` encodes the label using the fitted multiclass processor.

Validation
~~~~~~~~~~

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

Writing a Custom Task
----------------------

When a built-in task doesn't match your cohort or prediction target, you can
define your own by subclassing :class:`~pyhealth.tasks.BaseTask`. The class
needs three things: a name, input and output schemas, and a ``__call__``
method that processes one patient at a time.

.. code-block:: python

    from pyhealth.tasks import BaseTask
    from pyhealth.data import Patient
    from typing import List, Dict, Any

    class MyMortalityTask(BaseTask):
        task_name: str = "MyMortalityTask"

        input_schema: Dict[str, str] = {
            "conditions": "sequence",   # maps to SequenceProcessor
            "procedures": "sequence",
        }
        output_schema: Dict[str, str] = {
            "label": "binary"           # maps to BinaryLabelProcessor
        }

        def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
            samples = []
            for adm in patient.get_events("admissions"):
                label = 1 if adm.hospital_expire_flag == "1" else 0

                # Fetch historical diagnoses up to this admission
                conditions = patient.get_events("diagnoses_icd", end=adm.timestamp)
                cond_codes = [e.icd_code for e in conditions]

                if not cond_codes:
                    continue

                samples.append({
                    "conditions": [cond_codes],  # wrapped in a list for the sequence processor
                    "procedures": [[]],
                    "label": label,
                })
            return samples

The ``__call__`` method receives one ``Patient`` and returns a list of sample
dictionaries. Each dictionary's keys should match the schemas you declared.
Returning an empty list is fine — PyHealth simply skips that patient. Note
that event attribute names are always lowercase (e.g. ``e.icd_code`` rather
than ``e.ICD_CODE``) because PyHealth lowercases all column names at ingest
time. Timestamps are accessed through ``event.timestamp`` rather than the
original column name like ``charttime``, since PyHealth normalises them into
a single property.

Processor String Keys
----------------------

The string values in your schemas map to specific processor classes. Here is
a quick reference:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - String key
     - Processor
     - Typical use
   * - ``"sequence"``
     - ``SequenceProcessor``
     - Diagnosis codes, procedure codes, drug names
   * - ``"nested_sequence"``
     - ``NestedSequenceProcessor``
     - Cumulative visit history (drug recommendation, readmission)
   * - ``"nested_multihot"``
     - ``NestedMultiHotProcessor``
     - Per-visit code sets (HALO; see ``EHRGenerationMIMIC3``)
   * - ``"tensor"``
     - ``TensorProcessor``
     - Aggregated numeric values (e.g. last lab value per item)
   * - ``"timeseries"``
     - ``TimeseriesProcessor``
     - Irregular time-series measurements
   * - ``"multi_hot"``
     - ``MultiHotProcessor``
     - Demographics, comorbidity flags
   * - ``"text"``
     - ``TextProcessor``
     - Clinical notes
   * - ``"binary"``
     - ``BinaryLabelProcessor``
     - Binary classification label (0 / 1)
   * - ``"multiclass"``
     - ``MultiClassLabelProcessor``
     - Multi-class label
   * - ``"multilabel"``
     - ``MultiLabelProcessor``
     - Multi-label classification
   * - ``"regression"``
     - ``RegressionLabelProcessor``
     - Continuous regression target

How set_task() Works
---------------------

Calling ``dataset.set_task(task)`` iterates over every patient in the
dataset, runs your ``__call__`` method on each one, fits all the processors
on the collected samples, then serialises everything to disk as LitData
``.ld`` files. The result is a :class:`~pyhealth.datasets.SampleDataset` that
supports ``len()`` and index access, ready for a DataLoader.

.. code-block:: python

    sample_ds = dataset.set_task(MyMortalityTask(), num_workers=4)
    len(sample_ds)   # total ML samples across all patients
    sample_ds[0]     # a single sample dict with tensor values

If you re-run ``set_task()`` with the same task and processor configuration,
PyHealth detects the existing cache and skips reprocessing. The cache key holds
the task's arguments and schemas but not its code, so after you edit a task's
``__call__``, run ``dataset.clear_task_cache(task)`` (see *Task caches and when
to clear them* in :doc:`datasets`). During
development it is useful to set ``dev=True`` on the dataset, which limits
processing to 1 000 patients so iterations are fast.

.. note::

   **A note on multiprocessing.** ``set_task()`` can spawn worker processes
   when ``num_workers > 1``. On macOS and Linux this requires the standard
   Python multiprocessing guard around your top-level script:

   .. code-block:: python

       if __name__ == '__main__':
           sample_ds = dataset.set_task(task, num_workers=4)

   Without this guard, Python may try to re-import and re-run the script in
   each worker process, leading to infinite recursion. This is a general
   Python multiprocessing requirement, not specific to PyHealth.

Available Tasks
---------------

.. toctree::
    :maxdepth: 3

    Base Task <tasks/pyhealth.tasks.BaseTask>
    Task Card <tasks/pyhealth.tasks.TaskCard>
    PBC Competing Risks <tasks/pyhealth.tasks.competing_risks_pbc>
    Competing-Risk Preprocessing <tasks/pyhealth.tasks.competing_risks>
    In-Hospital Mortality (MIMIC-IV) <tasks/pyhealth.tasks.InHospitalMortalityMIMIC4>
    In-Hospital Mortality (MEDS) <tasks/pyhealth.tasks.InHospitalMortalityMEDS>
    ICU Admission (MEDS) <tasks/pyhealth.tasks.ICUAdmissionMEDS>
    Hospital Readmission (MEDS) <tasks/pyhealth.tasks.HospitalReadmissionMEDS>
    Prolonged Length of Stay (MEDS) <tasks/pyhealth.tasks.ProlongedLengthOfStayMEDS>
    ETHOS Timelines (MEDS) <tasks/pyhealth.tasks.EthosTimelineMEDS>
    ETHOS Zero-Shot Outcomes (MEDS) <tasks/pyhealth.tasks.EthosOutcomeMEDS>
    MIMIC-III ICD-9 Coding <tasks/pyhealth.tasks.MIMIC3ICD9Coding>
    BMD-HS Disease Classification <tasks/pyhealth.tasks.bmd_hs_disease_classification>
    Cardiology Detection <tasks/pyhealth.tasks.cardiology_detect>
    COVID-19 CXR Classification <tasks/pyhealth.tasks.COVID19CXRClassification>
    COVID-19 CXR Image Generation <tasks/pyhealth.tasks.COVID19CXRImageGeneration>
    PTB-XL Multilabel Classification <tasks/pyhealth.tasks.PTBXLMultilabelClassification>
    PTB-XL HeartLang Classification <tasks/pyhealth.tasks.PTBXLHeartLangClassification>
    PhysioNet 2012 Forecasting <tasks/pyhealth.tasks.PhysioNet2012Forecasting>
    Disease Progression (MIMIC-IV + Chest ImaGenome) <tasks/pyhealth.tasks.DiseaseProgressionMIMIC4>
    DKA Prediction (MIMIC-IV) <tasks/pyhealth.tasks.dka>
    Drug Recommendation <tasks/pyhealth.tasks.drug_recommendation>
    EHR + CXR In-Hospital Mortality (MIMIC-IV) <tasks/pyhealth.tasks.EHRCXRMortalityMIMIC4>
    EHR + CXR Phenotyping (MIMIC-IV) <tasks/pyhealth.tasks.EHRCXRPhenotypingMIMIC4>
    EHR Generation <tasks/pyhealth.tasks.generate_ehr>
    Length of Stay Prediction <tasks/pyhealth.tasks.length_of_stay_prediction>
    Length of Stay Prediction (StageNet MIMIC-IV) <tasks/pyhealth.tasks.length_of_stay_stagenet_mimic4>
    Medical Transcriptions Classification <tasks/pyhealth.tasks.MedicalTranscriptionsClassification>
    MPF Clinical Prediction (FHIR) <tasks/pyhealth.tasks.mpf_clinical_prediction>
    Mortality Prediction (Next Visit) <tasks/pyhealth.tasks.mortality_prediction>
    Mortality Prediction (StageNet MIMIC-IV) <tasks/pyhealth.tasks.mortality_prediction_stagenet_mimic4>
    Next-Visit Diagnosis Prediction <tasks/pyhealth.tasks.next_visit_diagnosis_prediction>
    Patient Linkage (MIMIC-III, legacy) <tasks/pyhealth.tasks.patient_linkage_mimic3_fn>
    Patient Linkage (MIMIC-III) <tasks/pyhealth.tasks.patient_linkage_mimic3>
    Readmission Prediction <tasks/pyhealth.tasks.readmission_prediction>
    Sleep Staging <tasks/pyhealth.tasks.sleep_staging>
    Sleep Staging (SleepEDF) <tasks/pyhealth.tasks.SleepStagingSleepEDF>
    Sleep Staging Sequences (SleepEDF) <tasks/pyhealth.tasks.SleepStagingSequenceSleepEDF>
    Sleep Staging (ISRUC) <tasks/pyhealth.tasks.SleepStagingISRUC>
    Temple University EEG Tasks <tasks/pyhealth.tasks.temple_university_EEG_tasks>
    CHB-MIT Seizure Detection <tasks/pyhealth.tasks.SeizureDetectionCHBMIT>
    Ear-EEG Sleep Staging (EESM23) <tasks/pyhealth.tasks.SleepStagingEarEEG>
    IIIC Seizure Classification <tasks/pyhealth.tasks.iiic_seizure_classification>
    EEGBCI Tasks <tasks/pyhealth.tasks.eegbci>
    PTB-XL Tasks <tasks/pyhealth.tasks.ptbxl>
    Sleep Staging v2 <tasks/pyhealth.tasks.sleep_staging_v2>
    Benchmark EHRShot <tasks/pyhealth.tasks.benchmark_ehrshot>
    EHR Text Serialization (LLM encoders) <tasks/pyhealth.tasks.ehr_text_serialization>
    ED Disposition and Outcomes (MIMIC-IV-ED) <tasks/pyhealth.tasks.ed_disposition_mimic4>
    ChestX-ray14 Binary Classification <tasks/pyhealth.tasks.ChestXray14BinaryClassification>
    De-Identification NER <tasks/pyhealth.tasks.DeIDNERTask>
    ChestX-ray14 Multilabel Classification <tasks/pyhealth.tasks.ChestXray14MultilabelClassification>
    MSD Segmentation <tasks/pyhealth.tasks.MSDSegmentationTask>
    CHAOS Segmentation Tasks <tasks/pyhealth.tasks.CHAOSSegmentation>
    RSNA Pneumonia Classification <tasks/pyhealth.tasks.RSNAPneumoniaClassification>
    Variant Classification (ClinVar) <tasks/pyhealth.tasks.VariantClassificationClinVar>
    Mutation Pathogenicity (COSMIC) <tasks/pyhealth.tasks.MutationPathogenicityPrediction>
    Cancer Survival Prediction (TCGA) <tasks/pyhealth.tasks.CancerSurvivalPrediction>
    In-Hospital Survival (MIMIC-IV) <tasks/pyhealth.tasks.InHospitalSurvivalMIMIC4>
    METABRIC Survival Prediction <tasks/pyhealth.tasks.MetabricSurvivalPrediction>
    Rare-Condition ICU Mortality (KnowRare, MIMIC-IV) <tasks/pyhealth.tasks.KnowRareICUMortalityMIMIC4>
    Lab Nowcasting (MIMIC-III/IV) <tasks/pyhealth.tasks.LabNowcasting>
    ICU Acuity and Therapies (MIMIC-IV) <tasks/pyhealth.tasks.ICUAcuityMIMIC4>
    Cancer Mutation Burden (TCGA) <tasks/pyhealth.tasks.CancerMutationBurden>
    Survival Preprocessing (SUPPORT2) <tasks/pyhealth.tasks.SurvivalPreprocessSupport2>
    YAIB Tasks (MIMIC-IV) <tasks/pyhealth.tasks.yaib_mimic4>
    YAIB Tasks (MIMIC-III) <tasks/pyhealth.tasks.yaib_mimic3>
    YAIB Tasks (eICU) <tasks/pyhealth.tasks.yaib_eicu>
    Value Set Completion (VSAC) <tasks/pyhealth.tasks.ValueSetCompletion>
    Time-to-event survival (GBSG and SUPPORT) <tasks/pyhealth.tasks.survival_time_to_event>
