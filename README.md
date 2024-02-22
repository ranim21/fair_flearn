# Fair Resource Allocation in Federated Learning

## Preparation

### Download Dependencies

```bash
pip3 install -r requirements.txt
```

## Create the vehicle database 

1. Import the vehicle data:

   - Place the `vehicle.mat` file in the appropriate directory.

2. Generate data using the Python script `create_dataset.py`:

   ```bash
   python create_dataset.py
   ```

## Experiments for Fairness Verification

Verify the fairness of the q-FFL (Quantile Federated Averaging) objective and compare it with uniform sampling schemes using the following commands:

```bash
mkdir log_vehicle
bash run.sh vehicle qffedavg 1 0 2 | tee log_vehicle/ffedavg_run1_q0
bash run.sh vehicle qffedavg 1 5 2 | tee log_vehicle/ffedavg_run1_q5
bash run.sh vehicle qffedavg 1 0 1 | tee log_vehicle/fedavg_uniform_run1
```

### Plot to Reproduce Results

To reproduce the results presented in the manuscript, use the following commands:

```bash
pip install seaborn
python plot_fairness.py
```

Compare the generated `fairness_vehicle.pdf` with Figure 1 (the Vehicle subfigure) and Figure 2 (the Vehicle subfigure) in the paper to validate reproducibility. Note that the reported accuracy distributions are averaged across 5 different train/test/validation data partitions with data partition seeds 1, 2, 3, 4, and 5.

## Experiments for Communication-Efficiency

Demonstrate the communication-efficiency of the proposed method q-FedAvg with the following command:

```bash
bash run.sh vehicle qffedsgd 1 5 2 | tee log_vehicle/ffedsgd_run1_q5
```

### Plot to Reproduce Results

```bash
mkdir log_$dataset
bash run.sh $dataset $method $seed $q $sampling | tee log_$dataset/$method_run$seed_q$q
```

Replace `$dataset` with one of `[vehicle, synthetic, sent140, shakespeare]` to match the data directory names under the `fair_flearn/data/` folder.
