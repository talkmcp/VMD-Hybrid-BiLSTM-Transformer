# VMD-Hybrid-BiLSTM-Transformer
VMD–Hybrid BiLSTM–Transformer
VMD–Hybrid BiLSTM–Transformer
Natural Rubber Price Forecasting — Code Repository

Paper: Multi-Scale Forecasting of Natural Rubber Prices Using a VMD–Hybrid BiLSTM–Transformer Framework
Journal: Forecasting (MDPI) — Under Review

Overview
This repository contains model code and test-set predictions for a dual-pathway deep learning framework that integrates Variational Mode Decomposition (VMD) with a Bidirectional LSTM encoder and a Transformer encoder to forecast daily RSS3 FOB price changes in the Thai physical rubber market.
Central contribution: The VMD-as-features design appends all five decomposed IMF components directly to the economic feature matrix, preserving multi-scale frequency information within a single forward pass, rather than forecasting each IMF independently.

Repository Structure
File	Description
step3b_vmd_decomposition.py	VMD decomposition and K selection (K = 5)
step6_stage3_v2.py	Main model: Hybrid BiLSTM-Transformer training
step7_inverse_v3.py	Inverse MinMax transformation and price-level reconstruction from differenced predictions
appendix_F_lookback_sensitivity.py	Lookback window sensitivity (Appendix F)
results/stage3_test_results.csv	Test-set predictions (n = 237 observations)
results/stage1_test_results.csv	Stage 1 out-of-sample results
results/stage2_test_results.csv	Stage 2 out-of-sample results
README.docx	This file


Requirements
Python >= 3.10
torch >= 2.1
numpy, pandas, scikit-learn, vmdpy, openpyxl, statsmodels

Install all dependencies:
pip install torch numpy pandas scikit-learn vmdpy openpyxl statsmodels


Data
The economic feature dataset (24 variables) was obtained from CEIC Data Company Ltd. under a commercial license. Due to licensing restrictions, the raw data cannot be redistributed. Researchers may obtain access directly at ceicdata.com.

Input Feature Description (24 Variables)
#	Variable	Description	Source	Freq.	Transform
1	rss3_FOBm1_diff	RSS3 FOB Thailand (front month) — Target	TRA	Daily	1st difference
2	rss3_FOBm2_diff	RSS3 FOB Thailand (2nd month)	TRA	Daily	1st difference
3	str20_fobm1_diff	STR20 FOB Thailand (front month)	TRA	Daily	1st difference
4	str20_fobm2_diff	STR20 FOB Thailand (2nd month)	TRA	Daily	1st difference
5	latex_fobm1_diff	Latex FOB Thailand (front month)	TRA	Daily	1st difference
6	latex_fobm2_diff	Latex FOB Thailand (2nd month)	TRA	Daily	1st difference
7	rss3_JPXm1_diff	RSS3 JPX Japan futures (front month)	JPX	Daily	1st difference
8	rss3_shfe_m1_diff	RSS3 SHFE Shanghai futures (front month)	SHFE	Daily	1st difference
9	rss3_shfe_m2_diff	RSS3 SHFE Shanghai futures (2nd month)	SHFE	Daily	1st difference
10	rss3_sgxsett_diff	RSS3 SGX Singapore settlement price	SGX	Daily	1st difference
11	tsr20_sgxsett_diff	TSR20 SGX Singapore settlement price	SGX	Daily	1st difference
12	CupLump_diff	Cup lump local price (Thailand)	TRA	Daily	1st difference
13	uss	Unsmoked Sheet local price	TRA	Daily	Level (stationary)
14	usd_thb_diff	USD/THB exchange rate	Bloomberg	Daily	1st difference
15	cny_thb_diff	CNY/THB exchange rate	Bloomberg	Daily	1st difference
16	usd_cny_diff	USD/CNY exchange rate	Bloomberg	Daily	1st difference
17	brent_usd_diff	Brent crude oil price (USD/bbl)	CEIC	Daily	1st difference
18	wti_usd_diff	WTI crude oil price (USD/bbl)	CEIC	Daily	1st difference
19	brent_return	Brent daily log return	CEIC	Daily	Derived
20	brent_lag1_diff	Brent price lagged 1 day	CEIC	Daily	1st difference
21	china_pmi_mfg	China Manufacturing PMI	CEIC	Monthly	Level (interpolated)
22	bdi	Baltic Dry Index	CEIC	Daily	Level (stationary)
23	enso_oni_diff	ENSO Oceanic Nino Index	NOAA	Monthly	1st difference
24	covid_period_diff	COVID-19 regime dummy	WHO	Daily	1st difference

Note: Features 1–20 are daily. Features 21–23 are monthly series interpolated to daily frequency using forward-fill. ADF unit root test results are reported in Appendix B of the paper.


Data Source Access
•	Public / Free Access
Source	Data Used	URL
NOAA	ENSO Oceanic Nino Index (ONI)	psl.noaa.gov/enso/mei
EIA	Brent crude oil, WTI crude oil	eia.gov/dnav/pet
WHO	COVID-19 regime dummy	who.int
TRA (Thai Rubber Association)	RSS3 FOB, STR20, Latex, CupLump, USS	rubberthai.com

•	Subscription / Paid
Source	Data Used
CEIC	CNY/THB, USD/CNY, China PMI Manufacturing, Baltic Dry Index
Bloomberg	USD/THB, CNY/THB, USD/CNY exchange rates
Reuters	Brent crude oil (partial)

•	Semi-public (Registration Required, Free Plan Available)
Source	Data Used
JPX (Japan Exchange Group)	RSS3 JPX futures (front month)
SHFE (Shanghai Futures Exchange)	RSS3 SHFE futures (m1, m2)
SGX (Singapore Exchange)	RSS3 SGX settlement, TSR20 SGX settlement



Reproducing Results
Step 1 — VMD Decomposition
python step3b_vmd_decomposition.py
Runs a grid search over K ∈ {4, 5, 6, 7, 8} and selects K = 5 based on reconstruction RMSE. Outputs five IMF columns appended to the feature matrix.
Step 2 — Train Main Model
python step6_stage3_v2.py
Trains the Hybrid BiLSTM-Transformer on the Stage 3 training partition (n = 2,140, 2018–2023).

Key hyperparameters:
Hyperparameter	Value	Rationale
Lookback window (L)	30 days	Sensitivity reported in Appendix F
Loss function	Huber (d = 0.5)	Robust to outlier price spikes
Optimiser	AdamW (lr = 5e-4)	Weight decay = 1e-4
LR scheduler	ReduceLROnPlateau x0.5	Patience = 10 epochs
Early stopping	Patience = 30 epochs	Best validation loss
Max epochs	300	Upper bound
Batch size	32	Mini-batch SGD
Total parameters	1,053,698	

Step 3 — Lookback Sensitivity (Appendix F)
python appendix_F_lookback_sensitivity.py
Evaluates model performance across L ∈ {10, 20, 30, 45, 60} trading days.


Test-Set Results (Stage 3, n = 237)
Metric	VMD–Hybrid	Best Baseline (ARIMA)	Improvement
Pearson r	0.812	0.150	+0.662
Directional Accuracy	67.1%	54.5%	+12.6 pp
MAE (THB/kg/day)	0.156	0.249	-37.3%
Std Ratio (StdR)	0.819	0.412	+0.407

Pre-computed predictions are available in results/stage3_test_results.csv for verification without retraining.


Data Availability Statement
The economic feature data used in this study were obtained from CEIC Data Company Ltd. under a commercial institutional license. These data cannot be publicly redistributed. Researchers may obtain access directly at ceicdata.com. All model code and test-set predictions are made openly available in this repository to support reproducibility within the constraints of the data license.


Citation
If you use this code, please cite:
@article{pinitjitsamut2026rubber,
  title   = {Multi-Scale Forecasting of Natural Rubber Prices Using a
             VMD--Hybrid BiLSTM--Transformer Framework},
  author  = {Pinitjitsamut, Montchai}
  journal = {Forecasting},
  year    = {2026},
  note    = {Under review}
}


License
Code is released under the MIT License. Data from CEIC Data Company Ltd. is subject to the provider's terms of use and is not included in this repository.
