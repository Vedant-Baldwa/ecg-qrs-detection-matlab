% run_project.m
clc; clear; close all;
project_root = '/MATLAB Drive/DSP Project';
data_dir = fullfile(project_root,'data');
code_dir = fullfile(project_root,'code');
results_dir = fullfile(project_root,'results');
if ~exist(results_dir,'dir'), mkdir(results_dir); end

fprintf('Running ECG project...\n');

% --- Load CSV (mitbih_train.csv expected) ---
csvfile = fullfile(data_dir,'mitbih_train.csv');
if ~exist(csvfile,'file')
    error('Put mitbih_train.csv into the data/ folder and re-run.');
end
M = readmatrix(csvfile);
[nrows,ncols] = size(M);
siglen = ncols - 1;
snips = double(M(:,1:siglen));
labels = M(:,end);

% --- Detection (calls detect_r_peaks) ---
params.fs = 360;        
params.bp_low = 5; params.bp_high = 25; params.bp_order = 2;
params.prom_scale = 0.25;
params.min_dist_ms = 120; 
params.center_window_ms = 80;
% run detection
[r_locs, r_vals, peak_found] = detect_r_peaks(snips, params);

% Save detection
save(fullfile(results_dir,'final_detection.mat'),'r_locs','r_vals','peak_found','labels','params','-v7.3');

% --- Analysis & plots (calls analyze_and_plot) ---
analyze_and_plot(snips, labels, r_locs, results_dir, params);

fprintf('Done. Results & figures are in %s\n', results_dir);


% --- Feature extraction + simple classification analysis (small extra part) ---
fprintf('Running small heartbeat feature analysis and classification...\n');
% load train and test CSVs
trainM = readmatrix(fullfile(data_dir,'mitbih_train.csv'));
testM  = readmatrix(fullfile(data_dir,'mitbih_test.csv'));
train_snips = double(trainM(:,1:end-1)); train_labels = trainM(:,end);
test_snips  = double(testM(:,1:end-1));  test_labels  = testM(:,end);

load(fullfile(results_dir,'final_detection.mat'),'r_locs','r_vals','peak_found','params');

% detect on test using same params
[test_r_locs, test_r_vals, test_peak_found] = detect_r_peaks(test_snips, params);

% compute features and run analysis
feature_analysis(train_snips, train_labels, r_locs, test_snips, test_labels, test_r_locs, params, fullfile(results_dir,'features'));
