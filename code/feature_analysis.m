function feature_analysis(train_snips, train_labels, train_r, test_snips, test_labels, test_r, params, results_dir)

if ~exist(results_dir,'dir'), mkdir(results_dir); end
addpath('.');

% 1) extract features for train & test
F_train = extract_features(train_snips, train_r, params);
F_test  = extract_features(test_snips, test_r, params);

F_train.Label = train_labels;
F_test.Label = test_labels;

% 2) save features to CSV
writetable(F_train, fullfile(results_dir,'features_train.csv'));
writetable(F_test, fullfile(results_dir,'features_test.csv'));

% 3) per-class summary (mean/std) for top features
vars = F_train.Properties.VariableNames(1:end-1); % exclude Label
uniq = unique(train_labels);
summary_table = table();

for k = 1:length(uniq)
    idx = train_labels == uniq(k);
    % compute means and stds as numeric arrays
    mu_tbl = varfun(@mean, F_train(idx, vars));
    sd_tbl = varfun(@std,  F_train(idx, vars));
    mu_arr = table2array(mu_tbl);
    sd_arr = table2array(sd_tbl);

    mean_names = strcat(vars, '_mean');
    std_names  = strcat(vars, '_std');

    Tmean = array2table(mu_arr, 'VariableNames', mean_names);
    Tstd  = array2table(sd_arr, 'VariableNames', std_names);

    row = [table(uniq(k), 'VariableNames', {'Label'}), Tmean, Tstd];

    summary_table = [summary_table; row];
end

writetable(summary_table, fullfile(results_dir,'feature_summary_by_label.csv'));

% 4) boxplots for a few informative features (R_amp, QRS_width_ms, RMS)
figure('Visible','off','Position',[100 100 1200 400]);
subplot(1,3,1); boxplot(F_train.R_amp, F_train.Label); title('R amplitude by label'); xlabel('Label'); ylabel('R amp');
subplot(1,3,2); boxplot(F_train.QRS_width_ms, F_train.Label); title('QRS width (ms) by label'); xlabel('Label'); ylabel('ms');
subplot(1,3,3); boxplot(F_train.RMS, F_train.Label); title('RMS energy by label'); xlabel('Label'); ylabel('RMS');
saveas(gcf, fullfile(results_dir,'feature_boxplots.png'));

% 5) PCA (2D) on normalized features for visualization
X = table2array(F_train(:,vars));
mu = mean(X); s = std(X);
Xn = (X - mu) ./ (s + eps);
[coeff, score] = pca(Xn);
figure('Visible','off'); gscatter(score(:,1), score(:,2), train_labels); title('PCA (train)'); xlabel('PC1'); ylabel('PC2'); legend('Location','best');
saveas(gcf, fullfile(results_dir,'pca_train.png'));

% 6) Simple classification: use linear discriminant (LDA) or SVM
Xtr = Xn;
Ytr = train_labels;
Xte = (table2array(F_test(:,vars)) - mu) ./ (s + eps);
Yte = test_labels;

model = fitcecoc(Xtr, Ytr, 'Learners', templateSVM('KernelFunction','linear','Standardize',true));
Ypred = predict(model, Xte);

% 7) evaluation: accuracy, confusion matrix, per-class precision/recall/F1
acc = mean(Ypred == Yte);
C = confusionmat(Yte, Ypred);
numClasses = size(C,1);
prec = zeros(numClasses,1); rec = zeros(numClasses,1); f1 = zeros(numClasses,1);
for k = 1:numClasses
    TP = C(k,k);
    FP = sum(C(:,k)) - TP;
    FN = sum(C(k,:)) - TP;
    prec(k) = TP / (TP + FP + eps);
    rec(k) = TP / (TP + FN + eps);
    f1(k) = 2 * prec(k) * rec(k) / (prec(k) + rec(k) + eps);
end

T = table((0:numClasses-1)', prec, rec, f1, 'VariableNames', {'Label','Precision','Recall','F1'});
writetable(T, fullfile(results_dir,'classification_metrics_by_label.csv'));
fid = fopen(fullfile(results_dir,'classification_summary.txt'),'w');
fprintf(fid,'Overall accuracy = %.4f\n', acc);
fprintf(fid,'Confusion matrix:\n');
fclose(fid);
save(fullfile(results_dir,'confusion_matrix.mat'),'C','acc');

% confusion plot
figure('Visible','off'); confusionchart(C); title(sprintf('Confusion matrix (accuracy=%.3f)', acc));
saveas(gcf, fullfile(results_dir,'confusion_matrix.png'));

% print to console
fprintf('Classification accuracy on test set: %.4f\n', acc);
disp(T);

end
