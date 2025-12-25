function features = extract_features(snips, r_locs, params)

if nargin < 3, params = struct(); end
if ~isfield(params,'fs'), params.fs = 360; end
fs = params.fs;
[N,L] = size(snips);

% Preallocate
R_amp = nan(N,1);
QRS_width = nan(N,1);
ptp = nan(N,1);
area_qrs = nan(N,1);
rms_energy = nan(N,1);
sk = nan(N,1);
ku = nan(N,1);
spec_centroid = nan(N,1);
spec_bw = nan(N,1);

halfwin = round(0.06 * fs);
for i = 1:N
    x = snips(i,:)';
    r = round(r_locs(i));
    if isnan(r) || r<1 || r>length(x)
        r = round(L/2);
    end
    % bandpass for amplitude/width (reuse same default b,a as pipeline)
    try
        % design filters locally (cheap) - same as detector
        [b,a] = butter(2, [5 25]/(fs/2), 'bandpass');
        xf = filtfilt(b,a,x);
    catch
        xf = x;
    end
    % R amplitude at detected index (signed)
    R_amp(i) = xf(r);
    peakmag = abs(xf(r));
    thr = 0.5 * peakmag; % 50% threshold
    left = r; while left>1 && abs(xf(left)) >= thr, left = left - 1; end
    right = r; while right<length(xf) && abs(xf(right)) >= thr, right = right + 1; end
    QRS_width(i) = (right - left + 1) / fs * 1000; % ms
    ptp(i) = max(xf) - min(xf);
    % area under QRS (within +/- halfwin)
    s = max(1, r-halfwin); e = min(L, r+halfwin);
    area_qrs(i) = sum(abs(xf(s:e))) / fs; % approximate area
    rms_energy(i) = sqrt(mean(xf.^2));
    sk(i) = skewness(xf);
    ku(i) = kurtosis(xf);
    % spectral features (FFT)
    Nfft = 256;
    X = abs(fft(xf, Nfft));
    freqs = (0:Nfft-1)/Nfft*fs;
    % only use positive freqs up to Nyquist
    pos = 1:floor(Nfft/2);
    P = X(pos);
    Psum = sum(P) + eps;
    spec_centroid(i) = sum(freqs(pos)' .* P) / Psum;
    spec_bw(i) = sqrt(sum(((freqs(pos)' - spec_centroid(i)).^2) .* P) / Psum);
end

features = table(R_amp, QRS_width, ptp, area_qrs, rms_energy, sk, ku, spec_centroid, spec_bw);
% add variable names clearly
features.Properties.VariableNames = {'R_amp','QRS_width_ms','PeakToPeak','QRS_area','RMS','Skewness','Kurtosis','SpecCentroid_Hz','SpecBandwidth_Hz'};
end
